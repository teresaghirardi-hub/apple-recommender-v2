"""
monitor.py - Drift detection using statistical comparison.
Compares new prediction data against the training reference dataset.

Usage:
    python 04-monitoring/monitor.py --current 04-monitoring/data/predictions.csv
"""

import argparse
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    'product_name', 'category', 'color',
    'customer_age_group', 'region', 'country', 'city'
]


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def run_drift_report(current_path, output_path):
    logger.info("Loading reference data from raw dataset")
    reference = pd.read_csv('data/apple_sales.csv')[FEATURE_COLS]

    logger.info(f"Loading current data from {current_path}")
    current = pd.read_csv(current_path)[FEATURE_COLS]

    logger.info(f"Reference: {len(reference):,} rows | Current: {len(current):,} rows")

    fig, axes = plt.subplots(len(FEATURE_COLS), 2, figsize=(14, len(FEATURE_COLS) * 3))

    for i, col in enumerate(FEATURE_COLS):
        ref_counts = reference[col].value_counts(normalize=True).head(10)
        cur_counts = current[col].value_counts(normalize=True).head(10)

        ref_counts.plot(kind='bar', ax=axes[i][0], color='#0071e3', alpha=0.7)
        cur_counts.plot(kind='bar', ax=axes[i][1], color='#34c759', alpha=0.7)

        axes[i][0].set_title(f'{col} — Reference', fontsize=10)
        axes[i][1].set_title(f'{col} — Current', fontsize=10)
        axes[i][0].tick_params(axis='x', rotation=30)
        axes[i][1].tick_params(axis='x', rotation=30)

    plt.suptitle('Drift Monitoring Report — Reference vs Current',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report_path = output_path.replace('.html', '.png')
    plt.savefig(report_path, dpi=100, bbox_inches='tight')
    logger.info(f"Drift report saved to {report_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_drift_report(
        current_path=args.current,
        output_path=cfg["monitoring"]["report_output_path"],
    )
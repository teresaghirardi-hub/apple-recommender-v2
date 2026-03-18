"""
simulate.py - Simulates incoming prediction requests and saves them
as a CSV for drift monitoring.

Usage:
    python 04-monitoring/simulate.py
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Simulate 500 new incoming requests
n = 500

# Slightly shift distributions to simulate drift
data = {
    "product_name": np.random.choice([
        "iPhone 15", "iPhone 15 Pro", "MacBook Air", "MacBook Pro",
        "iPad Pro", "iPad", "Apple Watch Series 9", "AirPods Pro",
        "Mac mini", "iMac"
    ], n, p=[0.2, 0.15, 0.15, 0.15, 0.1, 0.08, 0.07, 0.05, 0.03, 0.02]),
    "category": np.random.choice(
        ["iPhone", "Mac", "iPad", "Apple Watch", "AirPods", "Accessories"],
        n, p=[0.35, 0.30, 0.15, 0.08, 0.07, 0.05]
    ),
    "color": np.random.choice([
        "Black", "White", "Silver", "Gold", "Space Gray", "Midnight", "Starlight"
    ], n),
    "customer_age_group": np.random.choice(
        ["18\u201324", "25\u201334", "35\u201344", "45\u201354", "55+"],
        n, p=[0.1, 0.35, 0.30, 0.15, 0.10]
    ),
    "region": np.random.choice(
        ["North America", "Europe", "Asia", "South America",
         "Oceania", "Africa", "Middle East"],
        n, p=[0.40, 0.30, 0.15, 0.07, 0.04, 0.02, 0.02]
    ),
    "country": np.random.choice(
        ["United States", "United Kingdom", "Germany", "France", "Spain"],
        n, p=[0.50, 0.20, 0.15, 0.10, 0.05]
    ),
    "city": np.random.choice(
        ["New York", "London", "Berlin", "Paris", "Madrid"],
        n, p=[0.40, 0.25, 0.15, 0.12, 0.08]
    ),
}

df = pd.DataFrame(data)

os.makedirs("04-monitoring/data", exist_ok=True)
df.to_csv("04-monitoring/data/predictions.csv", index=False)
print(f"Saved {n} simulated predictions to 04-monitoring/data/predictions.csv")
print(df.head())
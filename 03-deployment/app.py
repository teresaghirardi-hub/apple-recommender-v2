"""
app.py - Multi-page Streamlit app for Apple Segment Classifier
Run from root: streamlit run 03-deployment/app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add 03-deployment to path so we can import predict.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

st.set_page_config(
    page_title="Apple Segment Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .block-container { padding: 2rem 2.5rem; }
    [data-testid="stSidebar"] { background: #1a1a2e; }
    [data-testid="stSidebar"] * { color: white !important; }
    .card { background: white; border-radius: 20px; padding: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin-bottom: 1rem; }
    .metric-card { background: white; border-radius: 16px; padding: 1.2rem; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
    .metric-value { font-size: 2rem; font-weight: 800; color: #667eea; margin: 0; }
    .metric-label { color: #888; font-size: 0.85rem; margin: 0; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

SEGMENT_COLORS = {
    "Individual": "#f5576c",
    "Business":   "#4facfe",
    "Education":  "#43e97b",
    "Government": "#fa709a",
}

SEGMENT_EMOJIS = {
    "Individual": "👤",
    "Business":   "💼",
    "Education":  "🎓",
    "Government": "🏛️",
}

SEGMENT_PRODUCTS = {
    "Individual":  ["iPhone 15", "AirPods Pro", "Apple Watch Series 9"],
    "Business":    ["MacBook Pro M3", "iPad Pro", "Mac mini M2"],
    "Education":   ["MacBook Air M2", "iPad 10th Gen", "Apple Pencil"],
    "Government":  ["iPhone 15 Pro", "Mac mini M2", "MacBook Pro M3"],
}

SEGMENT_OFFERS = {
    "Individual":  "Trade in your old device and save up to $200.",
    "Business":    "Volume licensing available — contact our Business team.",
    "Education":   "Education bundle: save up to $300 with student verification.",
    "Government":  "GSA Schedule pricing available. Request a quote.",
}


@st.cache_resource
def get_pipeline():
    from predict import load_pipeline
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'pipeline.pkl')
    return load_pipeline(model_path)

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'apple_sales.csv')
    return pd.read_csv(data_path)

# Default values
product_name = "iPhone 15"
category = "iPhone"
color = "Black"
age_group = "18\u201324"
region = "North America"
country = "United States"
city = "New York"
submitted = False
# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍎 Apple Classifier")
    st.markdown("---")
    page = st.radio("", ["🏠 Homepage", "📊 Analytics", "🤖 Model Info"])
    st.markdown("---")

if page == "🏠 Homepage":   
    st.markdown("##### Tell us about you")
    product_name = st.selectbox("Interested in", [
        "iPhone 15", "iPhone 15 Pro", "MacBook Air", "MacBook Pro",
        "iPad Pro", "iPad", "Apple Watch Series 9", "AirPods Pro", "Mac mini", "iMac",
    ])
    category = st.selectbox("Category", [
        "iPhone", "Mac", "iPad", "Apple Watch", "AirPods", "Accessories"
    ])
    color = st.selectbox("Color", [
        "Black", "White", "Silver", "Gold", "Space Gray", "Midnight", "Starlight"
    ])
    age_group = st.selectbox("Age group", [
        "18\u201324", "25\u201334", "35\u201344", "45\u201354", "55+"
    ])
    region = st.selectbox("Region", [
        "North America", "Europe", "Asia", "South America",
        "Oceania", "Africa", "Middle East"
    ])
    country = st.text_input("Country", value="United States")
    city = st.text_input("City", value="New York")

    st.markdown("---")
    submitted = st.button("✦ Show my homepage", use_container_width=True)
# ── PAGE 1: Homepage ───────────────────────────────────────────────────────────
if page == "🏠 Homepage":
    try:
        pipeline = get_pipeline()
        from predict import predict_segment

        input_data = {
            "product_name": product_name, "category": category, "color": color,
            "customer_age_group": age_group, "region": region,
            "country": country, "city": city,
        }

        if not submitted:
            st.markdown("""
            <div style="background:#f8f9fa;border-radius:20px;padding:3rem;text-align:center;">
                <div style="font-size:4rem">🍎</div>
                <h2 style="color:#667eea;">Welcome to Apple Classifier</h2>
                <p style="color:#888;">Fill in the questionnaire on the left and click <b>Show my homepage</b></p>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        

        result  = predict_segment(pipeline, input_data)
        segment = result["segment"]
        proba   = result["probabilities"]
        color_h = SEGMENT_COLORS[segment]

        # Hero
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, {color_h}cc, {color_h}88);
                    border-radius:20px;padding:2.5rem;color:white;margin-bottom:1.5rem;">
            <div style="font-size:3rem">{SEGMENT_EMOJIS[segment]}</div>
            <h1 style="font-size:2.5rem;font-weight:800;margin:0.5rem 0;color:white;">
                Your segment: {segment}
            </h1>
            <p style="font-size:1.1rem;opacity:0.9;margin:0">{SEGMENT_OFFERS[segment]}</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 🛍️ Recommended for you")
            c1, c2, c3 = st.columns(3)
            for c, product in zip([c1, c2, c3], SEGMENT_PRODUCTS[segment]):
                with c:
                    st.markdown(f"""
                    <div style="background:#f8f9fa;border-radius:14px;padding:1.2rem;
                                text-align:center;border:2px solid {color_h}33;">
                        <div style="font-size:2rem">💻</div>
                        <p style="font-weight:600;font-size:0.9rem;margin:0.5rem 0 0 0">{product}</p>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Confidence")
            for seg, prob in sorted(proba.items(), key=lambda x: -x[1]):
                w = int(prob * 100)
                bold = "font-weight:700;" if seg == segment else "color:#999;"
                st.markdown(f"""
                <div style="margin-bottom:0.6rem;">
                    <div style="display:flex;justify-content:space-between;{bold}font-size:0.85rem;">
                        <span>{SEGMENT_EMOJIS[seg]} {seg}</span><span>{w}%</span>
                    </div>
                    <div style="background:#f0f0f0;border-radius:8px;height:8px;overflow:hidden;">
                        <div style="width:{w}%;height:8px;background:{SEGMENT_COLORS[seg]};border-radius:8px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Make sure you ran `python 02-experiment-tracking/train.py` from the root first.")

# ── PAGE 2: Analytics ──────────────────────────────────────────────────────────
elif page == "📊 Analytics":
    st.markdown("## 📊 Analytics Dashboard")
    st.caption("Insights from the Apple Sales Dataset")

    try:
        df = load_data()

        # KPI cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><p class="metric-value">{len(df):,}</p><p class="metric-label">Total Records</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><p class="metric-value">${df["revenue_usd"].sum()/1e6:.1f}M</p><p class="metric-label">Total Revenue</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><p class="metric-value">{df["country"].nunique()}</p><p class="metric-label">Countries</p></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><p class="metric-value">{df["product_name"].nunique()}</p><p class="metric-label">Products</p></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            counts = df['customer_segment'].value_counts()
            colors = list(SEGMENT_COLORS.values())
            ax.pie(counts.values, labels=counts.index, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Segment Distribution', fontweight='bold')
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            rev = df.groupby('category')['revenue_usd'].sum().sort_values()
            ax.barh(rev.index, rev.values/1e6, color='#667eea')
            ax.set_xlabel('Revenue (M USD)')
            ax.set_title('Revenue by Category', fontweight='bold')
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            top10 = df.groupby('product_name')['revenue_usd'].sum().sort_values(ascending=False).head(10)
            ax.barh(top10.index, top10.values/1e6, color='#764ba2')
            ax.set_xlabel('Revenue (M USD)')
            ax.set_title('Top 10 Products', fontweight='bold')
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            age_seg = df.groupby(['customer_age_group', 'customer_segment']).size().unstack()
            age_seg.plot(kind='bar', ax=ax, color=list(SEGMENT_COLORS.values()))
            ax.set_title('Age Group by Segment', fontweight='bold')
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=15)
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Could not load data: {e}")


# ── PAGE 3: Model Info ─────────────────────────────────────────────────────────
elif page == "🤖 Model Info":
    st.markdown("## 🤖 Model Information")
    st.caption("How the segment classifier works")

    (tab1,) = st.tabs(["🎯 Segment Classifier"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="card">
                <h4>🧠 Algorithm</h4>
                <p><b>Type:</b> Classification</p>
                <p><b>Model:</b> Random Forest Classifier</p>
                <p><b>Training data:</b> 11,500 Apple sales records</p>
                <p><b>Classes:</b> Individual, Business, Education, Government</p>
                <p><b>Split:</b> 80% train / 20% test (stratified)</p>
                <p><b>Tuning:</b> Grid search over n_estimators, max_depth, min_samples_split</p>
                <p><b>Tracking:</b> MLflow experiment logging</p>
            </div>
            <div class="card">
                <h4>⚙️ Preprocessing Pipeline</h4>
                <p>🔵 <b>Target Encoding</b> → country, city, product_name, color<br>
                <span style="color:#888;font-size:0.85rem;">High cardinality — replaces each value with avg target</span></p>
                <p>🟢 <b>One-Hot Encoding</b> → category, region<br>
                <span style="color:#888;font-size:0.85rem;">Low cardinality — creates a binary column per value</span></p>
                <p>🟡 <b>Ordinal Encoding</b> → customer_age_group<br>
                <span style="color:#888;font-size:0.85rem;">Ordered — respects natural age progression</span></p>
            </div>
            <div class="card">
                <h4>📊 Metrics</h4>
                <p><b>Accuracy:</b> ~26%</p>
                <p><b>Weighted F1:</b> ~26%</p>
                <p><b>Baseline (random):</b> 25%</p>
                <p style="color:#888;font-size:0.85rem;">⚠️ Low accuracy is expected — the dataset is synthetic
                with randomly assigned segments. With real Apple
                clickstream data this pipeline would perform significantly better.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
                <h4>📋 Features Used (7)</h4>
            """, unsafe_allow_html=True)
            features = [
                ("product_name", "43 unique values", "High cardinality → Target Encoding", "#f5576c"),
                ("category", "6 categories", "Low cardinality → One-Hot Encoding", "#4facfe"),
                ("color", "20 colors", "High cardinality → Target Encoding", "#43e97b"),
                ("customer_age_group", "5 age groups", "Ordered → Ordinal Encoding", "#fa709a"),
                ("region", "8 regions", "Low cardinality → One-Hot Encoding", "#667eea"),
                ("country", "47 countries", "High cardinality → Target Encoding", "#764ba2"),
                ("city", "514 cities", "High cardinality → Target Encoding", "#f093fb"),
            ]
            for feat, desc, enc, c in features:
                st.markdown(f"""
                <div style="display:flex;align-items:flex-start;gap:0.75rem;margin-bottom:0.75rem;">
                    <div style="width:10px;height:10px;border-radius:50%;background:{c};flex-shrink:0;margin-top:4px;"></div>
                    <div>
                        <b>{feat}</b> <span style="color:#888;font-size:0.85rem;">— {desc}</span><br>
                        <span style="color:#aaa;font-size:0.78rem;">{enc}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
                <h4>🗑️ Excluded Columns</h4>
                <p>❌ <b>Data leakage:</b> units_sold, revenue_usd, discounted_price_usd, return_status</p>
                <p>❌ <b>Noise:</b> sales_channel, payment_method (uniform across segments)</p>
                <p>❌ <b>Missing data:</b> previous_device_os (70%), customer_rating (29%)</p>
                <p>❌ <b>Identifiers:</b> sale_id, sale_date, currency</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-top:1rem;">
        <h4>🚀 Tech Stack</h4>
        <div style="display:flex;gap:1rem;flex-wrap:wrap;">
            <span style="background:#667eea22;color:#667eea;padding:4px 12px;border-radius:20px;font-weight:600;">🔬 scikit-learn</span>
            <span style="background:#f5576c22;color:#f5576c;padding:4px 12px;border-radius:20px;font-weight:600;">📊 MLflow</span>
            <span style="background:#43e97b22;color:#43e97b;padding:4px 12px;border-radius:20px;font-weight:600;">⚡ FastAPI</span>
            <span style="background:#4facfe22;color:#4facfe;padding:4px 12px;border-radius:20px;font-weight:600;">🎨 Streamlit</span>
            <span style="background:#fa709a22;color:#fa709a;padding:4px 12px;border-radius:20px;font-weight:600;">🐳 Docker</span>
            <span style="background:#764ba222;color:#764ba2;padding:4px 12px;border-radius:20px;font-weight:600;">⚙️ GitHub Actions</span>
            <span style="background:#f5a62322;color:#f5a623;padding:4px 12px;border-radius:20px;font-weight:600;">☁️ Render</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
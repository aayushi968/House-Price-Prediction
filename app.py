import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Bengaluru House Price Predictor", layout="wide")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
    color: #1f2937 !important;
}

.stApp {
    background: linear-gradient(135deg, #f8fbff 0%, #e8f4fd 100%);
}

/* Main Title */
h1 {
    color: #1e3a8a !important;
    font-weight: 700 !important;
    text-align: center;
}

/* Headers */
h2 {
    color: #1e3a8a !important;
    font-weight: 600 !important;
}

h3 {
    color: #1e3a8a !important;
    font-weight: 600 !important;
}

/* Main text content */
p, div, span {
    color: #1f2937 !important;
}

/* Make ALL widget labels visible */
[data-testid="stWidgetLabel"] {
    color: #1e3a8a !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}

/* Selectbox, Slider, Inputs styling */
.stSelectbox > div > div,
.stSlider > div > div {
    border-radius: 12px !important;
    border: 2px solid #3b82f6 !important;
    background: #ffffff !important;
    box-shadow: 0 2px 8px rgba(59,130,246,0.15) !important;
    color: #1f2937 !important;
}

/* Dropdown options styling */
[data-baseweb="select"] {
    color: #1f2937 !important;
}

[data-baseweb="select"] div {
    color: #1f2937 !important;
}

[data-baseweb="popover"] {
    color: #1f2937 !important;
}

[role="option"] {
    color: #1f2937 !important;
    background-color: #ffffff !important;
}

[role="option"]:hover {
    background-color: #e0f2fe !important;
    color: #1e3a8a !important;
}

/* Dropdown list items */
.stSelectbox ul li {
    color: #1f2937 !important;
}

.stSelectbox ul li:hover {
    background-color: #e0f2fe !important;
    color: #1e3a8a !important;
}

/* Input field text */
input {
    color: #1f2937 !important;
}

input::placeholder {
    color: #9ca3af !important;
}

/* Info box styling */
.stAlert {
    color: #1f2937 !important;
    background-color: #e0f2fe !important;
    border: 1px solid #0284c7 !important;
}

/* Button styling */
.stButton > button {
    border-radius: 12px !important;
    border: 2px solid #1e3a8a !important;
    background: #1e3a8a !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    padding: 0.75rem 2rem !important;
    transition: 0.3s ease;
}

.stButton > button:hover {
    background: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
}

/* Metric Card Styling */
[data-testid="metric-container"] {
    background: white;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
    color: #1e3a8a;
}

[data-testid="metric-container"] p {
    color: #1f2937 !important;
}

/* Sidebar styling */
.stSidebar [data-testid="stMarkdownContainer"] {
    color: #1f2937 !important;
}

.stSidebar h2 {
    color: #1e3a8a !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    loaded = pickle.load(open('house_model.pkl', 'rb'))
    return loaded['model'], loaded['scaler'], loaded['le'], loaded['top_locations']

model, scaler, le, top_locations = load_model()

def predict_house_price(location, total_sqft, bhk, bath, balcony):
    input_df = pd.DataFrame({
        'location': [location],
        'total_sqft': [total_sqft],
        'bhk': [bhk],
        'bath': [bath],
        'has_balcony': [1 if balcony > 0 else 0]
    })

    input_df['location'] = input_df['location'].apply(
        lambda x: x if x in top_locations else 'other'
    )

    input_df['location'] = le.transform(input_df['location'])

    features = ['location', 'total_sqft', 'bhk', 'bath', 'has_balcony']
    input_scaled = scaler.transform(input_df[features])

    return model.predict(input_scaled)[0]

locations_data = {
    'Whitefield': 350,
    'Koramangala': 550,
    'Electronic City Phase II': 180,
    'Indiranagar': 650,
    'other': 250
}

sample_houses = pd.DataFrame({
    'location': ['Whitefield', 'Koramangala', 'Indiranagar', 'Electronic City Phase II'],
    'avg_price_cr': [3.5, 5.5, 6.5, 1.8]
})

st.title("Bengaluru House Price Predictor")

with st.sidebar:
    st.header("Model Performance")
    st.info("**RMSE:** ±₹12 Lakh\n\n**R² Score:** 0.65\n\n**Features:** Location + Size + Amenities")


st.markdown("## Bengaluru Market Snapshot")

col1, col2 = st.columns(2)

with col1:
    fig_pie = px.pie(
        values=list(locations_data.values()),
        names=list(locations_data.keys()),
        hole=0.4,
        title="Location Price Share"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_bar = px.bar(
        sample_houses,
        x='location',
        y='avg_price_cr',
        title="Average Prices by Location",
        color='avg_price_cr',
        color_continuous_scale='blugrn'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("## Enter Property Details")

col1, col2 = st.columns(2, gap="large")

with col1:
    location = st.selectbox(
        "Location",
        ['Whitefield', 'Koramangala', 'Electronic City Phase II', 'Indiranagar', 'other']
    )
    total_sqft = st.slider("Total Sqft", 300, 5000, 1500, 50)
    bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])

with col2:
    bath = st.selectbox("Bathrooms", [1, 2, 3, 4])
    balcony = st.selectbox("Balconies", [0, 1, 2, 3])


if st.button("Get Price Prediction", use_container_width=True):

    with st.spinner("Analyzing market data..."):
        price = predict_house_price(location, total_sqft, bhk, bath, balcony)

    st.markdown("---")
    st.markdown("## Prediction Results")

    center_col1, center_col2, center_col3 = st.columns([1, 2, 1])

    with center_col2:
        if price >= 100: 
            display_price = f"₹{price/100:.2f} Cr"
        else:
            display_price = f"₹{price:.2f} Lakh"
        st.markdown(f"""
                    <div style='
                    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
                    border-radius: 15px;
                    padding: 30px;
    text-align: center;
    box-shadow: 0 8px 30px rgba(30, 58, 138, 0.2);
    border: 2px solid #60a5fa;
'>
    <p style='font-size: 2rem; font-weight: 600; margin: 0;'> Estimated Price</p>
    <h1 style='font-size: 3.5rem; font-weight: 800; margin: 10px 0;'>{display_price}</h1>
    <p style='font-size: 0.9rem; color: #0284c7; margin: 0;'>Based on Market Analysis</p>
</div>
""", unsafe_allow_html=True)

        st.info(f"""
        **Property Details**
        - Location: {location}
        - Area: {total_sqft:,} sqft
        - {bhk} BHK | {bath} Bath | {balcony} Balcony
        """)

    st.markdown("### Your Price vs Market")

    market_avg = locations_data.get(location, 300)

    comparison_data = pd.DataFrame({
        'Category': ['Your Property', 'Location Average'],
        'Price_Cr': [price/100, market_avg/100]
    })

    fig_compare = px.bar(
        comparison_data,
        x='Category',
        y='Price_Cr',
        color='Price_Cr',
        color_continuous_scale='blues',
        title=f"{location} Price Comparison"
    )

    fig_compare.update_layout(showlegend=False)
    st.plotly_chart(fig_compare, use_container_width=True)

    st.markdown("### Price Breakdown")

    price_per_sqft = (price/100) / total_sqft * 10000

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Price per Sqft", f"₹{price_per_sqft:.0f}")

    with col2:
        st.metric("Total Area", f"{total_sqft:,} sqft")

st.markdown("---")

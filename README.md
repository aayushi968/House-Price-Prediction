# Bengaluru House Price Predictor

A Streamlit web application for real-time Bengaluru house price predictions using a pre-trained linear regression model. Features interactive Plotly visualizations, clean professional UI, and comprehensive market analysis.

##Features

- **Real-time Predictions**: Instant price estimates for Bengaluru properties
- **Interactive Charts**: Location price distribution, market comparisons, price-per-sqft analysis
- **Key Features**: Location, total sqft (300-5000), BHK (1-5), bathrooms (1-4), balconies (0-3)
- **Professional UI**: Clean light blue/white design with Poppins font
- **Model Metrics**: R² = 0.65 accuracy, ±₹12 lakh error margin

##Folder Structure
bengaluru-house-predictor/
├── app.py # Streamlit web application
├── house_model.pkl # Pre-trained linear regression model
├── kaggle_notebook.ipynb # Complete ML development workflow
├── README.md # This file


## Quick Start (Local)

1. **Clone & Install** Requirements - pip install streamlit==1.38.0 pandas==2.2.2 numpy==1.26.4 plotly==5.22.0 scikit-learn==1.5.1
2. **Run App** - python -m streamlit run app.py
3. **Access** : http://localhost:8501

##Kaggle Notebook
Complete ML pipeline included:
1. Data preprocessing & feature engineering
2. Location encoding + feature scaling
3. Linear regression training & evaluation
4. Residual analysis & model validation
5. Model export (house_model.pkl)







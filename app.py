import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page configuration for premium feel
st.set_page_config(page_title="YouTube Ad Revenue Modeler", page_icon="📈", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #ff0000;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #cc0000;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    .header-style {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#ff0000, #ff4b4b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }
    .subheader-style {
        text-align: center;
        color: #6c757d;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='header-style'>YouTube Ad Revenue Modeler</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader-style'>Predict your video's potential earnings using Machine Learning</p>", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    model_path = 'model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model_pipeline = load_model()

# Sidebar for User Inputs
st.sidebar.header("📊 Video Metrics Input")
st.sidebar.write("Adjust the inputs to set your video's performance metrics:")

views = st.sidebar.number_input("Views", min_value=0, max_value=100000000, value=10000, step=1000)
likes = st.sidebar.number_input("Likes", min_value=0, max_value=10000000, value=1000, step=100)
comments = st.sidebar.number_input("Comments", min_value=0, max_value=1000000, value=200, step=50)
watch_time_minutes = st.sidebar.number_input("Watch Time (Minutes)", min_value=0.0, value=15000.0, step=1000.0)
video_length_minutes = st.sidebar.number_input("Video Length (Minutes)", min_value=0.0, value=10.0, step=1.0)
subscribers = st.sidebar.number_input("Channel Subscribers", min_value=0, value=500000, step=10000)

st.sidebar.markdown("---")
category = st.sidebar.selectbox("Category Code", options=[1, 2, 3, 4, 5, 6])
country = st.sidebar.selectbox("Country Code", options=[1, 2, 3, 4, 5, 6])

# Predict Button
predict_btn = st.sidebar.button("Predict Ad Revenue 🚀")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Model Prediction")
    if model_pipeline is None:
        st.warning("⚠️ Model file (`model.pkl`) not found. Please run the Jupyter notebook to train and save the model first.")
    else:
        if predict_btn:
            # Prepare input dataframe
            input_df = pd.DataFrame({
                'views': [views],
                'likes': [likes],
                'comments': [comments],
                'watch_time_minutes': [watch_time_minutes],
                'video_length_minutes': [video_length_minutes],
                'subscribers': [subscribers],
                'category': [category],
                'country': [country]
            })
            
            # Predict
            try:
                prediction = model_pipeline.predict(input_df)[0]
                
                # Display Prediction
                st.markdown(f"""
                <div class='metric-card'>
                    <h2 style='color: #6c757d; font-size: 1.5rem;'>Estimated Ad Revenue</h2>
                    <h1 style='color: #28a745; font-size: 4rem; margin: 10px 0;'>${max(0, prediction):,.2f}</h1>
                    <p style='color: #adb5bd;'>Based on current model predictions</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.success("Prediction generated successfully!")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.info("👈 Enter video metrics in the sidebar and click **Predict Ad Revenue** to see the estimation.")
            
    st.markdown("---")
    st.markdown("### 📈 Project Insights")
    st.write("This app uses a Machine Learning model trained on over 120,000 YouTube video records to accurately predict ad revenue. The model analyzes historical performance metrics like views, engagement (likes/comments), and video properties.")

with col2:
    st.markdown("### Feature Importance")
    if model_pipeline is not None:
        try:
            # Try to get feature importances from pipeline
            # The model is likely wrapped in a pipeline: Pipeline(steps=[('preprocessor', ...), ('regressor', model)])
            model = model_pipeline.named_steps['regressor']
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # We need to get feature names from the preprocessor if possible
                try:
                    preprocessor = model_pipeline.named_steps['preprocessor']
                    num_features = ['views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes', 'subscribers']
                    # for categorical we passed them as is, or OHE. Assuming passing as numeric (category code)
                    feature_names = num_features + ['category', 'country']
                except:
                    feature_names = ['views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes', 'subscribers', 'category', 'country']
                
                # Match lengths just in case
                if len(importances) == len(feature_names):
                    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                    imp_df = imp_df.sort_values(by='Importance', ascending=True)
                    
                    fig, ax = plt.subplots(figsize=(6, 8))
                    sns.barplot(x='Importance', y='Feature', data=imp_df, ax=ax, palette='viridis')
                    ax.set_title("Relative Impact of Video Metrics", fontsize=14)
                    ax.set_xlabel("Importance Score")
                    ax.set_ylabel("")
                    sns.despine(left=True, bottom=True)
                    st.pyplot(fig)
                else:
                    st.write("Feature importances array length doesn't match features.")
            else:
                st.info("The selected model does not support feature importance visualization.")
        except Exception as e:
            st.info(f"Feature importance could not be visualized. Error: {e}")
    else:
        # Placeholder
        st.markdown("<div style='height: 300px; background-color: #e9ecef; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #6c757d;'>Train the model to see feature importances!</div>", unsafe_allow_html=True)

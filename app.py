# streamlit_dashboard.py
# Main Streamlit dashboard file

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import our ML functions
from ML import MLPredictor, find_similar_students, get_confidence_level

# Set page configuration
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title of the dashboard
st.title("📊 Student Performance Dashboard")
st.markdown("---")

# Load data function
@st.cache_data
def load_data():
    # You need to upload your CSV file to the same directory
    df = pd.read_csv('StudentsPerformance.csv')
    
    # Clean column names
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    
    # Convert categorical columns
    categorical_cols = ['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    # Add average score
    df['average_score'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1)
    
    return df

# Initialize ML Predictor
@st.cache_resource
def initialize_ml_model(df):
    """Initialize and train ML model (cached for performance)"""
    predictor = MLPredictor(df)
    model, metrics, test_data = predictor.prepare_and_train()
    return predictor

# Load the data
try:
    df = load_data()
    st.success(f"✅ Data loaded successfully! {len(df)} students in dataset")
    
    # Initialize ML model
    ml_predictor = initialize_ml_model(df)
    st.success("🤖 ML model trained successfully!")
    
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.error("Make sure 'StudentsPerformance.csv' is in the same directory and ml_functions.py is available.")
    st.stop()

# Sidebar for navigation
st.sidebar.title("🔍 Dashboard Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Overview", "Score Distributions", "Gender Analysis", "Education Impact", "Test Prep Analysis", "🤖 ML Prediction"]
)

# Overview Page
if page == "Overview":
    st.header("📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        st.metric("Average Math Score", f"{df['math_score'].mean():.1f}")
    with col3:
        st.metric("Average Reading Score", f"{df['reading_score'].mean():.1f}")
    with col4:
        st.metric("Average Writing Score", f"{df['writing_score'].mean():.1f}")
    
    st.subheader("📊 Data Sample")
    st.dataframe(df.head(10))
    
    st.subheader("📈 Basic Statistics")
    st.dataframe(df[['math_score', 'reading_score', 'writing_score']].describe())

# Score Distributions Page
elif page == "Score Distributions":
    st.header("📊 Score Distributions")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Math score distribution
    axes[0].hist(df['math_score'], bins=20, color='skyblue', alpha=0.7)
    axes[0].set_title('Math Score Distribution')
    axes[0].set_xlabel('Math Score')
    axes[0].set_ylabel('Number of Students')
    
    # Reading score distribution
    axes[1].hist(df['reading_score'], bins=20, color='orange', alpha=0.7)
    axes[1].set_title('Reading Score Distribution')
    axes[1].set_xlabel('Reading Score')
    axes[1].set_ylabel('Number of Students')
    
    # Writing score distribution
    axes[2].hist(df['writing_score'], bins=20, color='green', alpha=0.7)
    axes[2].set_title('Writing Score Distribution')
    axes[2].set_xlabel('Writing Score')
    axes[2].set_ylabel('Number of Students')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("🔗 Score Correlations")
    fig, ax = plt.subplots(figsize=(8, 6))
    correlation = df[['math_score', 'reading_score', 'writing_score']].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    plt.title("Correlation Between Scores")
    st.pyplot(fig)

# Gender Analysis Page
elif page == "Gender Analysis":
    st.header("👥 Gender Analysis")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Box plots for each subject by gender
    sns.boxplot(data=df, x='gender', y='math_score', ax=axes[0], palette='pastel')
    axes[0].set_title('Math Score by Gender')
    
    sns.boxplot(data=df, x='gender', y='reading_score', ax=axes[1], palette='pastel')
    axes[1].set_title('Reading Score by Gender')
    
    sns.boxplot(data=df, x='gender', y='writing_score', ax=axes[2], palette='pastel')
    axes[2].set_title('Writing Score by Gender')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Average scores by gender
    st.subheader("📊 Average Scores by Gender")
    gender_avg = df.groupby('gender')[['math_score', 'reading_score', 'writing_score']].mean().round(1)
    st.dataframe(gender_avg)

# Education Impact Page
elif page == "Education Impact":
    st.header("🎓 Parental Education Impact")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Box plots for each subject by parental education
    sns.boxplot(data=df, x='parental_level_of_education', y='math_score', ax=axes[0])
    axes[0].set_title('Math Score by Parental Education')
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=df, x='parental_level_of_education', y='reading_score', ax=axes[1])
    axes[1].set_title('Reading Score by Parental Education')
    axes[1].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=df, x='parental_level_of_education', y='writing_score', ax=axes[2])
    axes[2].set_title('Writing Score by Parental Education')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Summary table
    st.subheader("📋 Average Scores by Parental Education Level")
    edu_summary = df.groupby('parental_level_of_education')[['math_score', 'reading_score', 'writing_score']].mean().round(1)
    st.dataframe(edu_summary)

# Test Prep Analysis Page
elif page == "Test Prep Analysis":
    st.header("📚 Test Preparation Analysis")
    
    # Test prep impact on scores
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.boxplot(data=df, x='test_preparation_course', y='math_score', ax=axes[0])
    axes[0].set_title('Math Score by Test Prep')
    
    sns.boxplot(data=df, x='test_preparation_course', y='reading_score', ax=axes[1])
    axes[1].set_title('Reading Score by Test Prep')
    
    sns.boxplot(data=df, x='test_preparation_course', y='writing_score', ax=axes[2])
    axes[2].set_title('Writing Score by Test Prep')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Average scores by test prep
    st.subheader("📊 Average Scores by Test Preparation")
    prep_avg = df.groupby('test_preparation_course')[['math_score', 'reading_score', 'writing_score']].mean().round(1)
    st.dataframe(prep_avg)
    
    # Combined analysis: Gender + Test Prep
    st.subheader("🔍 Combined Analysis: Gender & Test Prep")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='gender', y='average_score', hue='test_preparation_course', ax=ax)
    plt.title("Average Score by Gender and Test Prep")
    st.pyplot(fig)

# ML Prediction Page
elif page == "🤖 ML Prediction":
    st.header("🤖 Math Score Prediction with Machine Learning")
    
    # Explain what we're doing
    st.markdown("""
    ### 📚 What is this?
    This uses **Machine Learning** to predict a student's math score based on their characteristics.
    We use a **Random Forest** model - think of it as many decision trees working together to make a prediction.
    
    ### 🔍 How it works:
    1. **Training**: The model learns patterns from existing student data
    2. **Features**: It looks at reading score, writing score, gender, education level, etc.
    3. **Prediction**: Based on these patterns, it predicts the math score
    """)
    
    # Get model metrics and test data
    metrics = ml_predictor.metrics
    test_data = ml_predictor.test_data
    
    # Show model performance
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Accuracy (R²)", f"{metrics['r2']:.3f}")
        st.caption("1.0 = Perfect, 0.8+ = Very Good")
    with col2:
        st.metric("Prediction Error (RMSE)", f"{metrics['rmse']:.1f}")
        st.caption("Average error in points")
    
    # Feature importance
    st.subheader("📊 What Factors Matter Most?")
    importance_data = ml_predictor.get_feature_importance()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.bar(range(len(importance_data['importance'])), importance_data['importance'])
    plt.xticks(range(len(importance_data['importance'])), importance_data['features'], rotation=45)
    plt.title("Which Factors Most Influence Math Scores?")
    plt.ylabel("Importance")
    st.pyplot(fig)
    
    # Prediction vs Actual scatter plot
    st.subheader("🎯 How Good Are Our Predictions?")
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(test_data['y_test'], test_data['y_pred'], alpha=0.5)
    plt.plot([test_data['y_test'].min(), test_data['y_test'].max()], 
             [test_data['y_test'].min(), test_data['y_test'].max()], 'r--', lw=2)
    plt.xlabel("Actual Math Score")
    plt.ylabel("Predicted Math Score")
    plt.title("Predictions vs Reality")
    st.pyplot(fig)
    st.caption("Points closer to the red line = better predictions")
    
    st.markdown("---")
    
    # Interactive Prediction
    st.subheader("🔮 Make Your Own Prediction!")
    st.markdown("Enter student characteristics below to predict their math score:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        reading_score = st.slider("Reading Score", 0, 100, 70)
        writing_score = st.slider("Writing Score", 0, 100, 70)
        gender = st.selectbox("Gender", ["female", "male"])
        lunch = st.selectbox("Lunch Type", ["free/reduced", "standard"])
    
    with col2:
        education_options = ["some high school", "high school", "some college", 
                           "associate's degree", "bachelor's degree", "master's degree"]
        parental_education = st.selectbox("Parental Education Level", education_options)
        test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])
        race_options = ["group A", "group B", "group C", "group D", "group E"]
        race = st.selectbox("Race/Ethnicity", race_options)
    
    # Make prediction when button is clicked
    if st.button("🎯 Predict Math Score!", type="primary"):
        # Prepare input data
        input_data = {
            'reading_score': reading_score,
            'writing_score': writing_score,
            'gender': gender,
            'parental_education': parental_education,
            'test_prep': test_prep,
            'lunch': lunch,
            'race': race
        }
        
        try:
            # Make prediction using our ML predictor
            predicted_score = ml_predictor.predict(input_data)
            
            # Show result
            st.success(f"🎯 **Predicted Math Score: {predicted_score:.1f}**")
            
            # Show confidence level
            confidence, color = get_confidence_level(metrics['r2'])
            st.markdown(f"**Confidence Level:** :{color}[{confidence}] (Model accuracy: {metrics['r2']:.1f})")
            
            # Show similar students
            st.subheader("👥 Similar Students in Dataset")
            similar_students = find_similar_students(df, reading_score, writing_score)
            
            if len(similar_students) > 0:
                avg_math = similar_students['math_score'].mean()
                st.info(f"📊 Students with similar reading/writing scores averaged {avg_math:.1f} in math")
                st.dataframe(similar_students[['math_score', 'reading_score', 'writing_score', 'gender', 'parental_level_of_education']].head())
            else:
                st.info("No similar students found in dataset")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("📊 **Student Performance Dashboard** | Built with Streamlit")
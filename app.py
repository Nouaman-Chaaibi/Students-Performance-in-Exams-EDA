import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# Load the data
try:
    df = load_data()
    st.success(f"✅ Data loaded successfully! {len(df)} students in dataset")
except:
    st.error("❌ Could not load data. Make sure 'StudentsPerformance.csv' is in the same directory.")
    st.stop()

# Sidebar for navigation
st.sidebar.title("🔍 Dashboard Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Overview", "Score Distributions", "Gender Analysis", "Education Impact", "Test Prep Analysis"]
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

# Footer
st.markdown("---")
st.markdown("📊 **Student Performance Dashboard** | Built with Streamlit")
# Students Performance in Exams - EDA, ML Predictions, and Streamlit Dashboard

This project analyzes the *Students Performance in Exams* dataset using **Exploratory Data Analysis (EDA)** and **Machine Learning** techniques. It also includes an **interactive dashboard built with Streamlit** that allows users to explore the data and predict student scores.

🔗 **Live App:** [Open Streamlit Dashboard](https://students-performance-analysis-eda-ml.streamlit.app/)

---

## Dataset

The dataset `StudentsPerformance.csv` contains student scores in three subjects:
- Math
- Reading
- Writing

It also contains categorical features like gender, race/ethnicity, parental level of education, lunch type, and test preparation course.

---

## Features

- ✅ Data cleaning and preprocessing
- 📊 Visualizations:
  - Distribution of scores (histograms)
  - Score comparisons by gender, lunch type, parental education, and test preparation course (boxplots)
  - Correlation heatmap between scores
  - Countplots for categorical features relationships
- 🚨 Outlier detection
- 📈 Summary statistics by groups
- 🤖 Machine Learning:
  - Preprocessing with Label Encoding
  - Train/test split and feature selection
  - Predictive models (Random Forest)
  - Performance metrics (R², MSE)
- 🧠 Interactive ML prediction section in Streamlit
- 🖥️ Interactive dashboard using Streamlit to explore data and make predictions

---

## How to run the app locally

1. Clone the repository:  
  git clone https://github.com/Nouaman-Chaaibi/Students-Performance-in-Exams-EDA.git
2. Navigate to the project directory:  
  cd Students-Performance-in-Exams-EDA
3. (Optional but recommended) Create and activate a virtual environment:  
  python -m venv env  
  env\Scripts\activate
4. Install dependencies:
  pip install -r requirements.txt
5. Run the Streamlit app:
  streamlit run app.py

---

## Dependencies

- pandas
- matplotlib
- seaborn
- streamlit
- scikit-learn

---

## Author

Nouaman Chaaibi

---

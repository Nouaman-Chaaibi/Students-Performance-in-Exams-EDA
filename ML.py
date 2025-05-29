import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

def prepare_ml_data(df):
    """
    Step 1: Prepare data for machine learning
    - Convert text categories to numbers
    - Select features (inputs) and target (what we want to predict)
    
    Returns:
        X: Features (input data)
        y: Target (what we predict)
        label_encoders: Dictionary to convert categories back and forth
        feature_names: Names of features for display
    """
    # Make a copy to avoid changing original data
    ml_df = df.copy()
    
    # Step 1a: Convert categorical variables to numbers
    # LabelEncoder converts text like 'male'/'female' to numbers like 0/1
    label_encoders = {}
    categorical_columns = ['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
    
    for column in categorical_columns:
        le = LabelEncoder()
        ml_df[column] = le.fit_transform(ml_df[column])
        label_encoders[column] = le  # Save for later use
    
    # Step 1b: Select features (what we use to predict) and target (what we predict)
    features = ['reading_score', 'writing_score', 'gender', 'parental_level_of_education', 
                'test_preparation_course', 'lunch', 'race/ethnicity']
    target = 'math_score'
    
    # Feature names for display (more readable)
    feature_names = ['Reading Score', 'Writing Score', 'Gender', 'Parent Education', 
                     'Test Prep', 'Lunch Type', 'Race/Ethnicity']
    
    X = ml_df[features]  # Input features
    y = ml_df[target]    # What we want to predict
    
    return X, y, label_encoders, feature_names

def train_model(X, y):
    """
    Step 2: Train the machine learning model
    - Split data into training and testing parts
    - Train Random Forest model
    - Test how well it works
    
    Returns:
        model: Trained model
        metrics: Dictionary with performance metrics
        test_data: Dictionary with test results for plotting
    """
    # Step 2a: Split data - 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 2b: Create and train the model
    # Random Forest is like having many decision trees vote on the answer
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)  # Learn from training data
    
    # Step 2c: Test the model
    y_pred = model.predict(X_test)  # Make predictions on test data
    
    # Calculate how good our predictions are
    mse = mean_squared_error(y_test, y_pred)  # Lower is better
    r2 = r2_score(y_test, y_pred)  # Higher is better (max = 1.0)
    rmse = np.sqrt(mse)  # Root Mean Square Error
    
    # Package results
    metrics = {
        'r2': r2,
        'mse': mse,
        'rmse': rmse
    }
    
    test_data = {
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return model, metrics, test_data

def get_feature_importance(model, feature_names):
    """
    Get feature importance from trained model
    
    Returns:
        importance_data: Dictionary with feature names and their importance
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]  # Sort from highest to lowest
    
    importance_data = {
        'features': [feature_names[i] for i in indices],
        'importance': importance[indices],
        'indices': indices
    }
    
    return importance_data

def make_prediction(model, label_encoders, input_data):
    """
    Make a prediction for a single student
    
    Args:
        model: Trained ML model
        label_encoders: Dictionary to convert categories to numbers
        input_data: Dictionary with student characteristics
    
    Returns:
        predicted_score: Predicted math score
    """
    # Convert categorical inputs to numbers using the same encoders from training
    processed_data = {
        'reading_score': input_data['reading_score'],
        'writing_score': input_data['writing_score'],
        'gender': label_encoders['gender'].transform([input_data['gender']])[0],
        'parental_level_of_education': label_encoders['parental_level_of_education'].transform([input_data['parental_education']])[0],
        'test_preparation_course': label_encoders['test_preparation_course'].transform([input_data['test_prep']])[0],
        'lunch': label_encoders['lunch'].transform([input_data['lunch']])[0],
        'race/ethnicity': label_encoders['race/ethnicity'].transform([input_data['race']])[0]
    }
    
    # Create DataFrame for prediction (model expects this format)
    input_df = pd.DataFrame([processed_data])
    
    # Make prediction
    predicted_score = model.predict(input_df)[0]
    
    return predicted_score

def find_similar_students(df, reading_score, writing_score, tolerance=10):
    """
    Find students with similar reading and writing scores
    
    Args:
        df: Original dataset
        reading_score: Target reading score
        writing_score: Target writing score
        tolerance: How close scores need to be (default: ±10 points)
    
    Returns:
        similar_students: DataFrame with similar students
    """
    similar_mask = (
        (abs(df['reading_score'] - reading_score) <= tolerance) & 
        (abs(df['writing_score'] - writing_score) <= tolerance)
    )
    similar_students = df[similar_mask]
    
    return similar_students

def get_confidence_level(r2_score):
    """
    Convert R² score to human-readable confidence level
    
    Returns:
        confidence: String describing confidence level
        color: Color for Streamlit display
    """
    if r2_score > 0.8:
        return "High", "green"
    elif r2_score > 0.6:
        return "Medium", "orange"
    else:
        return "Low", "red"

class MLPredictor:
    """
    A class to handle all ML operations together
    This makes it easier to manage all the components
    """
    def __init__(self, df):
        """Initialize with dataset"""
        self.df = df
        self.model = None
        self.label_encoders = None
        self.feature_names = None
        self.metrics = None
        self.test_data = None
        self.is_trained = False
    
    def prepare_and_train(self):
        """Prepare data and train model in one step"""
        # Prepare data
        X, y, self.label_encoders, self.feature_names = prepare_ml_data(self.df)
        
        # Train model
        self.model, self.metrics, self.test_data = train_model(X, y)
        
        self.is_trained = True
        
        return self.model, self.metrics, self.test_data
    
    def predict(self, input_data):
        """Make prediction (only if model is trained)"""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call prepare_and_train() first.")
        
        return make_prediction(self.model, self.label_encoders, input_data)
    
    def get_feature_importance(self):
        """Get feature importance (only if model is trained)"""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call prepare_and_train() first.")
        
        return get_feature_importance(self.model, self.feature_names)
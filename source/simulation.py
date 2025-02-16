import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from constants import NUM_SAMPLES, DATA_PATH, MODEL_PATH
import os
import joblib

def generate_labeled_data(NUM_SAMPLES):
    """
    Generate synthetic IoT task data with labels.
    """
    data = {
        'data_size': np.random.uniform(0.1, 5, NUM_SAMPLES),
        'deadline': np.random.uniform(0.5, 2, NUM_SAMPLES),
        'required_cpu': np.random.uniform(0.5, 3, NUM_SAMPLES),
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(DATA_PATH, 'filename.csv'))
    
    # Label tasks based on a threshold (e.g., delay-sensitive or resource-intensive)
    df['label'] = df.apply(lambda x: 0 if (x['data_size'] * x['required_cpu'] / x['deadline']) < 1.0 else 1, axis=1)
    # 0 = delay-sensitive, 1 = resource-intensive
    
    return df

def train_logistic_regression(data):
    """
    Train a Logistic Regression model for task categorization.
    """
    # Features and labels
    X = data[['data_size', 'deadline', 'required_cpu']]
    y = data['label']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    MODEL_path= os.path.join(MODEL_PATH, 'logistic_regression_model.joblib')

    joblib.dump(model, MODEL_path)

    return model

if __name__ == "__main__":
    df = generate_labeled_data(NUM_SAMPLES)
    train_logistic_regression(df)
    print(df)
import numpy as np
import pandas as pd
import os
def generate_labeled_data(num_samples=1000):
    """
    Generate synthetic IoT task data with labels.
    """
    data = {
        'data_size': np.random.uniform(0.1, 5, num_samples),
        'deadline': np.random.uniform(0.5, 2, num_samples),
        'required_cpu': np.random.uniform(0.5, 3, num_samples),
    }
    df = pd.DataFrame(data)
    
    # Label tasks based on a threshold (e.g., delay-sensitive or resource-intensive)
    df['label'] = df.apply(lambda x: 0 if (x['data_size'] * x['required_cpu'] / x['deadline']) < 1.0 else 1, axis=1)
    # 0 = delay-sensitive, 1 = resource-intensive
    
    return df

def save_data(df, file_path):
    """
    Save the generated data to a CSV file.
    """
    os.makedirs(os.path.dirname(data), exist_ok=True)
    df.to_csv(data, index=False)

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(data)
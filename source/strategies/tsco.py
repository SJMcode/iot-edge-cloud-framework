import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .ibts import calculate_Cindex, identify_KDk_KRk
from constants import K, A, M, Sl, N, D
from .dmo import sort_ekl, update_ekl
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def task_offloading(Kink, lambda_k, r, beta_a, X, Ekl, D, model):
    """
    TSCO Framework with Logistic Regression-based task classification.
    """
    # Calculate Cindex(t)
    t = 1  # Example time parameter
    Cindex_t = calculate_Cindex(Kink, lambda_k, r, beta_a, t)

    # Use Logistic Regression model to classify tasks
    tasks = pd.DataFrame({
        'data_size': Kink,
        'deadline': lambda_k,
        'required_cpu': np.random.rand(len(Kink))  # Example feature
    })
    tasks['label'] = model.predict(tasks[['data_size', 'deadline', 'required_cpu']])

    # Identify KDk and KRk based on ML predictions
    KDk = tasks[tasks['label'] == 0].index.tolist()  # Tasks to process locally (delay-sensitive)
    KRk = tasks[tasks['label'] == 1].index.tolist()  # Tasks to offload to the cloud (resource-intensive)

    # Step 4: Offload KRk to cloud servers
    for k in KRk:
        n = np.random.choice(N)  # Randomly select a cloud server
        print(f"Task {k} offloaded to cloud server {n}")

    # Step 5: Assign tasks to processing devices using DMO
    for l in range(Sl):
        # Sort ekl in ascending order
        sorted_ekl, sorted_indices = sort_ekl(Ekl[:, l])

        # Identify arg min{ekl}
        min_ekl_index = sorted_indices[0]
        min_ekl_value = sorted_ekl[0]

        # Apply Equation (23) to update ekl
        Ekl[:, l] = update_ekl(Ekl[:, l])

        # Assign task to resource
        if min_ekl_index is not None:
            print(f"Task {min_ekl_index} assigned to processing device {l}")
            Ekl[min_ekl_index, l] = np.inf  # Mark as assigned

    # Step 6: Process remaining tasks locally
    for k in KDk:
        print(f"Task {k} processed locally")

# Example usage
if __name__ == "__main__":
    
    from simulation import generate_labeled_data, train_logistic_regression

    # Generate synthetic data
    data = generate_labeled_data()

    # Train Logistic Regression model
    model = train_logistic_regression(data)

    # Save the model
    joblib.dump(model, 'models/task_classifier_lr.pkl')

    
    Kink = np.random.rand(K)  # Example initialization
    lambda_k = np.random.rand(K)  # Task arrival rate
    r = np.random.rand(A, M)  # Data transmission rate
    beta_a = np.random.rand(A)  # Transmission bandwidth
    X = np.zeros((K, A))  # Binary computation offloading decision matrix
    Ekl = np.random.rand(K, Sl)  # Energy consumption on computing devices
    D = 0.5  # Threshold for task offloading

    # Run the TSCO framework with Logistic Regression
    task_offloading(Kink, lambda_k, r, beta_a, X, Ekl, D, model)
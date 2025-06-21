=SIM_4=====================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import random
class ExecutionEnvironment:
    def __init__(self, id, computation_capacity, frequency, transmission_rate):
        self.__id = id
        self.__computation_capacity = computation_capacity
        self.__frequency = frequency
        self.__transmission_rate = transmission_rate

    def get_id(self):
        return self.__id

    def get_computation_capacity(self):
        return self.__computation_capacity

    def get_frequency(self):
        return self.__frequency

    def get_transmission_rate(self):
        return self.__transmission_rate

class Device:
    def __init__(self, id, computation_capacity, frequency):
        self.__id = id
        self.__computation_capacity = computation_capacity
        self.__frequency = frequency

    def get_id(self):
        return self.__id

    def get_frequency(self):
        return self.__frequency

class IIoTSystem:
    def __init__(self, num_iot_devices, num_fog_devices, num_cloud_servers, D, bandwidth, K_k_in_range, lambda_k_range):
        self.num_iot_devices = num_iot_devices
        self.num_fog_devices = num_fog_devices
        self.num_cloud_servers = num_cloud_servers
        self.D = D
        self.bandwidth = bandwidth
        self.lambda_k_range = lambda_k_range
        self.K_k_in_range = K_k_in_range
        self.iot_devices = []
        self.fog_devices = []
        self.cloud_servers = []
        self.create_devices()
        self.create_fog_environment()
        self.create_cloud_environment()

    def create_devices(self):
        np.random.seed(42)
        for i in range(self.num_iot_devices):
            computation_capacity = np.random.uniform(1, 5)
            frequency = np.random.uniform(0.5, 2)
            device = Device(i, computation_capacity, frequency)
            self.iot_devices.append(device)

    def create_fog_environment(self):
        np.random.seed(43)
        for i in range(self.num_fog_devices):
            computation_capacity = np.random.uniform(5, 10)
            frequency = np.random.uniform(2, 4)
            fog_device = ExecutionEnvironment(i, computation_capacity, frequency, transmission_rate=10)
            self.fog_devices.append(fog_device)

    def create_cloud_environment(self):
        np.random.seed(44)
        for i in range(self.num_cloud_servers):
            computation_capacity = np.random.uniform(10, 20)
            frequency = np.random.uniform(4, 8)
            cloud_server = ExecutionEnvironment(i, computation_capacity, frequency, transmission_rate=100)
            self.cloud_servers.append(cloud_server)

def rco_task_offloading(tasks, cloud_devices):
    execution_environments = []
    total_energy = 0
    total_delay = 0

    def calculate_cloud_delay(task, cloud_devices):
        cloud_device = random.choice(cloud_devices)
        transmission_time = task['transmission_time'] / cloud_device.get_transmission_rate()
        computation_time = task['required_cpu'] / cloud_device.get_computation_capacity()
        return transmission_time + computation_time

    def calculate_cloud_energy(task, cloud_devices):
        cloud_device = random.choice(cloud_devices)
        computation_time = task['required_cpu'] / cloud_device.get_computation_capacity()
        return computation_time * 2.0  

    for index, task in tasks.iterrows():
        execution_environments.append('Cloud')
        total_energy += calculate_cloud_energy(task, cloud_devices)
        total_delay += calculate_cloud_delay(task, cloud_devices)

    return execution_environments, total_energy, total_delay

def pbco_task_offloading(tasks, fog_devices, cloud_devices):
    execution_environments = []
    total_energy = 0
    total_delay = 0

    def calculate_fog_delay(task, fog_devices):
        fog_device = random.choice(fog_devices)
        transmission_time = task['transmission_time'] / fog_device.get_transmission_rate()
        computation_time = task['required_cpu'] / fog_device.get_computation_capacity()
        return transmission_time + computation_time

    def calculate_cloud_delay(task, cloud_devices):
        cloud_device = random.choice(cloud_devices)
        transmission_time = task['transmission_time'] / cloud_device.get_transmission_rate()
        computation_time = task['required_cpu'] / cloud_device.get_computation_capacity()
        return transmission_time + computation_time

    def calculate_fog_energy(task, fog_devices):
        fog_device = random.choice(fog_devices)
        computation_time = task['required_cpu'] / fog_device.get_computation_capacity()
        return computation_time * 1.0  

    def calculate_cloud_energy(task, cloud_devices):
        cloud_device = random.choice(cloud_devices)
        computation_time = task['required_cpu'] / cloud_device.get_computation_capacity()
        return computation_time * 2.0  

    for index, task in tasks.iterrows():
        offload_type = random.choices(['Fog', 'Cloud'], weights=[0.3, 0.7], k=1)[0]  # 70% Cloud, 30% Fog

        if offload_type == 'Fog':
            execution_environments.append('Fog')
            total_energy += calculate_fog_energy(task, fog_devices)
            total_delay += calculate_fog_delay(task, fog_devices)
        else:
            execution_environments.append('Cloud')
            total_energy += calculate_cloud_energy(task, cloud_devices)
            total_delay += calculate_cloud_delay(task, cloud_devices)

    return execution_environments, total_energy, total_delay

def task_offloading(tasks, model, iot_devices, fog_devices, cloud_devices):
    execution_environments = []
    total_energy = 0
    total_delay = 0
    def calculate_cloud_delay(task, cloud_devices):
      
      cloud_device = min(cloud_devices, key=lambda c: c.get_computation_capacity())
      transmission_time = task['transmission_time'] / cloud_device.get_transmission_rate()
      computation_time = task['required_cpu'] / cloud_device.get_computation_capacity()
      total_delay = transmission_time + computation_time
      return total_delay

    def calculate_fog_delay(task, fog_devices):
      fog_device = min(fog_devices, key=lambda f: f.get_computation_capacity())
      transmission_time = task['transmission_time'] / fog_device.get_transmission_rate()
      computation_time = task['required_cpu'] / fog_device.get_computation_capacity()
      total_delay = transmission_time + computation_time
      return total_delay
    def calculate_cloud_energy(task, cloud_devices):
        cloud_device = min(cloud_devices, key=lambda c: c.get_computation_capacity())  # Select best cloud device
        computation_time = task['required_cpu'] / cloud_device.get_computation_capacity()
        energy_consumption = computation_time * 2.0  # Example coefficient for cloud energy
        return energy_consumption

    def calculate_fog_energy(task, fog_devices):
        fog_device = min(fog_devices, key=lambda f: f.get_computation_capacity())  # Select best fog device
        computation_time = task['required_cpu'] / fog_device.get_computation_capacity()
        energy_consumption = computation_time * 1.0  # Example coefficient for fog energy
        return energy_consumption

    for index, task in tasks.iterrows():
        task_type = None
        device = next((dev for dev in iot_devices if dev.get_id() == task['device_id']), None)

        if device and device.get_frequency() >= task['frequency']:
            execution_environments.append('IIoT')
        else:
            task_features = task[['required_cpu', 'frequency', 'exec_deadline']].values.reshape(1, -1)
            task_type = model.predict(task_features)[0]

            if task_type == 1:
                execution_environments.append('Fog')
                total_energy += calculate_fog_energy(task, fog_devices)
                total_delay += calculate_fog_delay(task, fog_devices)
            else:
                execution_environments.append('Cloud')
                total_energy += calculate_cloud_energy(task, cloud_devices)
                total_delay += calculate_cloud_delay(task, cloud_devices)

    return execution_environments, total_energy, total_delay

# Create system and load model
iiot_system = IIoTSystem(20, 10, 3, 0.5, 50, [1, 5], [0.1, 1.0])
task_df = pd.read_csv('./source/data/iot_task_data_per_device.csv')
model = joblib.load('./source/models/logistic_regression_model.joblib')

# Execute task offloading
execution_environments, total_energy, total_delay = task_offloading(task_df, model, iiot_system.iot_devices, iiot_system.fog_devices, iiot_system.cloud_servers)

print(execution_environments)
print(total_energy)
print(total_delay)

rco_execution, rco_energy, rco_delay = rco_task_offloading(task_df, iiot_system.cloud_servers)
pbco_execution, pbco_energy, pbco_delay = pbco_task_offloading(task_df, iiot_system.fog_devices, iiot_system.cloud_servers)
ml_execution, ml_energy, ml_delay = task_offloading(task_df, model, iiot_system.iot_devices, iiot_system.fog_devices, iiot_system.cloud_servers)

# Use the extracted values for plotting
energy_values = [rco_energy, pbco_energy, ml_energy]  # ML has the lowest energy
delay_values = [rco_delay, pbco_delay, ml_delay]

# Offloading strategies
strategies = ['ML-Based', 'RCO', 'PBCO']

# Energy consumption values
energy_values = [ml_energy, rco_energy, pbco_energy]

# Delay values
delay_values = [ml_delay, rco_delay, pbco_delay]

# Set bar width
bar_width = 0.4
x = np.arange(len(strategies))

# Create subplots for energy and delay
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
colors = ['lightblue', 'lightcoral', 'lightgreen']
# Bar chart for Energy Consumption
ax[0].bar(x, energy_values, color=colors, width=bar_width)
ax[0].set_xticks(x)
ax[0].set_xticklabels(strategies)
ax[0].set_ylabel("Total Energy (J)")
ax[0].set_title("Energy Consumption Comparison")
ax[0].grid(axis='y', linestyle='--', alpha=0.7)

# Bar chart for Delay
ax[1].bar(x, delay_values, color=['blue', 'red', 'green'], width=bar_width)
ax[1].set_xticks(x)
ax[1].set_xticklabels(strategies)
ax[1].set_ylabel("Total Delay (s)")
ax[1].set_title("Delay Comparison")
ax[1].grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()
=SIM_5============================================================
# %% Data and MODEL
import numpy as np
import pandas as pd
import os

# Data Path
DATA_PATH = "./source/data"
# %%
def generate_labeled_data(NUM_SAMPLES, D=0.5):
    """
    Generate synthetic IoT task data with labels based on priority index.

    Parameters:
        NUM_SAMPLES (int): Number of samples to generate.
        D (float): Threshold for classifying tasks.

    Returns:
        DataFrame: Generated dataset.
    """
    # Generate Random Values for IoT Task Parameters
    data = {
        'required_cpu': np.random.uniform(0.1, 5, NUM_SAMPLES),
        'frequency': np.random.uniform(0.5, 2, NUM_SAMPLES),
        'exec_deadline': np.random.uniform(0.5, 3, NUM_SAMPLES),
        'task_arrival_rate': np.random.uniform(0.1, 5, NUM_SAMPLES),  # λ_k
        'transmission_time': np.random.uniform(0.1, 1, NUM_SAMPLES),  # β_a
        'computation_cost': np.random.randint(1, 5, NUM_SAMPLES),  # X(k, a)
        'energy_consumption': np.random.uniform(1, 10, NUM_SAMPLES),  # E_kl
        'buffered_tasks': np.random.randint(1, 10, NUM_SAMPLES),  # K_k^in
    }
    # %%
    # Create DataFrame
    df = pd.DataFrame(data)


    # %%
    # Compute Priority Index C_index using Equation (18)
    df['C_index'] = (df['buffered_tasks'] / (df['computation_cost'] * df['task_arrival_rate'])) * df['transmission_time']
    print(df['C_index'])
    # %%
    # Label Tasks Based on Priority Index
    df['Task_Type'] = (df['C_index'] <= D).astype(int)  # 1 = Delay Sensitive, 0 = Resource Intensive
    # %%
    # Save the Data
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    filename = os.path.join(DATA_PATH, 'iot_task_data.csv')
    df.to_csv(filename, index=False)

    print(f"Dataset saved successfully at: {filename}")
    return df

# Generate Data
# %%
df = generate_labeled_data(1000)  # Generate 1000 samples
#print(df.head())  # Preview first 5 rows

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,precision_score, recall_score, f1_score
# %%
iot_data= pd.read_csv('./source/data/iot_task_data.csv')
iot_data.head()
# %%
iot_data.info()
# %%
#sns.pairplot(iot_data, hue='Task_Type')
#plt.show()
# %%
#sns.heatmap(iot_data.corr(), annot=True)
#plt.show()
# %%
iot_data['Task_Type'].value_counts(normalize=True)
# %%
for col in iot_data.columns[iot_data.dtypes == 'object']:
    sns.countplot(x=col, data=iot_data, hue='Task_Type')
    plt.show()
# %%
X = df[['required_cpu', 'frequency', 'exec_deadline', 'task_arrival_rate',
        'transmission_time', 'computation_cost', 'energy_consumption', 'buffered_tasks']]
y = df['Task_Type']

# Split Data into Training and Testing Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                     y,
                                                       test_size=0.3,
                                                         random_state=42,
                                                         stratify=y)
print(X_train[:5], '\n', y_train[:5])
# %%
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
# %%
y_pred = clf.predict(X_test)
# %%
print(y_pred[:5])
# %%
print(clf.score(X_test, y_test))

conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['delay-sensitive', 'resource-intensive'])
disp.plot()
plt.show()

clf.score(X_test, y_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

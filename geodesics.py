import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models

# Minkowski metric tensor (4x4 for 3D spacetime)
minkowski_metric = np.diag([-1, 1, 1, 1])

# Schrödinger metric tensor (4x4 for 3D spacetime)
schrodinger_metric = np.diag([1, 1, 1, -1])

# Function to apply a metric based on the dataset configuration
def apply_metric(position, velocity, metric):
    spacetime_coordinates = np.concatenate([position, velocity])
    metric_coordinates = np.dot(metric, spacetime_coordinates)
    return metric_coordinates[:3], metric_coordinates[3:]

# Simulated dataset configuration (replace with your data)
# Features: Initial position, initial velocity, planet parameters, etc.
# Target: Spacecraft trajectory
dataset_configurations = np.random.choice(['minkowski', 'schrodinger'], size=1000)
X = np.random.rand(1000, 10)  
y = np.random.rand(1000, 3)  

# Applying metric based on dataset configuration
X_transformed = []
for config, pos, vel in zip(dataset_configurations, X[:, :3], X[:, 3:]):
    if config == 'minkowski':
        transformed_pos, transformed_vel = apply_metric(pos, vel, minkowski_metric)
    elif config == 'schrodinger':
        transformed_pos, transformed_vel = apply_metric(pos, vel, schrodinger_metric)
    else:
        raise ValueError("Invalid dataset configuration")

    X_transformed.append(np.concatenate([transformed_pos, transformed_vel]))

X_transformed = np.array(X_transformed)

#splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

#standardizing funcs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='linear') 
])

def custom_loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

model.compile(optimizer='adam', loss=custom_loss)

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

loss = model.evaluate(X_test, y_test)
print(f'Final Loss: {loss}')

y_pred = model.predict(X_test)

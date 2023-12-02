import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from sympy import symbols, diff, Matrix, simplify, sin

# Define symbolic variables
M, t, r, theta, phi = symbols("M t r theta phi")
coor = [t, r, theta, phi]

# Schwarzschild metric tensor (4x4 for 3D spacetime)
schwarzschild_metric = np.diag([1 - 2 * M / r, -1 / (1 - 2 * M / r), -r**2, -(r**2 * sin(theta)**2)])

# Function to apply the Schwarzschild metric
def apply_schwarzschild_metric(position, velocity):
    spacetime_coordinates = np.concatenate([position, velocity])
    schwarzschild_coordinates = np.dot(schwarzschild_metric, spacetime_coordinates)
    return schwarzschild_coordinates[:3], schwarzschild_coordinates[3:]

dataset_configurations = np.random.choice(['schwarzschild'], size=1000)
X = np.random.rand(1000, 10)  # Replace with actual features
y = np.random.rand(1000, 3)  # Replace with actual trajectories

# Applying Schwarzschild metric to features
X_transformed = []
for config, pos, vel in zip(dataset_configurations, X[:, :3], X[:, 3:]):
    if config == 'schwarzschild':
        transformed_pos, transformed_vel = apply_schwarzschild_metric(pos, vel)
    else:
        raise ValueError("Invalid dataset configuration")

    X_transformed.append(np.concatenate([transformed_pos, transformed_vel]))

X_transformed = np.array(X_transformed)

# Spliting dataset
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Standardize funcs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='linear')  # Adjust output dimensions based on trajectory
])

def custom_loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

model.compile(optimizer='adam', loss=custom_loss)

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

loss = model.evaluate(X_test, y_test)
print(f'Final Loss: {loss}')

y_pred = model.predict(X_test)

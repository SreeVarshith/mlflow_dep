# import mlflow
# import mlflow.tensorflow
# import tensorflow as tf
# from tensorflow.keras.datasets import fashion_mnist
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# from tensorflow.keras.utils import to_categorical

# # Load the Fashion MNIST dataset
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# # Preprocess the data
# x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
# x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# model = tf.keras.Sequential([
#     tf.keras.Input(shape=(28, 28, 1)),
#     tf.keras.layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'),
#     tf.keras.layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2), strides=2),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'),
#     tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2), strides=2),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'),
#     tf.keras.layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'),
#     tf.keras.layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2), strides=2),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax'),
# ])


# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Start an MLflow run
# mlflow.tensorflow.autolog()

# with mlflow.start_run():
#     # Train the model
#     model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)
    
#     # Evaluate the model
#     loss, accuracy = model.evaluate(x_test, y_test)
    
#     # Log metrics manually if needed
#     mlflow.log_metric("loss", loss)
#     mlflow.log_metric("accuracy", accuracy)
    
#     # Log the model
#     mlflow.tensorflow.log_model(model, artifact_path="model")

#     print(f"Model logged with accuracy: {accuracy}")
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



# Define hyperparameters
params = {
    "batch_size": 128,
    "epochs": 5,
    "learning_rate": 0.001,
    "dropout_rate": 0.3,
    "optimizer": "adam",
    "conv_filters": [32, 64, 128],
    "kernel_size": (3, 3),
    "activation": "relu",
    "l2_reg": 0.001
}

# Create the model
model = Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    Conv2D(params["conv_filters"][0], kernel_size=params["kernel_size"], activation=params["activation"], padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(params["dropout_rate"]),
    Conv2D(params["conv_filters"][1], kernel_size=params["kernel_size"], activation=params["activation"], padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(params["dropout_rate"]),
    Conv2D(params["conv_filters"][2], kernel_size=params["kernel_size"], activation=params["activation"], padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(params["dropout_rate"]),
    Flatten(),
    Dense(256, activation=params["activation"], kernel_regularizer=l2(params["l2_reg"])),
    Dense(128, activation=params["activation"], kernel_regularizer=l2(params["l2_reg"])),
    Dense(10, activation='softmax')
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Start an MLflow run
mlflow.tensorflow.autolog()


remote_server ="https://dagshub.com/SreeVarshith/mlflow_dep.mlflow"
mlflow.set_tracking_uri(remote_server)





with mlflow.start_run() as run:
    # Log hyperparameters
    for key, value in params.items():
        mlflow.log_param(key, value)
    
    # Train the model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=params["epochs"], batch_size=params["batch_size"])
    
    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    
    # Log metrics manually if needed
    mlflow.log_metric("loss", loss)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log the model
    mlflow.tensorflow.log_model(model, artifact_path="model")

    print(f"Model logged with run ID: {run.info.run_id}")


# Register the model
model_name = "FashionMNIST_CNN"
model_uri = f"runs:/{run.info.run_id}/model"
mlflow.register_model(model_uri=model_uri, name=model_name)

# Transition the model to 'Production' stage (optional)
client = mlflow.tracking.MlflowClient()
latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage="Production"
)

print(f"Model version {latest_version} transitioned to Production stage.")

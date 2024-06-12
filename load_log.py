import mlflow
import mlflow.tensorflow
import numpy as np
from PIL import Image
import io

# Load the model from MLflow
model_uri = "runs:/72dfde0750ca4a4c8ac981d67c2a57d2/model"  # Replace <your_run_id>
model = mlflow.tensorflow.load_model(model_uri)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def predict(image_bytes):
    # Convert the image to grayscale and resize to 28x28
    image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
    # Normalize the image
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape to fit model input
    
    # Predict the class
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    return class_names[predicted_class]

# Example usage

with open(r"C:/Users/Munna/Downloads/trouser.jpg", "rb") as f:
    image_bytes = f.read()
    predicted_class = predict(image_bytes)
    print(f"Predicted class: {predicted_class}")

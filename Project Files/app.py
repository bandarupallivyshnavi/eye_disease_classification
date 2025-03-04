# app.py
from flask import Flask, request, render_template
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load trained model
MODEL_PATH = "model/eye_disease_model.h5"
model = load_model(MODEL_PATH)

# Define upload folder
UPLOAD_FOLDER = "static/uploads/"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to preprocess image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define class labels (Update these based on dataset labels)
class_labels = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return "No file part"

        file = request.files["image"]
        if file.filename == "":
            return "No selected file"

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Preprocess image and predict
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        result = class_labels[predicted_class]

        return render_template("result.html", filename=filename, result=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

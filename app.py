from flask import Flask, render_template, request, flash, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ====== Load Trained Model ======
MODEL_PATH = "DenseNet121_LSTM_GRU_final.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    model = None
    print(f"⚠️ Could not load model: {e}")

# ====== Correct Class Order based on training =====
# During training: {'Parasitized': 0, 'Uninfected': 1}
CLASS_NAMES = ["Parasitized", "Uninfected"]

# ====== DenseNet121 Feature Extractor (same as training) ======
base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False
out = GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(inputs=base_model.input, outputs=out)
print("✅ DenseNet121 feature extractor ready (input 128x128).")

# ====== Routes ======
@app.route("/")
def home():
    return render_template("Home.html", title="Home")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded!", "danger")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected!", "warning")
            return redirect(request.url)

        if model is None:
            flash("Model not loaded. Please check your model path.", "danger")
            return redirect(request.url)

        try:
            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            # ===== Preprocess =====
            img = image.load_img(file_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # ===== Extract Features =====
            features = feature_extractor.predict(img_array)   # (1,1024)
            features = np.expand_dims(features, axis=1)       # (1,1,1024)

            # ===== Predict =====
            preds = model.predict(features)
            pred_idx = int(np.argmax(preds))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = round(100 * np.max(preds), 2)

            # ===== Debug Logging =====
            print("========== DEBUG ==========")
            print("Raw predictions:", preds)
            print("Predicted index:", pred_idx)
            print("Class mapping:", CLASS_NAMES)
            print("Final Prediction:", pred_class, "Confidence:", confidence, "%")
            print("===========================")

            result = f"Prediction: {pred_class} (Confidence: {confidence}%)"

            if os.path.exists(file_path):
                os.remove(file_path)

            return render_template("Predict.html", title="Predict", result=result)

        except Exception as e:
            flash(f"Error during prediction: {str(e)}", "danger")
            return redirect(request.url)

    return render_template("Predict.html", title="Predict")

@app.route("/login")
def login():
    return render_template("Login.html", title="Login")

@app.route("/signup")
def signup():
    return render_template("Signup.html", title="Signup")

# ====== Run Server ======
if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request, flash, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import os
from scipy.stats import entropy   # for uncertainty check

# ==============================
# CONFIG
# ==============================
app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = (128, 128)
users = {}  # Mock DB

CONF_THRESHOLD = 0.7
UNCERTAINTY_THRESHOLD = 0.4



# ====== Load Trained Model ======
MODEL_PATH = "DenseNet121_LSTM_GRU_final.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    model = None
    print(f"⚠️ Could not load model: {e}")

# ====== Class Names ======
CLASS_NAMES = ["Parasitized", "Uninfected"]

# ====== DenseNet121 Feature Extractor ======
base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False
out = GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(inputs=base_model.input, outputs=out)
print("✅ DenseNet121 feature extractor ready (input 128x128).")

# ==============================
# PREPROCESSING FUNCTIONS
# ==============================
def resize_image(img, size=IMG_SIZE):
    return cv2.resize(img, size)

def denoise_image(img):
    return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10,
                                           templateWindowSize=7, searchWindowSize=21)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def sharpen_image(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def normalize_image(img):
    return img / 255.0

def preprocess_pipeline(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_image(img)
    img = denoise_image(img)
    img = apply_clahe(img)
    img = sharpen_image(img)
    img = normalize_image(img)
    return (img * 255).astype(np.uint8)

# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return render_template("Home.html", title="Home")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        if email in users:
            flash("Email already registered!", "danger")
        else:
            users[email] = {"username": username, "password": password}
            flash("Signup successful! Please login.", "success")
            return redirect(url_for("login"))
    return render_template("Signup.html", title="Signup")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        if email in users and users[email]["password"] == password:
            flash(f"Welcome {users[email]['username']}!", "success")
            return redirect(url_for("predict"))
        else:
            flash("Invalid email or password", "danger")
    return render_template("Login.html", title="Login")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    uploaded_img = None
    processed_img_name = None

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
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            uploaded_img = filename

            # ===== Preprocess for DenseNet121 =====
            img = image.load_img(file_path, target_size=IMG_SIZE)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # ===== Extract Features =====
            features = feature_extractor.predict(img_array)
            features = np.expand_dims(features, axis=1)

            # ===== Predict =====
            preds = model.predict(features)
            probs = preds[0]
            pred_idx = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(np.max(probs))
            uncertainty = entropy(probs)

            # ===== Threshold Check =====
            if confidence < CONF_THRESHOLD or uncertainty > UNCERTAINTY_THRESHOLD:
                result = {
                    "prediction": "Not a Malaria Cell",
                    "is_cell": False,
                    "confidence": round(confidence, 2),
                    "description": "The uploaded image does not resemble a recognized malaria cell.",
                    "note": "Try uploading a clear microscopic blood smear image."
                }
                return render_template("Predict.html", title="Predict",
                                       result=result,
                                       uploaded_img=uploaded_img,
                                       processed_img=None)

            # ===== Valid Prediction =====
            descriptions = {
                "Parasitized": "This cell shows presence of malaria parasites.",
                "Uninfected": "This cell does not show malaria parasites."
            }

            result = {
                "prediction": pred_class,
                "is_cell": True,
                "confidence": round(confidence, 2),
                "description": descriptions.get(pred_class, "No description available."),
                "note": "Please consult a medical professional for confirmation."
            }

            # ===== Processed Image Preview =====
            processed_img = preprocess_pipeline(file_path)
            processed_img_name = "processed_" + filename
            processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_img_name)
            cv2.imwrite(processed_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))

        except Exception as e:
            flash(f"Error during prediction: {str(e)}", "danger")
            return redirect(request.url)

    return render_template("Predict.html",
                           title="Predict",
                           result=result,
                           uploaded_img=uploaded_img,
                           processed_img=processed_img_name)

# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(debug=True)

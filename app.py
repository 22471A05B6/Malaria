import os
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ==============================
# CONFIG
# ==============================
app = Flask(__name__)
app.secret_key = "your_secret_key_here"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = (128, 128)
users = {}  # Mock DB


# ==============================
# PREPROCESSING FUNCTIONS
# ==============================
def resize_image(image, size=IMG_SIZE):
    return cv2.resize(image, size)

def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10,
                                           templateWindowSize=7, searchWindowSize=21)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def normalize_image(image):
    return image / 255.0

def preprocess_pipeline(image_path):
    """Runs preprocessing pipeline on uploaded image."""
    img = load_img(image_path, target_size=IMG_SIZE)
    img = img_to_array(img).astype("uint8")

    resized = resize_image(img)
    denoised = denoise_image(resized)
    clahe_applied = apply_clahe(denoised)
    sharpened = sharpen_image(clahe_applied)
    normalized = normalize_image(sharpened)

    final_img = (normalized * 255).astype(np.uint8)
    return final_img


# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return render_template("Home.html")

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
    return render_template("Signup.html")

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
    return render_template("Login.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    uploaded_img = None

    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded!", "danger")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected!", "danger")
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Run preprocessing
            processed_img = preprocess_pipeline(file_path)

            # Save processed image for preview
            processed_path = os.path.join(app.config["UPLOAD_FOLDER"], "processed_" + filename)
            cv2.imwrite(processed_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))

            # Dummy prediction (replace with ML model later)
            result = "Parasite Detected ✅"

            uploaded_img = filename
            processed_img_name = "processed_" + filename

            return render_template("Predict.html",
                                   result=result,
                                   uploaded_img=uploaded_img,
                                   processed_img=processed_img_name)

    return render_template("Predict.html", result=result)


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    app.run(debug=True)

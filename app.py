
import os
from uuid import uuid4

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.efficientnet import preprocess_input


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "plant_disease_efficientnet.keras"
)
LABELS_PATH = os.path.join(BASE_DIR, "models", "class_names.txt")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

IMAGE_SIZE = (224, 224)
TOP_PREDICTIONS = 3

# Show a warning when the top two predictions are close.
# This is a provisional threshold, not a validated confidence measure.
UNCERTAINTY_MARGIN = 5.0

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Load the trained model
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


def load_class_names():
    class_names = {}

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or ":" not in line:
                continue

            index, name = line.split(":", 1)
            class_names[int(index.strip())] = name.strip()

    expected_count = model.output_shape[-1]

    if len(class_names) != expected_count:
        raise ValueError(
            f"Model outputs {expected_count} classes, "
            f"but {len(class_names)} labels were loaded."
        )

    expected_indices = set(range(expected_count))

    if set(class_names.keys()) != expected_indices:
        raise ValueError(
            "Class labels must have consecutive indices "
            "starting from 0."
        )

    return class_names


class_names = load_class_names()


def allowed_file(filename):
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def predict_leaf(image_path):
    # Match the model's expected image size and RGB input.
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)

        image_array = np.asarray(image, dtype=np.float32)

    # Keep preprocessing consistent with the evaluation scripts.
    # Do not divide by 255 again.
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    scores = model.predict(image_array, verbose=0)[0]

    if len(scores) != len(class_names):
        raise ValueError(
            "Prediction output does not match the class labels."
        )

    top_indices = np.argsort(scores)[::-1][:TOP_PREDICTIONS]

    results = []

    for index in top_indices:
        label = class_names[int(index)]
        parts = label.split("___", 1)

        plant_name = (
            parts[0].replace("_", " ").replace(",", "").strip()
        )

        condition = (
            parts[1].replace("_", " ").strip()
            if len(parts) > 1
            else "Unknown"
        )

        results.append({
            "plant": plant_name,
            "condition": condition,
            "label": label,
            "confidence": round(float(scores[index]) * 100, 2)
        })

    return results


@app.route("/", methods=["GET", "POST"])
def home():
    error = None
    uploaded_image = None
    predictions = None
    prediction_warning = None

    if request.method == "POST":
        file = request.files.get("leaf_image")

        if not file or not file.filename:
            error = "Please choose a plant leaf image first."

        elif not allowed_file(file.filename):
            error = "Please upload a JPG, JPEG, PNG or WEBP image."

        else:
            filename = secure_filename(file.filename)

            try:
                # Validate the uploaded image before saving it.
                with Image.open(file.stream) as check_image:
                    check_image.verify()

                file.stream.seek(0)

                with Image.open(file.stream) as image:
                    image = ImageOps.exif_transpose(image)
                    image = image.convert("RGB")

                    # Save JPEG content with a .jpg extension.
                    safe_stem = os.path.splitext(filename)[0] or "leaf"
                    unique_name = f"{uuid4().hex}_{safe_stem}.jpg"

                    save_path = os.path.join(
                        app.config["UPLOAD_FOLDER"],
                        unique_name
                    )

                    image.save(save_path, format="JPEG", quality=95)

                uploaded_image = unique_name
                predictions = predict_leaf(save_path)

                # Warn when the two leading predictions are close.
                if len(predictions) >= 2:
                    score_gap = (
                        predictions[0]["confidence"]
                        - predictions[1]["confidence"]
                    )

                    if score_gap < UNCERTAINTY_MARGIN:
                        prediction_warning = (
                            "Uncertain result: the top predictions "
                            "are close. Please verify the plant and "
                            "disease before relying on this result."
                        )

            except (UnidentifiedImageError, OSError, ValueError):
                error = (
                    "The image could not be processed. "
                    "Please try another image."
                )

            except Exception:
                app.logger.exception(
                    "Unexpected error during prediction."
                )
                error = (
                    "An unexpected error occurred while analyzing "
                    "the image. Please try again."
                )

    return render_template(
        "index.html",
        error=error,
        uploaded_image=uploaded_image,
        predictions=predictions,
        prediction_warning=prediction_warning
    )


if __name__ == "__main__":
    app.run(debug=True)

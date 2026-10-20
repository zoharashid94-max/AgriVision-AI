
import os
import csv
import numpy as np
import tensorflow as tf

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from tensorflow.keras.applications.efficientnet import preprocess_input


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "plant_disease_efficientnet.keras"
)
LABELS_PATH = os.path.join(BASE_DIR, "models", "class_names.txt")

RESULTS_DIR = os.path.join(
    BASE_DIR, "evaluation", "results", "plantdoc"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

REPORT_PATH = os.path.join(
    RESULTS_DIR, "plantdoc_classification_report.txt"
)
MATRIX_PATH = os.path.join(
    RESULTS_DIR, "plantdoc_confusion_matrix.csv"
)
ERRORS_PATH = os.path.join(
    RESULTS_DIR, "plantdoc_incorrect_predictions.csv"
)
SUMMARY_PATH = os.path.join(
    RESULTS_DIR, "plantdoc_prediction_summary.csv"
)


# --------------------------------------------------
# 2. Load model and class names
# --------------------------------------------------

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH, compile=False
)

class_names = {}

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if ":" in line:
            idx, name = line.split(":", 1)
            class_names[int(idx.strip())] = name.strip()

if len(class_names) != model.output_shape[-1]:
    raise ValueError(
        "Model output count does not match class_names.txt."
    )

model_label_to_index = {
    name: idx for idx, name in class_names.items()
}

print("Model input shape:", model.input_shape)
print("Model output classes:", len(class_names))


# --------------------------------------------------
# 3. PlantDoc to PlantVillage mapping
# --------------------------------------------------

mapping = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Corn Gray leaf spot":
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus":
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf":
        "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf black rot": "Grape___Black_rot",
}

valid_mapping = {
    dataset_label: model_label
    for dataset_label, model_label in mapping.items()
    if model_label in model_label_to_index
}

if not valid_mapping:
    raise ValueError("No valid mapped classes found.")

print("Valid mappings:", len(valid_mapping))


# --------------------------------------------------
# 4. Load PlantDoc test split
# --------------------------------------------------

print("\nLoading PlantDoc dataset...")

dataset = load_dataset(
    "geraldmc/plantdoc-full",
    revision="v0.1.0",
    split="train"
)

test_data = dataset.filter(
    lambda row: row["split"] == "test"
)

eligible = [
    row for row in test_data
    if row["class_label"] in valid_mapping
]

if not eligible:
    raise ValueError("No eligible PlantDoc test images found.")

print("Total PlantDoc test images:", len(test_data))
print("Mapped test images:", len(eligible))
print(
    "Mapped classes represented:",
    len(set(row["class_label"] for row in eligible))
)


# --------------------------------------------------
# 5. Image size
# --------------------------------------------------

input_shape = model.input_shape

if isinstance(input_shape, list) or len(input_shape) != 4:
    raise ValueError(f"Unexpected model input shape: {input_shape}")

height = input_shape[1] or 224
width = input_shape[2] or 224

print(f"Image size: {width} x {height}")


# --------------------------------------------------
# 6. Predict in batches and retain prediction details
# --------------------------------------------------

records = []
batch_size = 16

for start in range(0, len(eligible), batch_size):
    batch = eligible[start:start + batch_size]
    images = []
    true_indices = []

    for row in batch:
        image = row["image"].convert("RGB")
        image = image.resize((width, height))

        images.append(np.asarray(image, dtype=np.float32))

        target_name = valid_mapping[row["class_label"]]
        true_indices.append(model_label_to_index[target_name])

    x = np.asarray(images, dtype=np.float32)
    x = preprocess_input(x)

    scores = model.predict(x, verbose=0)

    if scores.ndim != 2 or scores.shape[1] != len(class_names):
        raise ValueError(f"Unexpected prediction shape: {scores.shape}")

    predictions = np.argmax(scores, axis=1)

    for row, true_idx, pred_idx, prediction_scores in zip(
        batch, true_indices, predictions, scores
    ):
        records.append({
            "filename": row.get("filename", ""),
            "dataset_label": row["class_label"],
            "true_index": int(true_idx),
            "true_label": class_names[int(true_idx)],
            "predicted_index": int(pred_idx),
            "predicted_label": class_names[int(pred_idx)],
            "confidence": float(prediction_scores[int(pred_idx)]),
            "correct": int(true_idx) == int(pred_idx),
        })

    print(f"Processed {len(records)} / {len(eligible)}")


# --------------------------------------------------
# 7. Metrics and classification report
# --------------------------------------------------

y_true = [r["true_index"] for r in records]
y_pred = [r["predicted_index"] for r in records]

accuracy = accuracy_score(y_true, y_pred)

# Use only classes present as true labels for the main report.
evaluated_classes = sorted(set(y_true))

report = classification_report(
    y_true,
    y_pred,
    labels=evaluated_classes,
    target_names=[class_names[i] for i in evaluated_classes],
    zero_division=0,
    digits=4
)

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write("PLANTDOC CROSS-DATASET EVALUATION\n")
    file.write(f"PlantDoc test images: {len(test_data)}\n")
    file.write(f"Mapped images evaluated: {len(records)}\n")
    file.write(f"Mapped classes evaluated: {len(evaluated_classes)}\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write(report)
    file.write(
        "\n\nNote: Results apply only to selected mapped classes. "
        "PlantDoc and PlantVillage have different image distributions. "
        "Label correspondence should be manually verified."
    )

print("\n========== PLANTDOC RESULTS ==========")
print("Test images:", len(test_data))
print("Evaluated images:", len(records))
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification report:\n")
print(report)


# --------------------------------------------------
# 8. Confusion matrix
# Include any predicted classes, even outside target classes.
# --------------------------------------------------

matrix_labels = sorted(set(y_true) | set(y_pred))

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=matrix_labels
)

with open(MATRIX_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    names = [class_names[i] for i in matrix_labels]
    writer.writerow(["Actual / Predicted"] + names)

    for name, values in zip(names, matrix):
        writer.writerow([name] + values.tolist())


# --------------------------------------------------
# 9. Save incorrect predictions
# --------------------------------------------------

fields = [
    "filename",
    "dataset_label",
    "true_label",
    "predicted_label",
    "confidence",
    "correct"
]

with open(ERRORS_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for record in records:
        if not record["correct"]:
            writer.writerow({
                key: record[key] for key in fields
            })


# --------------------------------------------------
# 10. Prediction summary
# --------------------------------------------------

summary = {}

for record in records:
    key = (record["true_label"], record["predicted_label"])
    summary[key] = summary.get(key, 0) + 1

with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Actual label", "Predicted label", "Count"])

    for (actual, predicted), count in sorted(
        summary.items(), key=lambda item: (-item[1], item[0])
    ):
        writer.writerow([actual, predicted, count])


# --------------------------------------------------
# 11. Finish
# --------------------------------------------------

print("\nEvaluation complete. Files saved in:")
print(RESULTS_DIR)
print("\nGenerated files:")
print("1.", REPORT_PATH)
print("2.", MATRIX_PATH)
print("3.", ERRORS_PATH)
print("4.", SUMMARY_PATH)

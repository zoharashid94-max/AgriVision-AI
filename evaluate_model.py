
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "plant_disease_efficientnet.keras"
)
LABELS_PATH = os.path.join(BASE_DIR, "models", "class_names.txt")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Loading class names...")
class_names = {}

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if ":" in line:
            index, name = line.split(":", 1)
            class_names[int(index.strip())] = name.strip()

if len(class_names) != model.output_shape[-1]:
    raise ValueError("Model output and class_names.txt do not match.")

print("Loading PlantVillage dataset from Hugging Face...")
dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    split="train"
)

test_data = dataset.filter(lambda row: row["split"] == "test")

print("Number of test images:", len(test_data))
print("Checking class labels...")

dataset_classes = set(test_data["class_label"])
model_classes = set(class_names.values())

if dataset_classes != model_classes:
    print("Classes missing from dataset:", model_classes - dataset_classes)
    print("Unexpected dataset classes:", dataset_classes - model_classes)
    raise ValueError("Dataset classes do not match the model labels.")

y_true = []
y_pred = []
batch_images = []
batch_labels = []
batch_size = 32

def evaluate_batch(images, labels):
    image_array = np.asarray(images, dtype=np.float32)
    image_array = preprocess_input(image_array)
    scores = model.predict(image_array, verbose=0)

    predictions = np.argmax(scores, axis=1)

    y_true.extend(labels)
    y_pred.extend(predictions.tolist())

print("Evaluating test images...")

for i, row in enumerate(test_data):
    image = row["image"].convert("RGB").resize((224, 224))
    batch_images.append(np.asarray(image, dtype=np.float32))
    batch_labels.append(
        next(index for index, name in class_names.items()
             if name == row["class_label"])
    )

    if len(batch_images) == batch_size:
        evaluate_batch(batch_images, batch_labels)
        batch_images = []
        batch_labels = []

        if (i + 1) % 320 == 0:
            print(f"Processed {i + 1} images...")

if batch_images:
    evaluate_batch(batch_images, batch_labels)

accuracy = accuracy_score(y_true, y_pred)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=[class_names[i] for i in range(len(class_names))],
    zero_division=0,
    digits=4
)

print("\n========== EVALUATION RESULTS ==========")
print(f"Test images evaluated: {len(y_true)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification report:")
print(report)

report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
with open(report_path, "w", encoding="utf-8") as file:
    file.write(f"Test images: {len(y_true)}\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write(report)

matrix = confusion_matrix(
    y_true, y_pred, labels=list(range(len(class_names)))
)

matrix_path = os.path.join(RESULTS_DIR, "confusion_matrix.csv")
with open(matrix_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    names = [class_names[i] for i in range(len(class_names))]
    writer.writerow(["Actual / Predicted"] + names)

    for name, values in zip(names, matrix):
        writer.writerow([name] + values.tolist())

print("\nEvaluation complete!")
print("Report saved to:", report_path)
print("Confusion matrix saved to:", matrix_path)

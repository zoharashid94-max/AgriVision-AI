# AgriVision AI Smart Plant Disease Detection

AgriVision AI is a deep learning project that identifies plant diseases from leaf images. I developed this project using Python, TensorFlow, and Flask to explore how AI can be used in agriculture.

Users can upload a plant leaf image, and the trained EfficientNet model predicts the most likely plant health or disease category along with its confidence score.

## Features

- Upload plant leaf images for prediction.
- Classify images into 38 plant health and disease categories.
- Display the predicted result and confidence score.
- Show alternative predictions where available.
- Simple and user-friendly web interface.

## Technologies Used

- Python
- TensorFlow and Keras
- EfficientNet
- Flask
- HTML and CSS

## How It Works

1. Upload a plant leaf image.
2. The application processes the image.
3. The trained model predicts the plant health or disease category.
4. The result and confidence score are displayed on the website.

## Model Performance

- **PlantVillage-style test set:** 97.85% accuracy on 10,948 images.
- **PlantDoc external evaluation:** 20.00% accuracy on 125 mapped images across 15 mapped classes.

The difference in results highlights the need for further testing and improvement on real-world images.



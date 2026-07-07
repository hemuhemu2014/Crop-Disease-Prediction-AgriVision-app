# 🌿 AgriVision AI: Crop Disease Prediction App

AgriVision AI is an intelligent agricultural assistant web application designed to help farmers, agronomists, and researchers diagnose plant diseases instantly. Using a high-performance deep learning model trained on crop leaf imagery, the application identifies **38 distinct health states (diseased and healthy)** across **14 different crop species** and provides recommended treatments.

This project is built using a modern **Streamlit** front-end, a custom CSS glassmorphism styling system, and a **TensorFlow/Keras** deep learning backend.

---

## 📂 Project Architecture

```
Crop-Disease-Prediction-AgriVision-app-main/
│
├── requirements.txt                         # App dependencies
├── README.md                                # Project documentation & methodology
│
└── Crop-Disease-Prediction-AgriVision-app-main/
    ├── streamlit_app.py                      # Main entrypoint for Streamlit UI
    ├── best_model_leaf.h5                    # Serialized TensorFlow/Keras CNN model
    ├── class_indices.json                    # Model index-to-class name mappings (0-37)
    ├── disease_info.json                     # Cure & description mappings for all 38 classes
    │
    ├── Model development/
    │   ├── Model.ipynb                       # Jupyter notebook for model training & evaluation
    │   └── Readme.md                         # Model readme placeholder
    │
    ├── pages/                                # Sidebar-routed Streamlit pages
    │   ├── dashboard.py                      # Home dashboard, mock stats & agricultural overview
    │   ├── scan_upload.py                    # Core page for image upload, inference & cure retrieval
    │   ├── crop_yield_prediction.py          # Yield prediction page (Under Construction)
    │   ├── disease_library.py                # Reference catalog of diseases & treatments
    │   ├── tips_best_practices.py            # Seasonal guides, soil health tips & educational videos
    │   └── faq_help.py                       # User guidelines, limitations & support
    │
    └── utils/
        ├── __init__.py
        └── helpers.py                        # Helper scripts for state management, CSS injection & mock data
```

---

## 🔬 Deep Learning Methodology

The machine learning workflow is designed for high accuracy and low latency, making it suitable for edge-device deployments. Below is a detailed breakdown of the methodology employed in [Model.ipynb](file:///c:/Users/Bharath%20B/Downloads/Crop-Disease-Prediction-AgriVision-app-main/Crop-Disease-Prediction-AgriVision-app-main/Model%20development/Model.ipynb).

### 1. Dataset & Input Pipeline
The model is trained on the popular **PlantVillage Dataset**, accessed via `tensorflow_datasets` (`tfds.load`). 
* **Data Volume:** Contains over 54,000 images of crop leaves classified into 38 classes.
* **Target Classes:** Covers 14 crops (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato) in both healthy and diseased states.
* **Data Splitting:** Split into **80% Training** ($43,442$ images) and **20% Validation** ($10,861$ images).

```python
# Train-Validation split definition in Model.ipynb
(ds_train, ds_val), ds_info = tfds.load(
    'plant_village',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True
)
```

### 2. Image Preprocessing & Performance Tuning
To ensure uniform inputs and fast pipelines, images undergo the following preprocessing pipeline:
* **Resizing:** Standardized to $224 \times 224$ pixels to match the input layer of the CNN.
* **Normalization:** Scaled pixel values from the $[0, 255]$ range to $[0.0, 1.0]$ via division by 255.0 to assist optimizer convergence.
* **Prefetching (`tf.data.AUTOTUNE`):** Prefetches data blocks into memory while the GPU processes the current batch, preventing I/O bottlenecks.

```python
IMG_SIZE = 224
BATCH_SIZE = 32

def preprocess(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, label

ds_train = ds_train.map(preprocess).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_val = ds_val.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
```

### 3. Model Architecture
The network leverages **Transfer Learning** using a pre-trained **MobileNetV2** base. MobileNetV2 is selected due to its inverted residuals and linear bottlenecks, making it exceptionally lightweight and fast without sacrificing accuracy.

* **Base Model (Feature Extractor):** MobileNetV2 initialized with pre-trained **ImageNet** weights. The base layers are frozen (`trainable = False`) to retain broad feature extraction capabilities and accelerate training.
* **Global Average Pooling 2D:** Flattens the spatial $7 \times 7 \times 1280$ feature maps from the base model into a $1280$-dimensional vector.
* **Dense Classifier Layer:** 128 nodes with a Rectified Linear Unit (**ReLU**) activation function.
* **Dropout Layer:** A rate of **0.3** is applied to reduce overfitting by randomly disabling 30% of neurons during training.
* **Output Layer:** A Dense layer with **38 units** (corresponding to each plant/disease class) with a **Softmax** activation to output probability distributions.

```python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
```

### 4. Model Training & Optimization
* **Loss Function:** `sparse_categorical_crossentropy` (since training labels are represented as integers).
* **Optimizer:** `adam` (Adaptive Moment Estimation), which dynamically adjusts learning rates for different parameters.
* **Metrics:** Evaluated primarily on `accuracy`.
* **Epochs:** Trained for `10` epochs.

---

## 📈 Evaluation Results

The model achieved strong generalization performance, showing minimal overfitting due to the frozen base weights and dropout regularization:

* **Final Validation Accuracy:** **`95.83%`**
* **Final Validation Loss:** **`0.1323`**

### Classification Report Summary (Weighted Averages)
| Metric | Score |
| :--- | :--- |
| **Precision** | **96%** |
| **Recall** | **96%** |
| **F1-Score** | **96%** |

#### Class-Specific Observations:
* **High Performance (>99% F1-Score):** The model achieved near-perfect scores on distinctive diseases such as *Orange Citrus Greening (Haunglongbing)*, *Corn Common Rust*, *Grape Leaf Blight*, and *Soybean Healthy*.
* **Underrepresented Classes:** Classes with smaller sample counts or subtle features—such as *Tomato Early Blight* (F1-Score: 66%) and *Potato Healthy* (F1-Score: 75%)—showed lower scores, representing opportunities for future data augmentation.

---

## 💻 Streamlit Web Application Features

The web interface is modularized into several dedicated views:

1. **📊 Dashboard (Home):**
   * Displays mock statistics of present-day leaf disease cases (total cases, most common disease, healthy leaves detected).
   * Renders dynamic progress bars detailing the percentage breakdown of disease categories.
   * Promotes weather awareness with real-time agricultural alerts (e.g., rainfall status for plan pruning/irrigation).

2. **📸 Scan / Upload Leaf Image:**
   * Allows uploading images (`.jpg`, `.jpeg`, `.png`).
   * Performs real-time inference on the image and outputs:
     * **Identified Crop Species** (e.g., Tomato, Apple, Grape).
     * **Disease Status** (Healthy or specific disease name).
     * **Prediction Confidence Score** (expressed as a percentage).
   * Displays a **Suggested Cure/Management strategy** pulled directly from [disease_info.json](file:///c:/Users/Bharath%20B/Downloads/Crop-Disease-Prediction-AgriVision-app-main/Crop-Disease-Prediction-AgriVision-app-main/disease_info.json) to guide the farmer on immediate next steps.

3. **📈 Crop Yield Prediction:**
   * Interactive page designed to predict regional yields based on environment/weather data *(currently flagged as Under Construction)*.

4. **📚 Disease Library & Info:**
   * A visual catalog of common plant diseases featuring sample reference images, disease descriptions, and recommended cures.

5. **💡 Tips & Best Practices:**
   * Educational resources containing seasonal farming tips (Rabi, Kharif, Zaid crops), soil health/crop care suggestions, natural pest management strategies (e.g., Neem oil, intercropping), and embedded educational videos.

6. **❓ FAQ / Help:**
   * Answers operational questions (e.g., how to take a high-quality picture for accurate scan results) and outlines the technical limitations of the system.

---

## 🚀 Setup & Installation

Follow these steps to run the application locally on your machine.

### Prerequisites
Make sure you have **Python 3.8+** installed.

### 1. Clone or Navigate to the Workspace Directory
```bash
cd "Crop-Disease-Prediction-AgriVision-app-main"
```

### 2. Install Dependencies
Install all the required backend libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
Navigate into the main app directory (where `streamlit_app.py` resides) and run the command:
```bash
cd "Crop-Disease-Prediction-AgriVision-app-main"
streamlit run streamlit_app.py
```

After running the command, your default web browser will automatically open the app at `http://localhost:8501`.

---

## 🛠 Tech Stack Used
* **Framework:** Streamlit (UI & Page Routing)
* **Deep Learning Library:** TensorFlow / Keras (Inference)
* **Image Processing:** Pillow (PIL), NumPy
* **Scientific Computing & Plotting (Training):** Matplotlib, Seaborn, Scikit-learn
* **Styling:** Custom CSS (Glassmorphism card effects, custom buttons, custom fonts)

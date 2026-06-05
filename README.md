# Real-Time Facial Emotion Detection Using CNN and OpenCV

This repository features an end-to-end machine learning pipeline that detects human facial expressions in real-time. The project utilizes a custom **Deep Convolutional Neural Network (CNN)** built with TensorFlow/Keras to classify emotions and leverages **OpenCV's Haar Cascade** classifier for live face tracking via webcam.

---

## 🚀 Features

- **Automated Data Processing:** Loads, grayscales, resizes ($48 \times 48$ pixels), and normalizes images efficiently.
- **Data Augmentation:** Features on-the-fly image transformations (rotation, zoom, horizontal flips) during training to mitigate overfitting.
- **Deep CNN Architecture:** Implements a powerful network structure utilizing multi-layered convolutions, Batch Normalization, Max Pooling, and Dropout for robust feature extraction.
- **Live Frame-Optimized Inference:** Intercepts real-time webcam video feeds, processes faces every 3rd frame to optimize performance, and draws dynamic bounding boxes with real-time accuracy percentages.

---

## 📊 Model Architecture

The deep neural network is sequentially structured with increasing filter sizes to learn hierarchical facial representations:

| Layer (Type) | Configuration | Hyperparameters / Activation |
| :--- | :--- | :--- |
| **Convolutional Block 1** | 2x Conv2D (64 filters, $3 \times 3$) | ReLU, Batch Normalization, MaxPool ($2 \times 2$), Dropout (0.25) |
| **Convolutional Block 2** | 2x Conv2D (128 filters, $3 \times 3$) | ReLU, Batch Normalization, MaxPool ($2 \times 2$), Dropout (0.25) |
| **Convolutional Block 3** | 2x Conv2D (256 filters, $3 \times 3$) | ReLU, Batch Normalization, MaxPool ($2 \times 2$), Dropout (0.25) |
| **Fully Connected Block** | Flatten $\rightarrow$ Dense (512) $\rightarrow$ Dense (256) | ReLU, Batch Normalization, Dropout (0.5) |
| **Output Layer** | Dense (7) | Softmax |

### Training Configuration
- **Optimizer:** Adam (Learning Rate: 0.001)
- **Loss Function:** `sparse_categorical_crossentropy`
- **Callbacks Hooked:** - `ModelCheckpoint` (Saves the absolute highest `val_accuracy` model configuration to `best_emotion_model.keras`)
  - `EarlyStopping` (Monitors validation loss with a patience of 10 epochs)
  - `ReduceLROnPlateau` (Halves the learning rate when validation loss plateaus for 5 epochs)

---

## 📂 Dataset Setup

To train this pipeline, set up your project directory structure as follows:

```text
├── Emotion_dataset/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── test/
│       ├── angry/
│       └── ... (same subfolders as training set)
├── Haar_code.ipynb
└── README.md

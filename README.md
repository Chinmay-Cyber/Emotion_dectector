# Emotion Detector 

A real-time facial emotion recognition system built with TensorFlow and MobileNetV2. Detects 7 emotions from a webcam feed using transfer learning and MTCNN face detection.

## Features

- Classifies 7 emotions: **angry, disgust, fear, happy, neutral, sad, surprise**
- Two-phase transfer learning with a MobileNetV2 backbone (pretrained on ImageNet)
- Handles class imbalance via computed class weights
- Real-time webcam inference using MTCNN face detection
- Color-coded bounding boxes per emotion with confidence scores and a top-3 overlay
- Saves training history plots and a confusion matrix to `results/`

## Project Structure

```
emotion_detector_v2.ipynb   # Main notebook (training + inference)
best_emotion_model_v2.keras # Saved model checkpoint (generated after training)
results/
  confusion_matrix.png      # Confusion matrix heatmap
  training_history.png      # Accuracy & loss curves across both phases
```

## Requirements

```
tensorflow
scikit-learn
numpy
matplotlib
seaborn
opencv-python
mtcnn
```

Install everything at once:

```bash
pip install tensorflow scikit-learn numpy matplotlib seaborn opencv-python mtcnn
```

A CUDA-capable GPU is recommended for training but not required for inference.

## Dataset Setup

The notebook expects your dataset organised into per-emotion subfolders:

```
train/
  angry/
  disgust/
  fear/
  happy/
  neutral/
  sad/
  surprise/
test/
  angry/
  ...
```

Set `TRAIN_PATH` and `TEST_PATH` in **Cell 3** to point to your local folders. The [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset (48×48 grayscale, ~35k images) works out of the box.

## Usage

### Training

Open `emotion_detector_v2.ipynb` and run cells in order:

1. **Cell 2** – imports and GPU check
2. **Cell 3** – set your paths and hyperparameters
3. **Cells 4–5** – load images and compute class weights
4. **Cell 6** – build the MobileNetV2 model
5. **Cell 7** – configure data augmentation
6. **Cell 8** – Phase 1: train the classification head (backbone frozen, 50 epochs max)
7. **Cell 9** – Phase 2: fine-tune the full model (backbone unfrozen, lower LR)
8. **Cells 10–11** – evaluate and save plots

The best checkpoint is saved automatically to `best_emotion_model_v2.keras` based on validation accuracy.

### Real-Time Detection

Run **Cell 12** after training (or point `MODEL_PATH` to an existing checkpoint). Press `q` to quit the webcam window.

```python
MODEL_PATH = 'best_emotion_model_v2.keras'  # update if needed
```

## Model Architecture

| Component | Detail |
|---|---|
| Backbone | MobileNetV2 (ImageNet weights, frozen in Phase 1) |
| Pooling | GlobalAveragePooling2D |
| Head | Dense 256 → BN → Dropout 0.4 → Dense 128 → BN → Dropout 0.3 → Softmax |
| Phase 1 LR | 0.001 (head only) |
| Phase 2 LR | 1e-5 (full model) |
| Input size | 48 × 48 × 3 |
| Batch size | 64 |

Early stopping and `ReduceLROnPlateau` are applied in both phases to prevent overfitting.

## Training Details

- **Augmentation:** random rotation (±15°), zoom (15%), horizontal flip, shift (10%), brightness jitter (0.8–1.2×)
- **Class imbalance:** handled with `sklearn.utils.class_weight.compute_class_weight`
- **Face detection (inference):** MTCNN with confidence threshold 0.90, run every 5 frames for performance

## Configuration

All key hyperparameters are in **Cell 3**:

| Variable | Default | Description |
|---|---|---|
| `IMG_SIZE` | 48 | Input image resolution |
| `BATCH_SIZE` | 64 | Training batch size |
| `EPOCHS` | 100 | Max epochs for Phase 2 |
| `MODEL_PATH` | `best_emotion_model_v2.keras` | Checkpoint save path |
| `RESULTS_DIR` | `results` | Output directory for plots |

## License

MIT

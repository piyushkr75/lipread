# Visual Speech Recognition (Lip Reading) with MediaPipe & CNN + BiLSTM

An end-to-end Visual Speech Recognition (VSR) system using deep learning to interpret spoken words and phrases from video lip movements.

---

## Key Highlights & Updated Architecture

- **Face & Mouth Detection:** Powered by **MediaPipe Face Mesh** (`FACEMESH_LIPS`), extracting dynamic lip landmarks frame-by-frame with adaptive margin padding and temporal fallback.
- **Robust Preprocessing:**
  - Grayscale conversion.
  - **Aspect-ratio-preserving resize** (letterboxing) to prevent non-linear mouth distortion.
  - Symmetric zero-padding to target canvas dimensions `(64, 96)`.
  - Z-score normalization (`(frame - mean) / std`).
- **Temporal Sequence Modeling:** Maintains chronological frame order over variable-length recordings ($T \le 75$) with zero-padding and masking.
- **Deep Neural Architecture:** Spatiotemporal **3D CNN** (or TimeDistributed 2D CNN) front-end paired with stacked **Bidirectional LSTMs (BiLSTM)**.
- **Classification Paradigm:** Trains on both **word** and **phrase** samples, treating each complete recording as an atomic classification sample with temporal pooling and Softmax Cross-Entropy loss.

---

## Project Structure

```
Visual_Speech_Recognition/
├── pipeline/
│   ├── __init__.py
│   ├── lip_preprocessor.py        # MediaPipe Face Mesh, dynamic bbox, aspect resize, pad, norm
│   ├── dataset_builder.py         # Word & phrase sample extraction, vocab encoding, tf.data pipeline
│   └── model.py                   # CNN + BiLSTM architectures (Classifier & CTC)
├── data/
│   ├── s1/                        # Speaker video recordings (.mpg)
│   └── alignments/s1/             # Alignment files with word boundaries (.align)
├── models/                        # Saved checkpoints and vocabularies
├── train_classifier.py            # Standalone training script for word/phrase classification
├── demo_mediapipe.py              # Inference demo with dynamic lip tracking & top-5 predictions
├── demo.py                        # Legacy CTC demo script
├── main_working_earlystop.ipynb   # Interactive training and evaluation notebook
└── README.md
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/lip-reading.git
   cd lip-reading/Visual_Speech_Recognition
   ```

2. **Install required dependencies:**
   ```bash
   pip install opencv-python mediapipe tensorflow matplotlib gdown numpy
   ```

---

## How to Run

### 1. Training the Word & Phrase Classifier

To train the CNN + BiLSTM model on word and phrase samples:
```bash
python train_classifier.py --data_dir data --speaker s1 --epochs 50 --batch_size 8 --cnn_type 3d
```

Key arguments:
- `--cnn_type`: Choose between `3d` (Conv3D spatiotemporal) or `2d` (TimeDistributed Conv2D).
- `--epochs`: Total epochs (default: `50`, with Early Stopping).
- `--batch_size`: Batch size (default: `8`).
- `--max_samples`: Optional limit for quick prototyping.

### 2. Running Inference / Demo

Run the dynamic MediaPipe inference script on any video:
```bash
python demo_mediapipe.py data/s1/bbaf2n.mpg
```
The script will dynamically track lip movements, run the model, and print top candidates with confidence percentages.

### 3. Interactive Jupyter Notebook

Launch Jupyter and open `main_working_earlystop.ipynb`:
```bash
jupyter notebook main_working_earlystop.ipynb
```
Follow the step-by-step cells for visual verification of crops, dataset generation, model compilation, training with early stopping, and evaluation.

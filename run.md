# How to Run the Lip Reading (Visual Speech Recognition) Project

This guide provides step-by-step instructions for:
1. **Running inference when the model is ALREADY TRAINED** (Quick Start).
2. **Displaying both Original (Ground Truth) and Predicted text**.
3. **Training the model from scratch** (CLI or Jupyter Notebook).

---

## ⚡ Quick Start: Running with an ALREADY TRAINED Model

Your repository already includes pre-trained weights in `models/checkpoint.weights.h5`. You can run inference immediately.

### Step 1: Activate Your Environment
From the `Visual_Speech_Recognition` folder:
- **Windows (PowerShell):**
  ```powershell
  cd Visual_Speech_Recognition
  .\venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  cd Visual_Speech_Recognition
  source venv/bin/activate
  ```

### Step 2: Run Prediction on Any Video File
Run the demo script with a video path as an argument:
```bash
python demo_mediapipe.py data/s1/bbaf2n.mpg
```
*(Or run `python demo.py data/s1/bbaf2n.mpg`)*

### Step 3: Output Display (Original vs. Predicted)
The script automatically finds the corresponding `.align` file, extracts the ground-truth spoken sentence, and displays both the original and predicted text side-by-side:

```text
=================================================================
LIP READING INFERENCE (Pre-Trained LipNet CTC Model)
Source       : data/s1/bbaf2n.mpg
Model Weights: models/checkpoint.weights.h5
=================================================================
Pre-trained LipNet model weights loaded successfully.

=================================================================
LIP READING RESULT
=================================================================
Video File : data/s1/bbaf2n.mpg
Original   : bin blue at f two now
Predicted  : bin blue at two now
=================================================================
```

### Step 4: Interactive Mode
You can also run without arguments and input the path interactively:
```bash
python demo_mediapipe.py
```
```text
Enter video path (example: data/s1/bbaf2n.mpg): data/s1/bbaf2n.mpg
```

---

## 🛠️ Environment Setup & Installation (First Time Setup)

### 1. Check Python
Python 3.9 through 3.12 is recommended:
```bash
python --version
```

### 2. Create and Activate Virtual Environment
```bash
cd Visual_Speech_Recognition
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install opencv-python mediapipe tensorflow matplotlib gdown numpy
```

---

## 📁 Dataset Layout

```
Visual_Speech_Recognition/
└── data/
    ├── s1/                        # Video recordings (.mpg or .mp4)
    │   ├── bbaf2n.mpg
    │   └── ...
    └── alignments/
        └── s1/                    # Alignment timestamps (.align)
            ├── bbaf2n.align
            └── ...
```

---

## 🏋️ Training the Model from Scratch

If you want to train the new CNN + BiLSTM model on word and phrase samples:

### Option A: Standalone CLI Training

1. **Quick Smoke Test** (30 video files to verify training pipeline in ~2 minutes):
   ```bash
   python train_classifier.py --data_dir data --speaker s1 --epochs 3 --batch_size 4 --max_samples 30
   ```

2. **Full Training Run:**
   ```bash
   python train_classifier.py --data_dir data --speaker s1 --epochs 50 --batch_size 8 --cnn_type 3d
   ```

### Key Training Arguments:
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--data_dir` | `data` | Base directory containing datasets |
| `--speaker` | `s1` | Speaker subfolder name |
| `--epochs` | `50` | Maximum number of training epochs |
| `--batch_size` | `8` | Number of samples per batch |
| `--lr` | `0.0001` | Initial learning rate |
| `--cnn_type` | `3d` | Front-end CNN type: `3d` (Conv3D) or `2d` (TimeDistributed Conv2D) |
| `--max_frames` | `75` | Fixed temporal window length per sample |
| `--output_dir` | `models` | Output directory for weights and vocab |
| `--max_samples` | `None` | Limit video count for quick testing |

### Option B: Jupyter Notebook Training
1. Start Jupyter:
   ```bash
   jupyter notebook main_working_earlystop.ipynb
   ```
2. Run cells sequentially to visually verify crops and train the model with Early Stopping.

---

## ❓ Troubleshooting

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'tensorflow'` | Script executed outside virtual environment | Activate the venv first (`.\venv\Scripts\Activate.ps1`) or run `.\venv\Scripts\python.exe <script>`. |
| `Original : N/A (No alignment file found)` | Custom or unaligned video | Normal for user-recorded videos without a corresponding `.align` file. |
| `ResourceExhaustedError: OOM on GPU` | GPU memory exceeded | Set `--batch_size 4` or `--cnn_type 2d`. |

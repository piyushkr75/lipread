import os
import sys
import json
import time
import warnings

# Suppress noisy TensorFlow and system warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import tensorflow as tf

from pipeline.lip_preprocessor import DynamicLipExtractor, preprocess_lip_roi
from pipeline.model import build_cnn_bilstm_classifier


def get_ground_truth_text(video_path: str) -> str:
    """
    Retrieves the original (ground truth) spoken text from the corresponding .align file.
    """
    basename = os.path.splitext(os.path.basename(video_path))[0]
    speaker_dir = os.path.basename(os.path.dirname(video_path))
    parent_dir = os.path.dirname(os.path.dirname(video_path))

    candidates = [
        os.path.join(parent_dir, "alignments", speaker_dir, f"{basename}.align"),
        os.path.join("data", "alignments", "s1", f"{basename}.align"),
        os.path.join("data", "alignments", speaker_dir, f"{basename}.align"),
        os.path.join(os.path.dirname(video_path), f"{basename}.align"),
    ]

    for c in candidates:
        norm_c = os.path.normpath(c)
        if os.path.exists(norm_c):
            tokens = []
            try:
                with open(norm_c, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3 and parts[2].lower() != "sil":
                            tokens.append(parts[2].lower())
                if tokens:
                    return " ".join(tokens)
            except Exception:
                pass

    return "N/A (No alignment file found)"


def print_comparison_result(video_path: str, original_text: str, predicted_text: str, confidence: float = None):
    print("\n" + "=" * 65)
    print("LIP READING RESULT")
    print("=" * 65)
    print(f"Video File : {video_path}")
    print(f"Original   : {original_text}")
    print(f"Predicted  : {predicted_text}")
    if confidence is not None:
        print(f"Confidence : {confidence:.2f}%")
    print("=" * 65)


def run_legacy_ctc_inference(video_path: str, weights_path: str = "models/checkpoint.weights.h5"):
    """
    Runs inference using the pre-trained LipNet CTC checkpoint.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv3D, LSTM, Dense, Dropout, Bidirectional,
        MaxPool3D, Activation, TimeDistributed, Flatten
    )

    vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
    char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
    num_to_char = tf.keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(),
        oov_token="",
        invert=True
    )

    print("=" * 65)
    print("LIP READING INFERENCE (Pre-Trained LipNet CTC Model)")
    print(f"Source       : {video_path}")
    print(f"Model Weights: {weights_path}")
    print("=" * 65)

    model = Sequential([
        Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding="same"),
        Activation("relu"),
        MaxPool3D(pool_size=(1, 2, 2)),
        Conv3D(256, 3, padding="same"),
        Activation("relu"),
        MaxPool3D(pool_size=(1, 2, 2)),
        Conv3D(75, 3, padding="same"),
        Activation("relu"),
        MaxPool3D(pool_size=(1, 2, 2)),
        TimeDistributed(Flatten()),
        Bidirectional(LSTM(128, kernel_initializer="orthogonal", return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(128, kernel_initializer="orthogonal", return_sequences=True)),
        Dropout(0.5),
        Dense(char_to_num.vocabulary_size() + 1, kernel_initializer="he_normal", activation="softmax")
    ])

    model.load_weights(weights_path)
    print("Pre-trained LipNet model weights loaded successfully.")

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        try:
            if len(frame.shape) != 3 or frame.shape[0] < 236 or frame.shape[1] < 220:
                continue

            frame_tensor = tf.convert_to_tensor(frame, dtype=tf.uint8)
            gray = tf.image.rgb_to_grayscale(frame_tensor)
            crop = gray[190:236, 80:220, :]

            if tuple(crop.shape) == (46, 140, 1):
                frames.append(crop)
        except Exception:
            continue

    cap.release()

    if not frames:
        raise ValueError(f"No valid frames found in {video_path}")

    frames = tf.stack(frames)
    frames = tf.cast(frames, tf.float32)
    mean = tf.reduce_mean(frames)
    std = tf.math.reduce_std(frames)
    frames = (frames - mean) / (std + 1e-8)

    # Pad or slice to 75
    frames = frames[:75]
    if len(frames) < 75:
        padding = tf.zeros((75 - len(frames), 46, 140, 1), dtype=tf.float32)
        frames = tf.concat([frames, padding], axis=0)

    frames = tf.expand_dims(frames, axis=0)

    # Predict with CTC
    yhat = model.predict(frames, verbose=0)
    decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
    prediction = tf.strings.reduce_join(
        [num_to_char(word) for word in decoded[0]]
    ).numpy().decode("utf-8").strip()

    original = get_ground_truth_text(video_path)
    print_comparison_result(video_path, original, prediction)
    return prediction


def run_classifier_inference(
    video_source: str,
    weights_path: str = "models/best_classifier.weights.h5",
    vocab_path: str = "models/vocab_classes.json",
    max_frames: int = 75,
    target_size: tuple = (64, 96),
    cnn_type: str = "3d"
):
    """
    Runs inference using the word/phrase classifier model.
    """
    if not os.path.exists(vocab_path):
        from pipeline.dataset_builder import WordPhraseDatasetBuilder
        builder = WordPhraseDatasetBuilder(data_dir="data", speaker="s1")
        builder.scan_dataset(include_words=True, include_phrases=True)
        builder.save_vocabulary(vocab_path)

    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    classes = data["classes"]
    idx_to_class = {int(v): k for k, v in data["class_to_idx"].items()}

    print("=" * 65)
    print("LIP READING INFERENCE (Word & Phrase Classifier)")
    print(f"Source       : {video_source}")
    print(f"Model Weights: {weights_path}")
    print("=" * 65)

    model = build_cnn_bilstm_classifier(
        input_shape=(max_frames, target_size[0], target_size[1], 1),
        num_classes=len(classes),
        cnn_type=cnn_type
    )
    model.load_weights(weights_path)
    print("Trained classifier model loaded successfully.")

    extractor = DynamicLipExtractor()
    cap = cv2.VideoCapture(video_source)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        roi = extractor.extract_lip_roi(frame)
        processed = preprocess_lip_roi(roi, target_size=target_size)
        frames.append(processed)

    cap.release()

    frames = frames[:max_frames]
    if len(frames) < max_frames:
        pad_len = max_frames - len(frames)
        pad = np.zeros((pad_len, target_size[0], target_size[1], 1), dtype=np.float32)
        frames = np.concatenate([np.array(frames, dtype=np.float32), pad], axis=0)
    else:
        frames = np.array(frames, dtype=np.float32)

    input_batch = np.expand_dims(frames, axis=0)
    probabilities = model.predict(input_batch, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:5]

    top_pred_class = idx_to_class[top_indices[0]]
    top_confidence = probabilities[top_indices[0]] * 100.0

    original = get_ground_truth_text(video_source)
    print_comparison_result(video_source, original, top_pred_class, top_confidence)

    print("\nTop 5 Candidates:")
    for rank, idx in enumerate(top_indices, start=1):
        c_name = idx_to_class[idx]
        conf = probabilities[idx] * 100.0
        print(f"  {rank}. {c_name:<30} {conf:6.2f}%")
    print("=" * 65)

    return top_pred_class


def main():
    if len(sys.argv) > 1:
        source = sys.argv[1].strip()
    else:
        source = input("\nEnter video path (example: data/s1/bbaf2n.mpg): ").strip()

    if not os.path.exists(source) and source not in ["0", "webcam"]:
        print(f"Error: Video file '{source}' does not exist.")
        sys.exit(1)

    classifier_weights = "models/best_classifier.weights.h5"
    legacy_weights = "models/checkpoint.weights.h5"

    if os.path.exists(classifier_weights):
        run_classifier_inference(source, weights_path=classifier_weights)
    elif os.path.exists(legacy_weights):
        run_legacy_ctc_inference(source, weights_path=legacy_weights)
    else:
        print("Error: No trained model weights found in models/.")
        print("Expected either models/best_classifier.weights.h5 or models/checkpoint.weights.h5")
        sys.exit(1)


if __name__ == "__main__":
    main()

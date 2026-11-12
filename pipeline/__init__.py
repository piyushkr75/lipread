"""
Lip reading pipeline package for Visual Speech Recognition.
Includes MediaPipe Face Mesh dynamic lip extraction, aspect-ratio-preserving preprocessing,
word and phrase dataset handling, and CNN + BiLSTM models.
"""

from .lip_preprocessor import DynamicLipExtractor, preprocess_lip_roi, process_video_sequence
from .dataset_builder import WordPhraseDatasetBuilder
from .model import build_cnn_bilstm_classifier, build_ctc_model

__all__ = [
    "DynamicLipExtractor",
    "preprocess_lip_roi",
    "process_video_sequence",
    "WordPhraseDatasetBuilder",
    "build_cnn_bilstm_classifier",
    "build_ctc_model",
]


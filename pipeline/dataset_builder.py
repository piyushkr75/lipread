import os
import glob
import json
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
from .lip_preprocessor import process_video_sequence, DynamicLipExtractor


class WordPhraseDatasetBuilder:
    """
    Parses and builds datasets for both word and phrase samples.
    Treats each complete word or phrase recording as one classification sample.
    """

    def __init__(
        self,
        data_dir: str = "data",
        speaker: str = "s1",
        target_size: tuple = (64, 96),
        max_frames: int = 75,
        fps: float = 25.0
    ):
        self.data_dir = data_dir
        self.speaker = speaker
        self.target_size = target_size
        self.max_frames = max_frames
        self.fps = fps

        self.video_dir = os.path.join(data_dir, speaker)
        self.align_dir = os.path.join(data_dir, "alignments", speaker)

        self.samples: List[Dict] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

    def parse_align_file(self, align_path: str) -> Tuple[str, List[Tuple[int, int, str]]]:
        """
        Parses a GRID alignment file.
        Returns:
            full_phrase: Full spoken phrase string.
            words: List of (start_frame, end_frame, word_label) tuples.
        """
        words = []
        full_phrase_tokens = []

        if not os.path.exists(align_path):
            return "", []

        with open(align_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start_time = float(parts[0])
                    end_time = float(parts[1])
                    label = parts[2].lower()

                    if label != "sil":
                        # GRID alignments are in 1/1000th second units (25 fps -> frame = time / 1000)
                        start_frame = max(0, int(round(start_time / 1000.0)))
                        end_frame = max(start_frame + 1, int(round(end_time / 1000.0)))
                        words.append((start_frame, end_frame, label))
                        full_phrase_tokens.append(label)

        full_phrase = " ".join(full_phrase_tokens)
        return full_phrase, words

    def scan_dataset(
        self,
        include_words: bool = True,
        include_phrases: bool = True,
        max_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Scans video and alignment directories to collect word and phrase samples.
        Each sample dict contains:
        {
            'sample_type': 'word' | 'phrase',
            'video_path': str,
            'start_frame': int,
            'end_frame': int,
            'label': str
        }
        """
        video_files = glob.glob(os.path.join(self.video_dir, "*.mpg"))
        if not video_files:
            video_files = glob.glob(os.path.join(self.video_dir, "*.mp4"))

        print(f"Discovered {len(video_files)} video recordings in {self.video_dir}")
        self.samples = []
        unique_labels = set()

        count = 0
        for vid_path in video_files:
            basename = os.path.splitext(os.path.basename(vid_path))[0]
            align_path = os.path.join(self.align_dir, f"{basename}.align")

            full_phrase, words = self.parse_align_file(align_path)

            # 1. Complete Phrase Sample
            if include_phrases and full_phrase:
                self.samples.append({
                    "sample_type": "phrase",
                    "video_path": vid_path,
                    "start_frame": 0,
                    "end_frame": None,
                    "label": full_phrase
                })
                unique_labels.add(full_phrase)

            # 2. Complete Word Samples
            if include_words and words:
                for start_f, end_f, w_label in words:
                    self.samples.append({
                        "sample_type": "word",
                        "video_path": vid_path,
                        "start_frame": start_f,
                        "end_frame": end_f,
                        "label": w_label
                    })
                    unique_labels.add(w_label)

            count += 1
            if max_samples and count >= max_samples:
                break

        self.classes = sorted(list(unique_labels))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}

        print(f"Total collected samples: {len(self.samples)} (Unique classes: {len(self.classes)})")
        return self.samples

    def save_vocabulary(self, output_path: str = "vocab_classes.json"):
        """Saves class-to-index mapping to JSON file."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "classes": self.classes,
                "class_to_idx": self.class_to_idx,
                "num_classes": len(self.classes)
            }, f, indent=2)
        print(f"Vocabulary saved to {output_path}")

    def load_vocabulary(self, vocab_path: str = "vocab_classes.json"):
        """Loads class-to-index mapping from JSON file."""
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.classes = data["classes"]
        self.class_to_idx = data["class_to_idx"]
        self.idx_to_class = {int(v): k for k, v in self.class_to_idx.items()}
        print(f"Loaded {len(self.classes)} classes from {vocab_path}")

    def create_tf_dataset(
        self,
        samples: List[Dict],
        batch_size: int = 8,
        shuffle: bool = True
    ) -> tf.data.Dataset:
        """
        Creates a high-performance tf.data.Dataset pipeline.
        Yields (batch_frames, batch_labels) where:
            batch_frames: (batch_size, max_frames, target_h, target_w, 1)
            batch_labels: (batch_size,) integer class indices
        """
        extractor = DynamicLipExtractor()

        def generator():
            indices = list(range(len(samples)))
            if shuffle:
                np.random.shuffle(indices)

            for i in indices:
                sample = samples[i]
                try:
                    frames = process_video_sequence(
                        video_path=sample["video_path"],
                        start_frame=sample["start_frame"],
                        end_frame=sample["end_frame"],
                        max_frames=self.max_frames,
                        target_size=self.target_size,
                        extractor=extractor
                    )
                    label_idx = self.class_to_idx[sample["label"]]
                    yield frames, np.int32(label_idx)
                except Exception as e:
                    continue

        output_signature = (
            tf.TensorSpec(shape=(self.max_frames, self.target_size[0], self.target_size[1], 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )

        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset


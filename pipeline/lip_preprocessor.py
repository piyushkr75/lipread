import os
import cv2
import numpy as np

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    # Derive unique canonical lip indices from MediaPipe's mesh graph
    LIP_INDICES = sorted(list(set(
        idx for edge in mp_face_mesh.FACEMESH_LIPS for idx in edge
    )))
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError):
    mp = None
    mp_face_mesh = None
    # Canonical MediaPipe 468-point lip landmark indices (inner and outer lips)
    LIP_INDICES = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324
    ]
    LIP_INDICES = sorted(list(set(LIP_INDICES)))
    MEDIAPIPE_AVAILABLE = False


class DynamicLipExtractor:
    """
    Extracts dynamic lip region-of-interest (ROI) using MediaPipe Face Mesh.
    Calculates dynamic bounding box for every frame with adaptive margin and
    temporal fallback when detection briefly fails.
    """
    def __init__(
        self,
        margin_x: float = 0.25,
        margin_y: float = 0.35,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self.margin_x = margin_x
        self.margin_y = margin_y
        self.last_bbox = None

        if MEDIAPIPE_AVAILABLE:
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        else:
            self.face_mesh = None

    def reset(self):
        """Reset temporal state between videos."""
        self.last_bbox = None

    def extract_lip_roi(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Calculates dynamic bounding box of lips and returns cropped ROI.
        Falls back to last known bounding box or center mouth crop if face detection drops.
        """
        h, w = frame_bgr.shape[:2]

        if self.face_mesh is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                lip_pts = np.array([
                    [landmarks[idx].x * w, landmarks[idx].y * h]
                    for idx in LIP_INDICES
                ])

                min_x, min_y = np.min(lip_pts, axis=0)
                max_x, max_y = np.max(lip_pts, axis=0)

                box_w = max_x - min_x
                box_h = max_y - min_y
                pad_x = int(box_w * self.margin_x)
                pad_y = int(box_h * self.margin_y)

                x1 = max(0, int(min_x - pad_x))
                y1 = max(0, int(min_y - pad_y))
                x2 = min(w, int(max_x + pad_x))
                y2 = min(h, int(max_y + pad_y))

                # Smooth tracking: blend with last bbox if available
                if self.last_bbox is not None:
                    alpha = 0.8
                    x1 = int(alpha * x1 + (1 - alpha) * self.last_bbox[0])
                    y1 = int(alpha * y1 + (1 - alpha) * self.last_bbox[1])
                    x2 = int(alpha * x2 + (1 - alpha) * self.last_bbox[2])
                    y2 = int(alpha * y2 + (1 - alpha) * self.last_bbox[3])

                self.last_bbox = (x1, y1, x2, y2)
                return frame_bgr[y1:y2, x1:x2]

        # Fallback if detection missed this frame but we have prior tracking
        if self.last_bbox is not None:
            x1, y1, x2, y2 = self.last_bbox
            return frame_bgr[y1:y2, x1:x2]

        # Safe default heuristic crop for standard speaker videos
        y1, y2 = int(h * 0.60), int(h * 0.88)
        x1, x2 = int(w * 0.25), int(w * 0.75)
        return frame_bgr[y1:y2, x1:x2]


def preprocess_lip_roi(
    roi: np.ndarray,
    target_size: tuple = (64, 96)
) -> np.ndarray:
    """
    Applies:
    1. Grayscale conversion.
    2. Aspect-ratio-preserving resize.
    3. Symmetric zero-padding to target canvas (H, W).
    4. Normalisation (zero mean, unit variance standardisation).

    Returns: (target_height, target_width, 1) float32 array.
    """
    target_h, target_w = target_size

    if roi is None or roi.size == 0:
        return np.zeros((target_h, target_w, 1), dtype=np.float32)

    # 1. Grayscale conversion
    if len(roi.shape) == 3 and roi.shape[2] == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    elif len(roi.shape) == 3 and roi.shape[2] == 1:
        gray = roi.squeeze(-1)
    else:
        gray = roi

    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, 1), dtype=np.float32)

    # 2. Aspect-ratio-preserving resize
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 3. Symmetric zero-padding
    canvas = np.zeros((target_h, target_w), dtype=np.float32)
    start_y = (target_h - new_h) // 2
    start_x = (target_w - new_w) // 2
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized

    # 4. Normalisation (zero-mean, unit-variance)
    mean = np.mean(canvas)
    std = np.std(canvas) + 1e-7
    normalized = (canvas - mean) / std

    return np.expand_dims(normalized, axis=-1).astype(np.float32)


def process_video_sequence(
    video_path: str,
    start_frame: int = 0,
    end_frame: int = None,
    max_frames: int = 75,
    target_size: tuple = (64, 96),
    extractor: DynamicLipExtractor = None
) -> np.ndarray:
    """
    Loads video, extracts dynamic lip ROIs per frame, preprocesses each ROI,
    preserves temporal ordering, and pads/truncates to max_frames.

    Returns: (max_frames, target_h, target_w, 1) float32 array.
    """
    if extractor is None:
        extractor = DynamicLipExtractor()
    else:
        extractor.reset()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_idx >= start_frame:
            if end_frame is not None and frame_idx > end_frame:
                break

            roi = extractor.extract_lip_roi(frame)
            processed = preprocess_lip_roi(roi, target_size=target_size)
            frames.append(processed)

        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        # Return zeros if no valid frames
        return np.zeros((max_frames, target_size[0], target_size[1], 1), dtype=np.float32)

    # Temporal truncation or padding to fixed max_frames
    frames = frames[:max_frames]
    seq_len = len(frames)

    if seq_len < max_frames:
        pad_len = max_frames - seq_len
        pad_frames = np.zeros(
            (pad_len, target_size[0], target_size[1], 1),
            dtype=np.float32
        )
        frames = np.concatenate([np.array(frames, dtype=np.float32), pad_frames], axis=0)
    else:
        frames = np.array(frames, dtype=np.float32)

    return frames


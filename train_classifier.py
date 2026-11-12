import os
import argparse
import json
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from pipeline.dataset_builder import WordPhraseDatasetBuilder
from pipeline.model import build_cnn_bilstm_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train Word & Phrase Lip Reading Classifier")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--speaker", type=str, default="s1", help="Speaker folder (e.g., s1)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--max_frames", type=int, default=75, help="Max temporal frames per sample")
    parser.add_argument("--img_height", type=int, default=64, help="Target lip height")
    parser.add_argument("--img_width", type=int, default=96, help="Target lip width")
    parser.add_argument("--cnn_type", type=str, default="3d", choices=["3d", "2d"], help="CNN front-end type")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of video files for quick tests")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("LIP READING CLASSIFIER - TRAINING PIPELINE")
    print(f"Data directory: {args.data_dir}/{args.speaker}")
    print(f"CNN Front-end : {args.cnn_type.upper()}")
    print(f"Target size   : ({args.img_height}, {args.img_width})")
    print(f"Max frames    : {args.max_frames}")
    print("=" * 70)

    # 1. Build and scan dataset for both word and phrase samples
    builder = WordPhraseDatasetBuilder(
        data_dir=args.data_dir,
        speaker=args.speaker,
        target_size=(args.img_height, args.img_width),
        max_frames=args.max_frames
    )

    all_samples = builder.scan_dataset(
        include_words=True,
        include_phrases=True,
        max_samples=args.max_samples
    )

    if not all_samples:
        raise RuntimeError("No samples found. Please check data directory path.")

    # Save vocabulary for inference
    vocab_path = os.path.join(args.output_dir, "vocab_classes.json")
    builder.save_vocabulary(vocab_path)
    num_classes = len(builder.classes)

    # 2. Train / Validation split (85% train, 15% val)
    import random
    random.seed(42)
    random.shuffle(all_samples)

    split_idx = int(0.85 * len(all_samples))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"Dataset split: {len(train_samples)} train samples | {len(val_samples)} val samples")

    # 3. Create tf.data.Dataset streams
    train_ds = builder.create_tf_dataset(train_samples, batch_size=args.batch_size, shuffle=True)
    val_ds = builder.create_tf_dataset(val_samples, batch_size=args.batch_size, shuffle=False)

    # 4. Build Model
    model = build_cnn_bilstm_classifier(
        input_shape=(args.max_frames, args.img_height, args.img_width, 1),
        num_classes=num_classes,
        cnn_type=args.cnn_type,
        lstm_units=128,
        dropout_rate=0.4
    )

    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # 5. Callbacks
    checkpoint_filepath = os.path.join(args.output_dir, "best_classifier.weights.h5")
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 6. Fit
    steps_per_epoch = max(1, len(train_samples) // args.batch_size)
    val_steps = max(1, len(val_samples) // args.batch_size)

    print("\nStarting model training...")
    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        epochs=args.epochs,
        callbacks=callbacks
    )

    print("\nTraining completed successfully!")
    print(f"Best weights saved at: {checkpoint_filepath}")
    print(f"Vocabulary saved at   : {vocab_path}")


if __name__ == "__main__":
    main()


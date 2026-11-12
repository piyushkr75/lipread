import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_bilstm_classifier(
    input_shape: tuple = (75, 64, 96, 1),
    num_classes: int = 100,
    cnn_type: str = "3d",
    lstm_units: int = 128,
    dropout_rate: float = 0.4
) -> tf.keras.Model:
    """
    Builds the CNN + BiLSTM architecture for whole-sample word & phrase classification.
    Treats each complete word or phrase recording as one classification sample.

    Args:
        input_shape: (max_frames, height, width, channels)
        num_classes: Number of distinct word and phrase classes
        cnn_type: '3d' for Spatiotemporal Conv3D or '2d' for TimeDistributed Conv2D
        lstm_units: Units per LSTM direction
        dropout_rate: Dropout rate for regularization

    Returns:
        Compiled or uncompiled tf.keras.Model
    """
    inputs = layers.Input(shape=input_shape, name="video_frames")

    if cnn_type.lower() == "3d":
        # 3D Convolutional front-end
        x = layers.Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation="relu", name="conv3d_1")(inputs)
        x = layers.MaxPool3D(pool_size=(1, 2, 2), name="pool3d_1")(x)
        x = layers.BatchNormalization(name="bn_1")(x)

        x = layers.Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation="relu", name="conv3d_2")(x)
        x = layers.MaxPool3D(pool_size=(1, 2, 2), name="pool3d_2")(x)
        x = layers.BatchNormalization(name="bn_2")(x)

        x = layers.Conv3D(96, kernel_size=(3, 3, 3), padding="same", activation="relu", name="conv3d_3")(x)
        x = layers.MaxPool3D(pool_size=(1, 2, 2), name="pool3d_3")(x)
        x = layers.BatchNormalization(name="bn_3")(x)

        # Reshape to (Batch, Time, Features)
        x = layers.TimeDistributed(layers.Flatten(), name="td_flatten")(x)

    else:
        # TimeDistributed 2D Convolutional front-end
        cnn_2d = models.Sequential([
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.MaxPool2D((2, 2)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.MaxPool2D((2, 2)),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.GlobalAveragePooling2D()
        ], name="spatial_encoder")

        x = layers.TimeDistributed(cnn_2d, name="td_cnn2d")(inputs)

    # Temporal feature extraction with Bidirectional LSTMs
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=0.25),
        name="bilstm_1"
    )(x)
    x = layers.Dropout(0.3, name="lstm_dropout_1")(x)

    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=0.25),
        name="bilstm_2"
    )(x)
    x = layers.Dropout(0.3, name="lstm_dropout_2")(x)

    # Temporal Pooling: condense sequential frames into one recording representation
    x = layers.GlobalAveragePooling1D(name="temporal_pooling")(x)

    # Classification Head (Word & Phrase Multiclass Softmax)
    x = layers.Dense(256, activation="relu", name="dense_features")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="LipReading_Classifier")
    return model


def build_ctc_model(
    input_shape: tuple = (75, 64, 96, 1),
    vocab_size: int = 40
) -> tf.keras.Model:
    """
    Alternative sequence-to-sequence model using CTC decode for continuous text output.
    """
    inputs = layers.Input(shape=input_shape, name="video_frames")

    x = layers.Conv3D(32, (3, 3, 3), padding="same", activation="relu")(inputs)
    x = layers.MaxPool3D((1, 2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(64, (3, 3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPool3D((1, 2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(96, (3, 3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPool3D((1, 2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.TimeDistributed(layers.Flatten())(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(vocab_size + 1, activation="softmax", name="ctc_output")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="LipReading_CTC")
    return model


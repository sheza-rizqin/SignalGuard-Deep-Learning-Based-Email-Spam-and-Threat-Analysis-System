from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D


def build_model(
    vocabulary_size: int = 20_000,
    sequence_length: int = 300,
    l2_strength: float = 0.0,
    dense_dropout: float = 0.0,
):
  
    kernel_regularizer = regularizers.l2(l2_strength) if l2_strength > 0 else None
    layers = [
        Embedding(
            input_dim=vocabulary_size,
            output_dim=96,
            mask_zero=True,
        ),
        SpatialDropout1D(0.2),
        LSTM(64, kernel_regularizer=kernel_regularizer),
        Dense(32, activation="relu", kernel_regularizer=kernel_regularizer),
    ]
    if dense_dropout > 0:
        layers.append(Dropout(dense_dropout))
    layers.append(Dense(1, activation="sigmoid"))
    model = Sequential(layers)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

from __future__ import annotations

import json
import zipfile

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.models import load_model


class _CompatibleGlorotUniform(GlorotUniform):

    def __init__(self, seed=None, input_axes=None, output_axes=None, **kwargs):
        super().__init__(seed=seed, **kwargs)


def _load_weights_without_deserializing_model(path):
    """Rebuild this project's simple Sequential graph and restore its weights.

    ``load_weights`` reads the weight archive without deserializing the stale
    initializer configuration that caused the compatibility failure.
    """
    with zipfile.ZipFile(path) as archive:
        payload = json.loads(archive.read("config.json"))
    layer_configs = payload["config"]["layers"]
    layers = []
    sequence_length = 300
    for entry in layer_configs:
        class_name = entry["class_name"]
        config = entry["config"]
        if class_name == "InputLayer":
            shape = config.get("batch_shape") or [None, sequence_length]
            sequence_length = int(shape[-1])
        elif class_name == "Embedding":
            layers.append(Embedding(input_dim=config["input_dim"], output_dim=config["output_dim"]))
        elif class_name == "SpatialDropout1D":
            layers.append(SpatialDropout1D(config["rate"]))
        elif class_name == "LSTM":
            layers.append(
                LSTM(
                    config["units"], activation=config.get("activation", "tanh"),
                    recurrent_activation=config.get("recurrent_activation", "sigmoid"),
                    return_sequences=config.get("return_sequences", False),
                )
            )
        elif class_name == "Dense":
            layers.append(Dense(config["units"], activation=config.get("activation")))
        elif class_name == "Dropout":
            layers.append(Dropout(config["rate"]))
        else:
            raise ValueError(f"Unsupported layer in compatibility loader: {class_name}")
    model = Sequential(layers)
    model(np.zeros((1, sequence_length), dtype="int32"), training=False)
    model.load_weights(path)
    return model


def load_spam_model(path):
    """Load an artifact across supported TensorFlow/Keras patch versions."""
    try:
        return load_model(path, custom_objects={"GlorotUniform": _CompatibleGlorotUniform})
    except TypeError as error:
        if "input_axes" not in str(error):
            raise
        return _load_weights_without_deserializing_model(path)

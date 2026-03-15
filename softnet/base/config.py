"""
Layer configuration builders — translate user-facing params into Keras layers.
"""

from __future__ import annotations

import keras
from keras import layers


def build_dense_block(
    units: int,
    dropout: float = 0.0,
    batch_norm: bool = False,
    activation: str = "relu",
) -> list[keras.Layer]:
    """Return list of layers forming one Dense block."""
    block: list[keras.Layer] = [layers.Dense(units, activation=activation)]

    if batch_norm:
        block.append(layers.BatchNormalization())

    if dropout > 0.0:
        block.append(layers.Dropout(dropout))

    return block


def build_mlp(
    hidden_units: list[int],
    output_units: int,
    output_activation: str,
    dropout: float = 0.0,
    batch_norm: bool = False,
    input_dim: int | None = None,
) -> keras.Sequential:
    """Build a fully connected Sequential model."""

    model = keras.Sequential()

    if input_dim is not None:
        model.add(layers.Input(shape=(input_dim,)))

    for units in hidden_units:
        for layer in build_dense_block(units, dropout=dropout, batch_norm=batch_norm):
            model.add(layer)

    model.add(layers.Dense(output_units, activation=output_activation))
    return model

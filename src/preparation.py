from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True


def train_test_split(
    df: pd.DataFrame, ratio: float, normalise: bool = True
) -> Tuple[pd.DataFrame]:
    """
    Split dataset in train and test. Normalisation as an option.
    """
    train_df = df[: round(df.shape[0] * ratio)]
    test_df = df[round(df.shape[0] * ratio) :]

    if normalise:
        for col in (col for col in train_df.columns if col != "datetime"):
            mu, sigma = train_df[col].mean(), train_df[col].std()
            train_df[col] = (train_df[col] - mu) / sigma
            test_df[col] = (test_df[col] - mu) / sigma

    print("train size:", train_df.shape, "test size:", test_df.shape)

    return train_df, test_df


def make_sequences(
    df: pd.DataFrame, observations: int, offset: int, targets: int
) -> Tuple[Dict[str, List[np.ndarray]]]:
    """
    Make sequences of samples for observations and targets.

    | samples_to_drop | observations | offset | targets | observations | offset | targets | ...
                      <----------sequence 1-----------> <-----------sequence 2----------> ...
    """
    n_sequences = df.shape[0] // (observations + offset + targets)
    samples_to_drop = df.shape[0] % (observations + offset + targets)
    sequences_observations = {key: [] for key in df.columns}
    sequences_targets = {key: [] for key in df.columns}
    for feature in df.columns:
        arr = df[feature].to_numpy()
        arr = arr[samples_to_drop:]
        sequences_observations[feature] = [
            arr[
                sequence_idx
                * (observations + offset + targets) : (sequence_idx + 1)
                * observations
                + sequence_idx * (offset + targets)
            ]
            for sequence_idx in range(n_sequences)
        ]
        sequences_targets[feature] = [
            arr[
                (sequence_idx + 1) * (observations + offset)
                + sequence_idx
                * targets : (sequence_idx + 1)
                * (observations + offset + targets)
            ]
            for sequence_idx in range(n_sequences)
        ]
    print(f"To have uniform sequences, indexes dropped: [0:{samples_to_drop}]")
    print(f"Number of sequences: {n_sequences}")

    return sequences_observations, sequences_targets


def make_batches(
    sequences: Dict[str, List[np.ndarray]], n_sequences_per_batch: int
) -> Dict[str, List[np.ndarray]]:
    """
    Make batches, each comprise at least one sequence.

    | sequence 1 | sequence 2 | sequence 3 | sequence 4 | ...
     <-------------- batch 1 -------------> <------- batch 2 ----

    if n_sequences_per_batch = 3
    """
    return {
        feature: [
            sequences[k : k + n_sequences_per_batch]
            for k in range(0, len(sequences), n_sequences_per_batch)
        ]
        for feature, sequences in sequences.items()
    }

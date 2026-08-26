import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SENSOR_COLS = [
    'setting1', 'setting2', 'setting3', 'cycle_norm',
    's2', 's3', 's4', 's7', 's8', 's9', 's11', 's12',
    's13', 's14', 's15', 's17', 's20', 's21'
]


def load_and_label_data(path):
    """Load engine telemetry and create the 30-cycle critical-risk label."""
    df = pd.read_csv(path).sort_values(['id', 'cycle']).reset_index(drop=True)

    max_cycle = df.groupby('id')['cycle'].transform('max')
    df['RUL'] = max_cycle - df['cycle']
    df['RUL_clipped'] = df['RUL'].clip(upper=125)
    df['label_bc'] = (df['RUL'] <= 30).astype(np.int32)
    df['cycle_norm'] = df['cycle']

    return df


def split_by_engine(df, validation_fraction=0.1, random_state=42):
    """Split complete engine lifecycles so no engine appears in both sets."""
    engine_ids = np.array(sorted(df['id'].unique()))
    rng = np.random.default_rng(random_state)
    rng.shuffle(engine_ids)

    n_validation = max(1, int(round(len(engine_ids) * validation_fraction)))
    validation_ids = set(engine_ids[:n_validation])
    train_ids = set(engine_ids[n_validation:])

    train_df = df[df['id'].isin(train_ids)].copy()
    validation_df = df[df['id'].isin(validation_ids)].copy()

    return train_df, validation_df


def scale_data(df, scaler=None, fit=False):
    """Fit the scaler only on training data, then reuse it for validation/test."""
    df = df.copy()

    if scaler is None:
        scaler = MinMaxScaler()
        fit = True

    if fit:
        df[SENSOR_COLS] = scaler.fit_transform(df[SENSOR_COLS])
    else:
        df[SENSOR_COLS] = scaler.transform(df[SENSOR_COLS])

    return df, scaler


def create_sequences(df, seq_length=50):
    """Create temporal windows independently within each engine lifecycle."""
    sequences = []
    labels = []

    for engine_id in sorted(df['id'].unique()):
        engine_df = df[df['id'] == engine_id].sort_values('cycle')
        values = engine_df[SENSOR_COLS].to_numpy(dtype=np.float32)
        target = engine_df['label_bc'].to_numpy(dtype=np.float32)

        if len(values) <= seq_length:
            continue

        for end in range(seq_length, len(values)):
            sequences.append(values[end - seq_length:end])
            labels.append(target[end])

    if not sequences:
        raise ValueError('No sequences could be generated. Check sequence length and input data.')

    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.float32).reshape(-1, 1)


# Backward-compatible helpers retained for existing scripts.
def gen_sequence(id_df, seq_length, seq_cols):
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]


def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]

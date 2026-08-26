import joblib
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.data_preprocessing import (
    create_sequences,
    load_and_label_data,
    scale_data,
    split_by_engine,
)
from src.model_training import build_lstm_model

TRAIN_PATH = 'data/PM_train.csv'
SCALER_PATH = 'scaler.pkl'
MODEL_PATH = 'model.h5'
SEQUENCE_LENGTH = 50
VALIDATION_FRACTION = 0.10
RANDOM_STATE = 42


def main():
    print('Step 1: Loading and labeling training telemetry...')
    df = load_and_label_data(TRAIN_PATH)

    print('Step 2: Splitting complete engine lifecycles...')
    train_df, validation_df = split_by_engine(
        df,
        validation_fraction=VALIDATION_FRACTION,
        random_state=RANDOM_STATE,
    )

    train_ids = sorted(train_df['id'].unique())
    validation_ids = sorted(validation_df['id'].unique())
    print(f'Train engines: {len(train_ids)}')
    print(f'Validation engines: {len(validation_ids)}')
    print(f'Engine overlap: {set(train_ids) & set(validation_ids)}')

    print('Step 3: Fitting scaler on training engines only...')
    train_df, scaler = scale_data(train_df, fit=True)
    validation_df, _ = scale_data(validation_df, scaler=scaler, fit=False)
    joblib.dump(scaler, SCALER_PATH)

    print('Step 4: Creating temporal sequences...')
    X_train, y_train = create_sequences(train_df, SEQUENCE_LENGTH)
    X_val, y_val = create_sequences(validation_df, SEQUENCE_LENGTH)

    print(f'Train sequences: {X_train.shape}')
    print(f'Validation sequences: {X_val.shape}')

    print('Step 5: Building LSTM...')
    model = build_lstm_model(X_train, y_train)

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, mode='min'),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1),
    ]

    print('Step 6: Training with engine-level validation...')
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=200,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
    )

    print(f'Success: model saved to {MODEL_PATH}')
    print(f'Success: scaler saved to {SCALER_PATH}')


if __name__ == '__main__':
    main()

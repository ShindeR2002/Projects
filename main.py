from pathlib import Path

import joblib
import mlflow
from mlflow.tracking import MlflowClient
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
MLFLOW_EXPERIMENT = 'aircraft-engine-predictive-maintenance'
MLFLOW_DB = 'sqlite:///mlflow.db'
MLFLOW_ARTIFACT_DIR = Path('mlruns_sqlite').resolve()


def configure_mlflow():
    mlflow.set_tracking_uri(MLFLOW_DB)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        experiment_id = client.create_experiment(
            MLFLOW_EXPERIMENT,
            artifact_location=MLFLOW_ARTIFACT_DIR.as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)


def main():
    configure_mlflow()

    with mlflow.start_run() as run:
        mlflow.log_params({
            'sequence_length': SEQUENCE_LENGTH,
            'validation_fraction': VALIDATION_FRACTION,
            'random_state': RANDOM_STATE,
            'epochs': 50,
            'batch_size': 200,
        })

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
        overlap = set(train_ids) & set(validation_ids)
        print(f'Engine overlap: {overlap}')
        if overlap:
            raise ValueError(f'Engine leakage detected: {overlap}')

        mlflow.log_metrics({
            'train_engines': len(train_ids),
            'validation_engines': len(validation_ids),
        })

        print('Step 3: Fitting scaler on training engines only...')
        train_df, scaler = scale_data(train_df, fit=True)
        validation_df, _ = scale_data(validation_df, scaler=scaler, fit=False)
        joblib.dump(scaler, SCALER_PATH)

        print('Step 4: Creating temporal sequences...')
        X_train, y_train = create_sequences(train_df, SEQUENCE_LENGTH)
        X_val, y_val = create_sequences(validation_df, SEQUENCE_LENGTH)

        print(f'Train sequences: {X_train.shape}')
        print(f'Validation sequences: {X_val.shape}')
        mlflow.log_metrics({
            'train_sequences': X_train.shape[0],
            'validation_sequences': X_val.shape[0],
        })

        print('Step 5: Building LSTM...')
        model = build_lstm_model(X_train, y_train)

        callbacks = [
            ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, mode='min'),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1),
        ]

        print('Step 6: Training with engine-level validation...')
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=200,
            shuffle=True,
            verbose=1,
            callbacks=callbacks,
        )

        best_epoch = int(min(range(len(history.history['val_loss'])), key=lambda i: history.history['val_loss'][i])) + 1
        best_val_loss = float(min(history.history['val_loss']))
        best_val_accuracy = float(history.history['val_accuracy'][best_epoch - 1])

        mlflow.log_metrics({
            'best_val_loss': best_val_loss,
            'best_val_accuracy': best_val_accuracy,
            'best_epoch': best_epoch,
        })
        mlflow.log_artifact(MODEL_PATH)
        mlflow.log_artifact(SCALER_PATH)

        print(f'Best epoch: {best_epoch}')
        print(f'Best validation loss: {best_val_loss:.6f}')
        print(f'Best validation accuracy: {best_val_accuracy:.6f}')
        print(f'MLflow tracking URI: {mlflow.get_tracking_uri()}')
        print(f'MLflow run ID: {run.info.run_id}')
        print(f'Success: model saved to {MODEL_PATH}')
        print(f'Success: scaler saved to {SCALER_PATH}')


if __name__ == '__main__':
    main()

import keras
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from utils_rnn import *
import numpy as np
import pickle




# === Funzione di training ===
def train_rnn(X_train, y_train, X_val, y_val,
              epochs, param_grid, callback_list, rnn_type="lstm"):
    all_history = []
    best_score = 0
    best_run = None

    # Pre-computa lunghezze costanti per tutto il dataset
    batch_train = X_train.shape[0]
    batch_val   = X_val.shape[0]
    input_length_train = np.ones((batch_train, 1), dtype=np.int32) * X_train.shape[1]
    label_length_train = np.ones((batch_train, 1), dtype=np.int32) * y_train.shape[1]
    input_length_val   = np.ones((batch_val, 1), dtype=np.int32) * X_val.shape[1]
    label_length_val   = np.ones((batch_val, 1), dtype=np.int32) * y_val.shape[1]

    for dropout_rate in param_grid['dropout_rate']:
        for n_units in param_grid['n_units']:
            for n_layers in param_grid['n_layers']:
                for batch_size in param_grid['batch_size']:
                    for learning_rate in param_grid['learning_rate']:
                        print(f"dropout={dropout_rate}, units={n_units}, layers={n_layers}, bs={batch_size}, lr={learning_rate}")
                        # Costruisci il modello CTC
                        builder = lstm_rnn if rnn_type=="lstm" else gru_rnn
                        model = build_ctc_model(
                            rnn_builder=builder,
                            input_dim=X_train.shape[2],
                            output_dim=y_train.max()+1,
                            dropout=dropout_rate,
                            n_layers=n_layers,
                            n_units=n_units
                        )

                        model.summary()

                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            loss=ctc_loss_fn,
                            loss_weights=[1.0, 0.0],  # solo la loss 'ctc', ignora logits
                            metrics=["accuracy"]
                        )

                        # Dummy y per .fit()
                        dummy_y_train = np.zeros((batch_train,))
                        dummy_y_val = np.zeros((batch_val,))

                        history = model.fit(
                            x={
                                'input_spectrogram': X_train,
                                'labels': y_train,
                                'input_length': input_length_train,
                                'label_length': label_length_train
                            },
                            y=dummy_y_train,
                            validation_data=(
                                {
                                    'input_spectrogram': X_val,
                                    'labels': y_val,
                                    'input_length': input_length_val,
                                    'label_length': label_length_val
                                },
                                dummy_y_val
                            ),
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=callback_list,
                            verbose=1
                        )

                        # Miglior accuracy di validazione per la configurazione
                        max_val_acc = max(history.history['val_accuracy'])

                        run = {
                            'dropout_rate': dropout_rate,
                            'n_units': n_units,
                            'n_layers': n_layers,
                            'batch_size': batch_size,
                            'learning_rate': learning_rate,
                            'max_val_acc': max_val_acc,
                            'history': history.history,
                            'model': model
                        }
                        all_history.append(run)

                        if max_val_acc > best_score:
                            best_score = max_val_acc
                            best_run = run

    return all_history, best_run, best_score


if __name__ == "__main__":
    # Apertura dataset preprocessing
    dataset_train = np.load("dataset_split/asr_train.npz")
    X_train = dataset_train['X']
    y_train = dataset_train['y']
    dataset_val = np.load("dataset_split/asr_val.npz")
    X_val = dataset_val['X']
    y_val = dataset_val['y']
    dataset_test = np.load("dataset_split/asr_test.npz")
    X_test = dataset_test['X']
    y_test = dataset_test['y']

    print(f"Training set shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Validation labels shape: {y_val.shape}")

    with open("dataset_split/char2idx.pkl", "rb") as f:
        char2idx = pickle.load(f)

    with open("dataset_split/idx2char.pkl", "rb") as f:
        idx2chart = pickle.load(f)

    vocab_size = len(char2idx) + 1  # +1 per padding
    # fft_lenght è un parametro usato nella fase di preprocessing per estrarre lo spectrogram
    fft_length = 384  # definito in preprocessing
    spectrogram_features = fft_length // 2 + 1

    param_grid = {
        'dropout_rate': [0.0, 0.2, 0.5],
        'n_units': [64, 128],
        'n_layers': [1, 2, 3],
        'batch_size': [32, 64],
        'learning_rate': [0.001, 0.01]
    }
    epochs = 2

    # Esegui training LSTM
    lstm_callbacks_list = [
        keras.callbacks.ModelCheckpoint('lstm_tensorboard/checkpoint_model.keras', monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5),
        keras.callbacks.TensorBoard(log_dir="/lstm_tensorboard")
    ]
    lstm_all_history, lstm_best_run, lstm_best_score = train_rnn(
        X_train, y_train, X_val, y_val,
        epochs, param_grid, lstm_callbacks_list, rnn_type="lstm"
    )
    lstm_model = lstm_best_run['model']
    plot_model(lstm_model, to_file="lstm_tensorboard/plot_model.png", show_shapes=True)


    # Esegui training GRU
    gru_callbacks_list = [
        keras.callbacks.ModelCheckpoint('gru_tensorboard/checkpoint_model.keras', monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5),
        keras.callbacks.TensorBoard(log_dir="/gru_tensorboard")
    ]
    gru_hist, gru_best, gru_score = train_rnn(
        X_train, y_train, X_val, y_val,
        epochs, param_grid, gru_callbacks_list, rnn_type="gru"
    )
    gru_model = gru_best['model']
    plot_model(gru_model, to_file="gru_tensorboard/plot_model.png", show_shapes=True)

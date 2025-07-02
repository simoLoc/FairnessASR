import keras
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from utils_rnn import *
import numpy as np
import pickle
import copy


# === Funzione di training ===
def train_rnn(X_train, y_train, X_val, y_val,
              epochs, param_grid, callback_list, rnn_type="lstm"):
    all_history = []
    best_score = 0
    best_run = None
    best_model = None

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
                        print(f"dropout={dropout_rate}, units={n_units}, layers={n_layers}, batch_size={batch_size}, learning_rate={learning_rate}")
                        # Costruisci il modello CTC
                        builder = lstm_rnn if rnn_type=="lstm" else gru_rnn
                        model = build_ctc_model(
                            rnn_builder=builder,
                            input_dim=X_train.shape[2],
                            output_dim=vocab_size,
                            dropout=dropout_rate,
                            n_layers=n_layers,
                            n_units=n_units
                        )

                        model.summary()

                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            loss={'ctc': ctc_loss_fn, 'logits': None}
                        )

                        # Dummy y per .fit()
                        dummy_y_train = np.zeros((batch_train,))
                        dummy_y_val = np.zeros((batch_val,))

                        # dummy_y_train = {'ctc': np.zeros((batch_train,)), 'logits': None}
                        # dummy_y_val = {'ctc': np.zeros((batch_val,)), 'logits': None}

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
                            'history': history.history
                        }
                        all_history.append(run)

                        if max_val_acc > best_score:
                            best_score = max_val_acc
                            best_run = copy.copy(run)
                            best_model = copy.copy(model)

    return all_history, best_run, best_score, best_model


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

    # Dataset di training
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # Dataset di validazione
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    # Dataset di test
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))


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

    print("=== Esecuzione LSTM ===")
    # Esegui training LSTM
    lstm_callbacks_list = [
        keras.callbacks.ModelCheckpoint('lstm_tensorboard/checkpoint_model.keras', monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5),
        keras.callbacks.TensorBoard(log_dir="/lstm_tensorboard")
    ]
    lstm_all_history, lstm_best_run, lstm_best_score, lstm_best_model = train_rnn(
        X_train, y_train, X_val, y_val,
        epochs, param_grid, lstm_callbacks_list, rnn_type="lstm"
    )
    lstm_best_model.save("lstm_tensorboard/lstm_best_model.keras")
    plot_model(lstm_best_model, to_file="lstm_tensorboard/plot_model.png", show_shapes=True)
    plot_accuracy(lstm_best_run['history'], model="LSTM", dir = "lstm_tensorboard")
    plot_loss(lstm_best_run['history'], model="LSTM", dir = "lstm_tensorboard")
    save_best_run(lstm_best_run, dir="lstm_tensorboard")

    print("=== Esecuzione GRU ===")
    # Esegui training GRU
    gru_callbacks_list = [
        keras.callbacks.ModelCheckpoint('gru_tensorboard/checkpoint_model.keras', monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5),
        keras.callbacks.TensorBoard(log_dir="/gru_tensorboard")
    ]
    gru_hist, gru_best_run, gru_score, gru_best_model = train_rnn(
        X_train, y_train, X_val, y_val,
        epochs, param_grid, gru_callbacks_list, rnn_type="gru"
    )
    gru_best_model.save("gru_tensorboard/gru_best_model.keras")
    plot_model(gru_best_model, to_file="gru_tensorboard/plot_model.png", show_shapes=True)
    plot_accuracy(gru_best_run['history'], model="GRU", dir = "gru_tensorboard")
    plot_loss(gru_best_run['history'], model="GRU", dir = "gru_tensorboard")
    save_best_run(gru_best_run, dir="gru_tensorboard")

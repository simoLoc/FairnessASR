import pickle
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from utils_rnn import *

def train_rnn(X_train, y_train, X_val, y_val, epochs, param_grid, callback_list, model=""):
    all_history = []
    best_score = 0
    best_run = None

    for dropout_rate in param_grid['dropout_rate']:
        for n_units in param_grid['n_units']:
            for n_layers in param_grid['n_layers']:
                for batch_size in param_grid['batch_size']:
                    for learning_rate in param_grid['learning_rate']:

                        # definizione modello
                        if model == "lstm":
                            model = lstm_rnn(input_dim=mfcc_features, output_dim=vocab_size, 
                                            dropout=dropout_rate, n_layers=n_layers, n_units=n_units)
                        else:
                            model = gru_rnn(input_dim=mfcc_features, output_dim=vocab_size, 
                                            dropout=dropout_rate, n_layers=n_layers, n_units=n_units)
                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                            loss=ctc_loss,
                            metrics=["accuracy"]
                        )

                        # Train
                        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=callback_list
                        )
                        
                        # Miglior accuracy di validazione per la configurazione
                        max_val_acc = max(history.history['val_accuracy'])

                        # Salvatagio dell'history della configurazione
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

    # Apertura dizionari per il decoding
    with open("dataset_split/char2idx.pkl", "rb") as f:
        char2idx = pickle.load(f)

    with open("dataset_split/idx2char.pkl", "rb") as f:
        idx2chart = pickle.load(f)

    vocab_size = len(char2idx) + 1  # +1 per padding
    mfcc_features = 13  # numero di MFCC features per timestep (definito in preprocessing)

    param_grid = {
        'dropout_rate': [0.0, 0.2, 0.5],
        'n_units': [64, 128],
        'n_layers': [1, 2, 3],
        'batch_size': [64, 128],
        'learning_rate': [0.001, 0.01]
    }

    epochs = 2

    # LSTM 
    lstm_callbacks_list = [
        keras.callbacks.ModelCheckpoint('lstm_tensorboard/checkpoint_model.keras', monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5),
        keras.callbacks.TensorBoard(log_dir="/lstm_tensorboard")
    ]
    lstm_all_history, lstm_best_run, lstm_best_score = train_rnn(X_train, y_train, X_val, y_val, epochs, param_grid, lstm_callbacks_list, model="lstm")
    lstm_model = lstm_best_run['model']
    plot_model(lstm_model, to_file="lstm_tensorboard/plot_model.png", show_shapes=True)

    # GRU
    gru_callbacks_list = [
        keras.callbacks.ModelCheckpoint('gru_tensorboard/checkpoint_model.keras', monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5),
        keras.callbacks.TensorBoard(log_dir="/gru_tensorboard")
    ]
    gru_all_history, gru_best_run, gru_best_score = train_rnn(X_train, y_train, X_val, y_val, epochs, param_grid, gru_callbacks_list, model="gru")
    gru_model = gru_best_run['model']
    plot_model(gru_model, to_file="gru_tensorboard/plot_model.png", show_shapes=True)
    

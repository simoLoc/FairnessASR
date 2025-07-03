import keras
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from utils_rnn import *
import numpy as np
import pickle
import copy
from jiwer import wer, cer
from asr_models_hf import *

def evaluate_wer_numpy(model, X, y, idx2char):
    """
    Restituisce la WER media su (X, y).
    """
    batch = X.shape[0]
    # ricava input_length e label_length costanti
    input_len = np.full((batch,), X.shape[1], dtype=np.int32)
    
    # Predict logits
    out = model.predict({
        'input_spectrogram': X,
        'labels':            y,
        'input_length':      input_len[:, None],
        'label_length':      np.full((batch,1), y.shape[1], dtype=np.int32)
    })
    logits = out['logits']  # shape (batch, time, vocab)
    
    # Decodifica CTC
    decoded, _ = tf.keras.backend.ctc_decode(
        logits, 
        input_length=input_len
    )
    sparse = decoded[0].numpy()  # array shape (batch, max_decoded_len)
    
    # da sequenze di indici a stringhe
    preds = []
    for seq in sparse:
        chars = [ idx2char[int(c)] for c in seq if c >= 0 ]
        preds.append(''.join(chars))
    trues = []
    for seq in y:
        chars = [ idx2char[int(c)] for c in seq if c != 0 ]
        trues.append(''.join(chars))
    
    return wer(trues, preds)


def evaluate_by_groups(model, X, y, groups_dict, idx2char):
    """
    groups_dict: dict nome_gruppo -> array booleano (stessa lunghezza di X e y)
    Restituisce dict nome_gruppo -> WER (o None se il gruppo è vuoto).
    """
    results = {}
    for category, subgroups in groups_dict.items():
        category_results = {}
        for subgroup, mask in subgroups.items():
            X_sub = X[mask]
            y_sub = y[mask]
            if len(X_sub) == 0:
                category_results[subgroup] = None
            else:
                wer_score = evaluate_wer_numpy(model, X_sub, y_sub, idx2char)
                category_results[subgroup] = wer_score
        results[category] = category_results

    return results



def decode_predictions(logits_batch, input_lengths, idx2char):
    decoded_texts = []
    for i in range(logits_batch.shape[0]):
        logit = logits_batch[i, :input_lengths[i][0], :]
        decoded, _ = tf.keras.backend.ctc_decode(logit[None, ...], input_lengths[i])
        pred_indices = decoded[0][0].numpy()
        pred_text = ''.join([idx2char.get(int(i), '') for i in pred_indices])
        decoded_texts.append(pred_text)
    return decoded_texts


class WERCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val, input_lengths, label_lengths, idx2char, patience=5, save_path="best_model.keras"):
        super(WERCallback, self).__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.input_lengths = input_lengths
        self.label_lengths = label_lengths
        self.idx2char = idx2char
        self.patience = patience
        self.save_path = save_path
        self.best_wer = float("inf")
        self.wait = 0


    def on_epoch_end(self, epoch, logs=None):
        logits = self.model.predict({
            "input_spectrogram": self.X_val,
            "labels": self.y_val,
            "input_length": self.input_lengths,
            "label_length": self.label_lengths
        })["logits"]

        preds = decode_predictions(logits, self.input_lengths, self.idx2char)

        gt_texts = []
        for i in range(self.y_val.shape[0]):
            label_seq = self.y_val[i][:self.label_lengths[i][0]]
            gt_text = ''.join([self.idx2char.get(int(idx), '') for idx in label_seq])
            gt_texts.append(gt_text)

        wer_score = wer(gt_texts, preds)
        logs["val_wer"] = wer_score

        print(f"\n[Epoch {epoch+1}] WER = {wer_score:.4f}")

        if wer_score < self.best_wer:
            self.best_wer = wer_score
            self.wait = 0
            self.model.save(self.save_path)
            print(f"[Epoch {epoch+1}] Nuovo best model salvato con WER = {wer_score:.4f}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"[Epoch {epoch+1}] Early stopping per WER (no miglioramenti in {self.patience} epoche)")
                self.model.stop_training = True

# === Funzione di training ===
def train_rnn(X_train, y_train, X_val, y_val,
              epochs, param_grid, callback_list, rnn_type="lstm"):
    all_history = []
    best_score = float("inf")
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
                        # max_val_acc = max(history.history['val_accuracy'])

                        # Salviamo la WER minima
                        val_wer_min = min(history.history.get("val_wer", [float("inf")]))


                        run = {
                            'dropout_rate': dropout_rate,
                            'n_units': n_units,
                            'n_layers': n_layers,
                            'batch_size': batch_size,
                            'learning_rate': learning_rate,
                            'val_wer_min': val_wer_min,
                            'history': history.history
                        }
                        all_history.append(run)

                        if val_wer_min < best_score:
                            best_score = val_wer_min 
                            best_run = copy.copy(run)
                            best_model = copy.copy(model)

    return all_history, best_run, best_score, best_model

def main_rnn()
    # Apertura dataset preprocessing
    dataset_train = np.load("dataset_split/asr_train.npz")
    X_train = dataset_train['X'][384]  # Prendi solo le prime 384 features
    y_train = dataset_train['y'][384]
    dataset_val = np.load("dataset_split/asr_val.npz")
    X_val = dataset_val['X']
    y_val = dataset_val['y']
    dataset_test = np.load("dataset_split/asr_test.npz")
    X_test = dataset_test['X']
    y_test = dataset_test['y']


    # Pre-computa lunghezze costanti per tutto il dataset
    batch_train = X_train.shape[0]
    batch_val   = X_val.shape[0]
    input_length_train = np.ones((batch_train, 1), dtype=np.int32) * X_train.shape[1]
    label_length_train = np.ones((batch_train, 1), dtype=np.int32) * y_train.shape[1]
    input_length_val   = np.ones((batch_val, 1), dtype=np.int32) * X_val.shape[1]
    label_length_val   = np.ones((batch_val, 1), dtype=np.int32) * y_val.shape[1]

    print(f"Training set shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Validation labels shape: {y_val.shape}")

    men_test = dataset_test['man']
    women_test = dataset_test['woman']
    aave_test = dataset_test['aave']
    sae_test = dataset_test['sae']
    spanglish_test = dataset_test['spanglish']
    chicano_test = dataset_test['chicano_english']
    others_test = dataset_test['other_dialect_accent']

    group_dict = {
        "Gender": {
            "Men": men_test,
            "Women": women_test
        },
        "Dialect": {
            "AAVE": aave_test,
            "SAE": sae_test,
            "Spanglish": spanglish_test,
            "Chicano English": chicano_test,
            "Other Dialects": others_test
        }
    }

    with open("dataset_split/char2idx.pkl", "rb") as f:
        char2idx = pickle.load(f)

    with open("dataset_split/idx2char.pkl", "rb") as f:
        idx2char = pickle.load(f)

    vocab_size = len(char2idx) + 1  # +1 per padding
    # fft_lenght è un parametro usato nella fase di preprocessing per estrarre lo spectrogram
    fft_length = 384  # definito in preprocessing
    spectrogram_features = fft_length // 2 + 1

    # param_grid = {
    #     'dropout_rate': [0.0, 0.2, 0.5],
    #     'n_units': [64, 128],
    #     'n_layers': [1, 2, 3],
    #     'batch_size': [32, 64],
    #     'learning_rate': [0.001, 0.01]
    # }

    param_grid = {
        'dropout_rate': [0.2],
        'n_units': [64],
        'n_layers': [1],
        'batch_size': [32],
        'learning_rate': [0.001]
    }

    epochs = 1

    print("=== Esecuzione LSTM ===")
    # Esegui training LSTM
    lstm_callbacks_list = [
        WERCallback(X_val, y_val, input_length_val, label_length_val, idx2char,
                                        patience=5, save_path=f"lstm_tensorboard/best_model.keras"),
        keras.callbacks.ModelCheckpoint('lstm_tensorboard/checkpoint_model.keras', monitor='val_wer', mode='min', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_wer', mode='min', patience=5),
        keras.callbacks.TensorBoard(log_dir="/lstm_tensorboard")
    ]
    lstm_all_history, lstm_best_run, lstm_best_score, lstm_best_model = train_rnn(
        X_train, y_train, X_val, y_val,
        epochs, param_grid, lstm_callbacks_list, rnn_type="lstm"
    )
    lstm_best_model.save("lstm_tensorboard/lstm_best_model.keras")
    plot_model(lstm_best_model, to_file="lstm_tensorboard/plot_model.png", show_shapes=True)
    # plot_accuracy(lstm_best_run['history'], model="LSTM", dir = "lstm_tensorboard")
    plot_loss(lstm_best_run['history'], model="LSTM", dir = "lstm_tensorboard")
    save_best_run(lstm_best_run, dir="lstm_tensorboard")

    print("=== Esecuzione GRU ===")
    # Esegui training GRU
    gru_callbacks_list = [
        WERCallback(X_val, y_val, input_length_val, label_length_val, idx2char,
                                        patience=5, save_path=f"gru_tensorboard/best_model.keras"),
        keras.callbacks.ModelCheckpoint('gru_tensorboard/checkpoint_model.keras', monitor='val_wer', mode='min', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_wer', mode='min', patience=5),
        keras.callbacks.TensorBoard(log_dir="/gru_tensorboard")
    ]
    gru_hist, gru_best_run, gru_score, gru_best_model = train_rnn(
        X_train, y_train, X_val, y_val,
        epochs, param_grid, gru_callbacks_list, rnn_type="gru"
    )
    gru_best_model.save("gru_tensorboard/gru_best_model.keras")
    plot_model(gru_best_model, to_file="gru_tensorboard/plot_model.png", show_shapes=True)
    # plot_accuracy(gru_best_run['history'], model="GRU", dir = "gru_tensorboard")
    plot_loss(gru_best_run['history'], model="GRU", dir = "gru_tensorboard")
    save_best_run(gru_best_run, dir="gru_tensorboard")

    # === Evaluation LSTM ===
    result_lstm = lstm_best_model.evaluate(
        x={
            'input_spectrogram': X_test,
            'labels': y_test,
            'input_length': np.ones((X_test.shape[0], 1), dtype=np.int32) * X_test.shape[1],
            'label_length': np.ones((X_test.shape[0], 1), dtype=np.int32) * y_test.shape[1]
        },
        y=np.zeros((X_test.shape[0],)),
        batch_size = lstm_best_run['batch_size']
    )

    print(f"Risultati loss LSTM: {result_lstm}")

    # === Evaluation GRU ===
    result_gru = gru_best_model.evaluate(
        x={
            'input_spectrogram': X_test,
            'labels': y_test,
            'input_length': np.ones((X_test.shape[0], 1), dtype=np.int32) * X_test.shape[1],
            'label_length': np.ones((X_test.shape[0], 1), dtype=np.int32) * y_test.shape[1]
        },
        y=np.zeros((X_test.shape[0],)), 
        batch_size = gru_best_run['batch_size']
    )

    print(f"Risultati loss GRU: {result_gru}")

    # test_pred_lstm = lstm_best_model.predict(
    #     x={
    #         'input_spectrogram': X_test,
    #         'labels': y_test,
    #         'input_length': np.ones((X_test.shape[0], 1), dtype=np.int32) * X_test.shape[1],
    #         'label_length': np.ones((X_test.shape[0], 1), dtype=np.int32) * y_test.shape[1]
    #     })

    results_wer_lstm = evaluate_by_group(lstm_best_model, X_test, y_test, group_dict, idx2char)

    for category, subgroups in results_wer_lstm.items():
        print(f"\n Categoria: {category}")
    for name, score in subgroups.items():
        if score is None:
            print(f" {name}: nessun campione")
        else:
            print(f" {name}: WER = {score:.4f}")

    results_wer_gru = evaluate_by_group(gru_best_model, X_test, y_test, group_dict, idx2char)

    for category, subgroups in results_wer_gru.items():
        print(f"\n Categoria: {category}")
    for name, score in subgroups.items():
        if score is None:
            print(f" {name}: nessun campione")
        else:
            print(f" {name}: WER = {score:.4f}")



if __name__ == "__main__":
    main_rnn()
    




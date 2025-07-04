import keras
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from utils_rnn import *
import numpy as np
import pickle
import copy
from asr_models_hf import *

# def evaluate_wer_numpy(model, X, y, idx2char):
#     """
#     Restituisce la WER media su (X, y).
#     """
#     batch = X.shape[0]
#     # ricava input_length e label_length costanti
#     input_len = np.full((batch,), X.shape[1], dtype=np.int32)
    
#     # Predict logits
#     out = model.predict({
#         'input_spectrogram': X,
#         'labels':            y,
#         'input_length':      input_len[:, None],
#         'label_length':      np.full((batch,1), y.shape[1], dtype=np.int32)
#     })
#     logits = out['logits']  # shape (batch, time, vocab)
    
#     # Decodifica CTC
#     decoded, _ = tf.keras.backend.ctc_decode(
#         logits, 
#         input_length=input_len
#     )
#     sparse = decoded[0].numpy()  # array shape (batch, max_decoded_len)
    
#     # da sequenze di indici a stringhe
#     preds = []
#     for seq in sparse:
#         chars = [ idx2char[int(c)] for c in seq if c >= 0 ]
#         preds.append(''.join(chars))
#     trues = []
#     for seq in y:
#         chars = [ idx2char[int(c)] for c in seq if c != 0 ]
#         trues.append(''.join(chars))
    
#     return wer(trues, preds)


# def evaluate_by_groups(model, X, y, groups_dict, idx2char):
#     """
#     groups_dict: dict nome_gruppo -> array booleano (stessa lunghezza di X e y)
#     Restituisce dict nome_gruppo -> WER (o None se il gruppo è vuoto).
#     """
#     results = {}
#     for category, subgroups in groups_dict.items():
#         category_results = {}
#         for subgroup, mask in subgroups.items():
#             X_sub = X[mask]
#             y_sub = y[mask]
#             if len(X_sub) == 0:
#                 category_results[subgroup] = None
#             else:
#                 wer_score = evaluate_wer_numpy(model, X_sub, y_sub, idx2char)
#                 category_results[subgroup] = wer_score
#         results[category] = category_results

#     return results




# === Funzione di training ===
def train_rnn(X_train, y_train, X_val, y_val, epochs, param_grid, 
                callback_list, vocab_size, idx2char, rnn_type="lstm"):
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
                            loss=ctc_loss_fn,
                            metrics=[WERMetric(idx2char), CERMetric(idx2char)]
                        )

                        # Combina y_true = [labels | label_length | input_length]
                        max_label_len = y_train.shape[1]
                        y_train_all = np.concatenate([
                            y_train,
                            label_length_train.reshape(-1, 1),
                            input_length_train.reshape(-1, 1)
                        ], axis=1)

                        y_val_all = np.concatenate([
                            y_val,
                            label_length_val.reshape(-1, 1),
                            input_length_val.reshape(-1, 1)
                        ], axis=1)

                        history = model.fit(
                            x=X_train,
                            y=y_train_all,
                            validation_data=(X_val, y_val_all),
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=callback_list
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


def get_evaluate_results(group_dict, X_test, y_test, best_model, best_run):
    
    results_evaluation = []

    for category, subgroups in group_dict.items():
        
        for subgroup_name, mask in subgroups.items():
            # Estrai i sample per la sottocategoria

            X_sub = X_test[mask]
            y_sub = y_test[mask]
            input_length_sub = np.ones((X_sub.shape[0], 1), dtype=np.int32) * X_sub.shape[1]
            label_length_sub = np.ones((X_sub.shape[0], 1), dtype=np.int32) * y_sub.shape[1]

            # Costruisci y_test_all per questa sottocategoria
            y_sub_all = np.concatenate([
                y_sub,
                label_length_sub,
                input_length_sub
            ], axis=1)

            # Valutazione
            loss, wer, cer = best_model.evaluate(
                x=X_sub,
                y=y_sub_all,
                batch_size=best_run['batch_size']
            )

            print(f"{subgroup_name}: Loss={loss:.4f}, WER={wer:.4f}, CER={cer:.4f}")
            results_evaluation.append({
                "Category": category,
                "Group": subgroup_name,
                "Loss": loss,
                "WER": wer,
                "CER": cer
            })

    return results_evaluation



def main_rnn():
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


    # Pre-computa lunghezze costanti per tutto il dataset
    batch_train = X_train.shape[0]
    batch_val   = X_val.shape[0]
    batch_test  = X_test.shape[0]
    input_length_train = np.ones((batch_train, 1), dtype=np.int32) * X_train.shape[1]
    label_length_train = np.ones((batch_train, 1), dtype=np.int32) * y_train.shape[1]
    input_length_val   = np.ones((batch_val, 1), dtype=np.int32) * X_val.shape[1]
    label_length_val   = np.ones((batch_val, 1), dtype=np.int32) * y_val.shape[1]
    input_length_test  = np.ones((batch_test, 1), dtype=np.int32) * X_test.shape[1]
    label_length_test  = np.ones((batch_test, 1), dtype=np.int32) * y_test.shape[1]

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

    # Forziamo tali dataset con astype(bool) in quanto sono booleani, ma contengono valori float
    # (0.0 e 1.0), quindi per utilizzarli come mask dobbiamo forzare il tipo
    group_dict = {
        "Gender": {
            "Men": men_test.astype(bool),
            "Women": women_test.astype(bool)
        },
        "Dialect": {
            "AAVE": aave_test.astype(bool),
            "SAE": sae_test.astype(bool),
            "Spanglish": spanglish_test.astype(bool),
            "Chicano English": chicano_test.astype(bool),
            "Other Dialects": others_test.astype(bool)
        },
        "Gender and Dialect": {
            "AAVE Men": np.logical_and(men_test, aave_test).astype(bool),
            "AAVE Women": np.logical_and(women_test, aave_test).astype(bool),
            "SAE Men": np.logical_and(men_test, sae_test).astype(bool),
            "SAE Women": np.logical_and(women_test, sae_test).astype(bool),
            "Spanglish Men": np.logical_and(men_test, spanglish_test).astype(bool),
            "Spanglish Women": np.logical_and(women_test, spanglish_test).astype(bool),
            "Chicano English Men": np.logical_and(men_test, chicano_test).astype(bool),
            "Chicano English Women": np.logical_and(women_test, chicano_test).astype(bool),
            "Other Dialects Men": np.logical_and(men_test, others_test).astype(bool),
            "Other Dialects Women": np.logical_and(women_test, others_test).astype(bool)
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

    param_grid = {
        'dropout_rate': [0.2, 0.5],
        'n_units': [64, 128],
        'n_layers': [1, 2, 3],
        'batch_size': [32, 64],
        'learning_rate': [0.001, 0.01]
    }

    epochs = 30
    

    print("=== Train LSTM ===")
    # Esegui training LSTM
    lstm_callbacks_list = [
        keras.callbacks.ModelCheckpoint('lstm_tensorboard/checkpoint_model.keras', monitor='val_wer', mode='min', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_wer', mode='min', patience=5),
        keras.callbacks.TensorBoard(log_dir="/lstm_tensorboard")
    ]
    lstm_all_history, lstm_best_run, lstm_best_score, lstm_best_model = train_rnn(
        X_train, y_train, X_val, y_val,
        epochs, param_grid, lstm_callbacks_list, vocab_size, idx2char, rnn_type="lstm"
    )
    lstm_best_model.save("lstm_tensorboard/lstm_best_model.keras")
    # plot_model(lstm_best_model, to_file="lstm_tensorboard/plot_model.png", show_shapes=True)
    plot_wer(lstm_best_run['history'], model="LSTM", dir = "lstm_tensorboard")
    plot_cer(lstm_best_run['history'], model="LSTM", dir = "lstm_tensorboard")
    plot_loss(lstm_best_run['history'], model="LSTM", dir = "lstm_tensorboard")
    save_best_run(lstm_best_run, dir="lstm_tensorboard")

    y_test_all = np.concatenate([
        y_test,
        label_length_test.reshape(-1, 1),
        input_length_test.reshape(-1, 1)
    ], axis=1)

    # === Evaluation LSTM ===
    lstm_results_evaluation = get_evaluate_results(group_dict, X_test, y_test, lstm_best_model, lstm_best_run)

    # Evaluation overall
    print("=== Evaluation LSTM ===")
    lstm_loss, lstm_wer, lstm_cer = lstm_best_model.evaluate(
        x=X_test,
        y=y_test_all, 
        batch_size = lstm_best_run['batch_size']
    )

    lstm_results_evaluation.append({
        "Category": "Overall",
        "Loss": lstm_loss,
        "WER": lstm_wer,
        "CER": lstm_cer
    })


    save_evaluation_results(lstm_results_evaluation, "lstm")


    print("=== Train GRU ===")
    # Esegui training GRU
    gru_callbacks_list = [
        keras.callbacks.ModelCheckpoint('gru_tensorboard/checkpoint_model.keras', monitor='val_wer', mode='min', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_wer', mode='min', patience=5),
        keras.callbacks.TensorBoard(log_dir="/gru_tensorboard")
    ]
    gru_hist, gru_best_run, gru_score, gru_best_model = train_rnn(
        X_train, y_train, X_val, y_val,
        epochs, param_grid, gru_callbacks_list, vocab_size, idx2char, rnn_type="gru"
    )
    gru_best_model.save("gru_tensorboard/gru_best_model.keras")
    # plot_model(gru_best_model, to_file="gru_tensorboard/plot_model.png", show_shapes=True)
    plot_wer(gru_best_run['history'], model="GRU", dir = "gru_tensorboard")
    plot_cer(gru_best_run['history'], model="GRU", dir = "gru_tensorboard")
    plot_loss(gru_best_run['history'], model="GRU", dir = "gru_tensorboard")
    save_best_run(gru_best_run, dir="gru_tensorboard")

    # === Evaluation GRU ===
    gru_results_evaluation = get_evaluate_results(group_dict, X_test, y_test, gru_best_model, gru_best_run)

    # Evaluation overall
    print("=== Evaluation LSTM ===")
    gru_loss, gru_wer, gru_cer = lstm_best_model.evaluate(
        x=X_test,
        y=y_test_all, 
        batch_size = gru_best_run['batch_size']
    )

    gru_results_evaluation.append({
        "Category": "Overall",
        "Loss": gru_loss,
        "WER": gru_wer,
        "CER": gru_cer
    })
    
    save_evaluation_results(gru_results_evaluation, "gru")


if __name__ == "__main__":
    main_rnn()
    




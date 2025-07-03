import keras
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
import json

# === testo -> indici caratteri --> ENCODING ===
def text_to_sequence(text, char2idx):
    """
        Funzione che da un testo estrai i caratteri e gli associa il rispettivo id.
        Funzione di encoding.
    """
    text = text.lower()
    return [char2idx[c] for c in text if c in char2idx]

# === indici di caratteri -> testo --> DECODING ===
def sequence_to_text(seq, idx2char):
    """
        Funzione che dagli indici di caratteri estre il testo.
        Funzione di decoding.
    """
    return ''.join([idx2char.get(i, '') for i in seq if i != 0])


def lstm_rnn(input_dim, output_dim, dropout, n_layers, n_units):
    """
        LSTM RNN per ASR
        input_dim: numero di feature audio
        output_dim: dimensione del vocabolario + 1 (CTC blank)
    """
    inputs = tf.keras.Input(shape=(None, input_dim), name="input_spectrogram")  # (batch, time, features)
    # Ignora tutte le timestep in cui le feature sono tutte 0.0
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(n_units, return_sequences=True, dropout=dropout)
    )(x)
    if n_layers == 2:
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(n_units, return_sequences=True, dropout=dropout)
        )(x)
    elif n_layers == 3:
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(n_units, return_sequences=True, dropout=dropout)
        )(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(n_units, return_sequences=True, dropout=dropout)
        )(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    # La CTC Loss ha bisogno dei logits grezzi
    outputs = tf.keras.layers.Dense(output_dim, activation=None)(x)

    return tf.keras.Model(inputs, outputs, name="LSTM_ASR_Model")



def gru_rnn(input_dim, output_dim, dropout, n_layers, n_units):
    """
        GRU RNN per ASR
        input_dim: numero di feature audio
        output_dim: dimensione del vocabolario + 1 (CTC blank)
    """
    inputs = tf.keras.Input(shape=(None, input_dim), name="input_spectrogram")  # (batch, time, features)
    # Senza il Masking, il modello considererebbe anche questi zeri come dati validi
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(n_units, return_sequences=True, dropout=dropout)
    )(x)
    if n_layers == 2:
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(n_units, return_sequences=True, dropout=dropout)
        )(x)
    elif n_layers == 3:
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(n_units, return_sequences=True, dropout=dropout)
        )(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(n_units, return_sequences=True, dropout=dropout)
        )(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    # La CTC Loss ha bisogno dei logits grezzi
    outputs = tf.keras.layers.Dense(output_dim, activation=None)(x)

    return tf.keras.Model(inputs, outputs, name="GRU_ASR_Model")


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # y_pred: (batch, time, vocab_size) → (time, batch, vocab_size)
    y_pred = tf.transpose(y_pred, [1, 0, 2])
    # converte label dense->sparse
    labels = tf.cast(labels, tf.int32)
    label_length = tf.cast(tf.reshape(label_length, [-1]), tf.int32)
    sparse_labels = tf.keras.backend.ctc_label_dense_to_sparse(labels, label_length)
    # input_length già int32 1D
    input_length = tf.cast(tf.reshape(input_length, [-1]), tf.int32)

    loss = tf.nn.ctc_loss(
        labels=sparse_labels,
        logits=y_pred,
        label_length=label_length,
        logit_length=input_length,
        logits_time_major=True,
        blank_index=-1
    )
    return tf.reduce_mean(loss)

# Loss function chiamata per evitare il lambda serialization warning
# Da usare in model.compile() come loss function al posto di loss = lambda y_true, y_pred: y_pred
def ctc_loss_fn(y_true, y_pred):
    return y_pred

def ctc_logits_fn(y_pred):
    """
    Funzione per estrarre i logits dal modello CTC, applicando la softmax.
    """
    logits = tf.nn.softmax(y_pred['logits'], axis=-1)
    return logits


def build_ctc_model(rnn_builder, input_dim, output_dim, dropout, n_layers, n_units):
    feats = Input(shape=(None, input_dim), name="input_spectrogram")
    logits = rnn_builder(input_dim, output_dim, dropout, n_layers, n_units)(feats)
    labels = Input(name="labels", shape=(None,), dtype="int32")
    input_len = Input(name="input_length", shape=(1,), dtype="int32")
    label_len = Input(name="label_length", shape=(1,), dtype="int32")
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")([logits, labels, input_len, label_len])

    return Model(inputs=[feats, labels, input_len, label_len], outputs={"ctc": loss_out, "logits": logits}, name="CTC_Model")


def plot_loss(history, model="", dir=""):
    # Training vs Validation Loss
    plt.figure()
    plt.plot(history['loss'], marker='o', label='Training Loss')
    plt.plot(history['val_loss'], marker='o', label='Validation Loss')
    plt.title(f'{model} - Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{dir}/training_vs_validation_loss.png')
    # plt.show()

def plot_accuracy(history, model="", dir=""):
    # Training vs Validation WER
    plt.figure()
    plt.plot(history['accuracy'], marker='o', label='Training WER')
    plt.plot(history['val_accuracy'], marker='o', label='Validation WER')
    plt.title(f'{model} - Training vs Validation WER')
    plt.xlabel('Epoch')
    plt.ylabel('WER')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{dir}/training_vs_validation_wer.png')
    # plt.show()

def save_best_run(best_run, dir=""):
    with open(f"{dir}/best_run.txt", "w") as f:
        json.dump(best_run, f, indent=4)

# def CTC_loss(y_true, y_pred):
#     """
#         Loss Function CTC
#         y_true: target labels, shape (batch, max_label_length)
#         y_pred: logits (non softmax), shape (batch, time, vocab_size)
#     """
#     print("y_true shape:", y_true.shape)  # (batch, max_label_len)
#     print("y_pred shape:", y_pred.shape)  # (batch, time, vocab_size)
    
#     batch_len = tf.shape(y_true)[0]
#     label_length = tf.shape(y_true)[1]
#     logit_length = tf.shape(y_pred)[1]
    

#     label_lengths = label_length * tf.ones(shape=(batch_len,), dtype=tf.int32)
#     logit_lengths = logit_length * tf.ones(shape=(batch_len,), dtype=tf.int32)

#     # Converti y_true (dense) in sparse tensor
#     y_true = tf.cast(y_true, tf.int32)  # assicurati che siano interi
#     y_true_sparse = tf.keras.backend.ctc_label_dense_to_sparse(y_true, label_lengths)

#     # Transpose logits per shape richiesta da tf.nn.ctc_loss: (time, batch, vocab_size)
#     y_pred = tf.transpose(y_pred, [1, 0, 2])

#     # loss = tf.nn.ctc_loss(y_true, y_pred, input_length, label_length)
#     loss = tf.nn.ctc_loss(y_true_sparse, y_pred, label_lengths, logit_lengths, logits_time_major=True, blank_index=-1)
    

#     return tf.reduce_mean(loss)


# def ctc_loss(y_true, y_pred):
#     """
#     y_true:   dense (batch, max_label_len) padded con 0
#     y_pred:   logits float32 (batch, time, vocab_size)
#     input_length:  (batch, 1) int32
#     label_length:  (batch, 1) int32
#     """
#     label_length = tf.reduce_sum(
#         tf.cast(tf.not_equal(y_true, 0), tf.int32), axis=1
#     )
#     # 2) input_lengths = lunghezza time dimension
#     logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])


#     # # 1) Riduci le dimensioni (batch,1) → (batch,)
#     # input_length = tf.squeeze(input_length, axis=1)
#     # label_length = tf.squeeze(label_length, axis=1)

#     # 2) Converti y_true dense → SparseTensor
#     #    tf.keras.backend.ctc_label_dense_to_sparse si aspetta:
#     #    (dense_labels, label_length_vector)

#     # y_true = tf.cast(y_true, tf.int32)
#     sparse_labels = tf.keras.backend.ctc_label_dense_to_sparse(y_true, label_length)

#     # 3) tf.nn.ctc_loss vuole i logits TIME_MAJOR
#     #    (max_time, batch, vocab) invece che batch_major
#     logits_time_major = tf.transpose(y_pred, [1, 0, 2])

#     # 4) Calcola la loss
#     loss = tf.nn.ctc_loss(
#         labels=sparse_labels,
#         logits=logits_time_major,
#         label_length=label_length,
#         logit_length=logit_length,
#         logits_time_major=True,
#         blank_index=-1   # l’ultimo indice (vocab_size-1) come blank
#     )

#     # 5) tf.nn.ctc_loss restituisce un vettore (batch,) → prendi la media
#     return tf.reduce_mean(loss)
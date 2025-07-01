import keras
import tensorflow as tf

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
    inputs = tf.keras.Input(shape=(None, input_dim), name="input_mfcc")  # (batch, time, features)
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
    inputs = tf.keras.Input(shape=(None, input_dim), name="input_mfcc")
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


def ctc_loss(y_true, y_pred):
    """
        Loss Function CTC
        y_true: target labels, shape (batch, max_label_length)
        y_pred: logits (non softmax), shape (batch, time, vocab_size)
    """
    print("y_pred shape:", y_pred.shape)  # (batch, time, vocab_size)
    print("y_true shape:", y_true.shape)  # (batch, max_label_len)

    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

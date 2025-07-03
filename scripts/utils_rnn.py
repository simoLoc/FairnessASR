import keras
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Metric
from jiwer import wer, cer
import numpy as np
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
    outputs = tf.keras.layers.Dense(output_dim, activation=None, name="logits")(x)

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
    outputs = tf.keras.layers.Dense(output_dim, activation=None, name="logits")(x)

    return tf.keras.Model(inputs, outputs, name="GRU_ASR_Model")

@keras.saving.register_keras_serializable()
def ctc_loss_fn(y_true, y_pred):
    labels = tf.cast(y_true[:, :-2], tf.int32)
    label_length = tf.cast(y_true[:, -2], tf.int32)
    input_length = tf.cast(y_true[:, -1], tf.int32)

    batch_size = tf.shape(y_pred)[0]
    label_length = tf.reshape(label_length, [batch_size])
    input_length = tf.reshape(input_length, [batch_size])

    sparse_labels = tf.keras.backend.ctc_label_dense_to_sparse(labels, label_length)

    # Trasponi logits a time_major=True come richiesto da tf.nn.ctc_loss
    y_pred = tf.transpose(y_pred, [1, 0, 2])  # (time, batch, vocab)

    loss = tf.nn.ctc_loss(
        labels=sparse_labels,
        logits=y_pred,
        label_length=label_length,
        logit_length=input_length,
        logits_time_major=True,
        blank_index=-1
    )

    # loss shape: (batch,), quindi calcola la media sul batch
    return tf.reduce_mean(loss)


@keras.saving.register_keras_serializable()
class WERMetric(Metric):
    def __init__(self, idx2char, name='wer', **kwargs):
        super(WERMetric, self).__init__(name=name, **kwargs)
        self.idx2char = idx2char
        self.total_wer = self.add_weight(name='total_wer', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        labels = y_true  # <-- no [0]
        label_lengths = tf.math.count_nonzero(tf.cast(labels, tf.int32), axis=1)
        input_lengths = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        
        total_wer, count = tf.py_function(
            self.decode_and_wer,
            [labels, label_lengths, y_pred, input_lengths],
            [tf.float32, tf.float32]
        )
        total_wer.set_shape(())
        count.set_shape(())

        self.total_wer.assign_add(total_wer)
        self.count.assign_add(count)


    def decode_and_wer(self, labels_np, label_lengths_np, y_pred_np, input_lengths_np):
        # Decode predictions with numpy arrays
        decoded_texts = []
        for i in range(y_pred_np.shape[0]):
            logit = y_pred_np[i, :input_lengths_np[i], :]
            # Use keras ctc_decode with numpy inputs by wrapping them as tensors
            decoded, _ = tf.keras.backend.ctc_decode(
                tf.convert_to_tensor(logit[None, ...]), [input_lengths_np[i]]
            )
            pred_indices = decoded[0][0].numpy()
            pred_text = ''.join([self.idx2char.get(int(j), '') for j in pred_indices])
            decoded_texts.append(pred_text)

        total_wer = 0.0
        count = 0
        for i in range(len(labels_np)):
            label_seq = labels_np[i][:label_lengths_np[i]]
            true_text = ''.join([self.idx2char.get(int(idx), '') for idx in label_seq])
            pred = decoded_texts[i]
            total_wer += wer(true_text, pred)  # your WER function here
            count += 1

        return np.float32(total_wer), np.float32(count)

    def result(self):
        return self.total_wer / self.count

    def reset_states(self):
        self.total_wer.assign(0.0)
        self.count.assign(0.0)
    
    # PER SERIALIZZARE IDX2CHAR
    def get_config(self):
        config = super(WERMetric, self).get_config()
        config['idx2char'] = json.dumps({str(k): v for k, v in self.idx2char.items()})
        return config

    @classmethod
    def from_config(cls, config):
        idx2char_str = json.loads(config.pop('idx2char'))
        idx2char = {int(k): v for k, v in idx2char_str.items()}
        return cls(idx2char=idx2char, **config)


@keras.saving.register_keras_serializable()
class CERMetric(Metric):
    def __init__(self, idx2char, name='cer', **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.idx2char = idx2char
        self.total_cer = self.add_weight(name='total_cer', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        labels = y_true  # <-- no [0]
        label_lengths = tf.math.count_nonzero(tf.cast(labels, tf.int32), axis=1)
        input_lengths = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        
        total_cer, count = tf.py_function(
            self.decode_and_wer,
            [labels, label_lengths, y_pred, input_lengths],
            [tf.float32, tf.float32]
        )
        total_cer.set_shape(())
        count.set_shape(())

        self.total_cer.assign_add(total_cer)
        self.count.assign_add(count)


    def decode_and_wer(self, labels_np, label_lengths_np, y_pred_np, input_lengths_np):
        # Decode predictions with numpy arrays
        decoded_texts = []
        for i in range(y_pred_np.shape[0]):
            logit = y_pred_np[i, :input_lengths_np[i], :]
            # Use keras ctc_decode with numpy inputs by wrapping them as tensors
            decoded, _ = tf.keras.backend.ctc_decode(
                tf.convert_to_tensor(logit[None, ...]), [input_lengths_np[i]]
            )
            pred_indices = decoded[0][0].numpy()
            pred_text = ''.join([self.idx2char.get(int(j), '') for j in pred_indices])
            decoded_texts.append(pred_text)

        total_cer = 0.0
        count = 0
        for i in range(len(labels_np)):
            label_seq = labels_np[i][:label_lengths_np[i]]
            true_text = ''.join([self.idx2char.get(int(idx), '') for idx in label_seq])
            pred = decoded_texts[i]
            total_cer += cer(true_text, pred)  # your WER function here
            count += 1

        return np.float32(total_cer), np.float32(count)

    def result(self):
        return self.total_cer / self.count

    def reset_states(self):
        self.total_cer.assign(0.0)
        self.count.assign(0.0)

    # PER SERIALIZZARE IDX2CHAR
    def get_config(self):
        config = super(CERMetric, self).get_config()
        config['idx2char'] = json.dumps({str(k): v for k, v in self.idx2char.items()})
        return config

    @classmethod
    def from_config(cls, config):
        idx2char_str = json.loads(config.pop('idx2char'))
        idx2char = {int(k): v for k, v in idx2char_str.items()}
        return cls(idx2char=idx2char, **config)



def build_ctc_model(rnn_builder, input_dim, output_dim, dropout, n_layers, n_units):
    # logits della rnn Model
    return rnn_builder(input_dim, output_dim, dropout, n_layers, n_units)


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

def plot_wer(history, model="", dir=""):
    # Training vs Validation WER
    plt.figure()
    plt.plot(history['wer'], marker='o', label='Training WER')
    plt.plot(history['val_wer'], marker='o', label='Validation WER')
    plt.title(f'{model} - Training vs Validation WER')
    plt.xlabel('Epoch')
    plt.ylabel('WER')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{dir}/training_vs_validation_wer.png')
    # plt.show()

def plot_cer(history, model="", dir=""):
    # Training vs Validation CER
    plt.figure()
    plt.plot(history['cer'], marker='o', label='Training CER')
    plt.plot(history['val_cer'], marker='o', label='Validation CER')
    plt.title(f'{model} - Training vs Validation CER')
    plt.xlabel('Epoch')
    plt.ylabel('CER')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{dir}/training_vs_validation_cer.png')
    # plt.show()


def save_best_run(best_run, dir=""):
    with open(f"{dir}/best_run.txt", "w") as f:
        json.dump(best_run, f, indent=4)

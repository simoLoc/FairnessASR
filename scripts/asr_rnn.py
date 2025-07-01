import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import utils_dataset as utils_dataset
import librosa

# Carica il dataset
data = utils_dataset.get_dataset()

# Estrai audio e metadata separatamente per mantenere la corrispondenza con un identificatore
# Crea una lista con "Unnamed: 0" come ID univoco
audio_data = [{"Unnamed: 0": i, "audio": sample["audio"]} for i, sample in enumerate(data)]
metadata = pd.DataFrame({
    "Unnamed: 0": list(range(len(data))),
    "transcription": [sample["transcription"] for sample in data]
})

# Dividi il dataset
split = int(len(metadata) * 0.9)
df_train = metadata[:split]
df_val = metadata[split:]

print(f"Size of the training set: {len(df_train)}")
print(f"Size of the validation set: {len(df_val)}")

# Prepara il vocabolario
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# print(
#     f"The vocabulary is: {char_to_num.get_vocabulary()} "
#     f"(size ={char_to_num.vocabulary_size()})"
# )

# Parametri per STFT
frame_length = 256
frame_step = 160
fft_length = 384

# Mappa Unnamed: 0 -> audio
id_to_audio = {item["Unnamed: 0"]: item["audio"] for item in audio_data}


def encode_single_sample(id_tensor, label_tensor):
    id_val = int(id_tensor.numpy())
    label_val = label_tensor.numpy().decode("utf-8")

    audio_obj = id_to_audio[id_val]
    audio_array = audio_obj["array"]
    sampling_rate = audio_obj["sampling_rate"]

    audio_array = np.array(audio_array, dtype=np.float32)
    
    if sampling_rate != 16000:
        audio_tensor = librosa.resample(audio_array.astype(float),
                                       orig_sr=sampling_rate,
                                       target_sr=16000, dtype=tf.float32)
    else:
        audio_tensor = tf.convert_to_tensor(audio_array, dtype=tf.float32)

    spectrogram = tf.signal.stft(
        audio_tensor, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)

    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    label_tensor = tf.strings.lower(label_tensor)
    label_tensor = tf.strings.unicode_split(label_tensor, input_encoding="UTF-8")
    label_tensor = char_to_num(label_tensor)

    return spectrogram, label_tensor


def tf_encode(id_tensor, label_tensor):
    return tf.py_function(func=encode_single_sample, inp=[id_tensor, label_tensor], Tout=(tf.float32, tf.int64))

def convert_to_dataset(df, batch_size=32):
    tf_dataset = tf.data.Dataset.from_tensor_slices((df["Unnamed: 0"].values, df["transcription"].values))
    tf_dataset = tf_dataset.map(tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
    tf_dataset = tf_dataset.padded_batch(
        batch_size,
        padded_shapes=([
            None, None
        ], [None]),
        padding_values=(0.0, tf.cast(0, tf.int64))
    )
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    return tf_dataset

batch_size = 32
train_dataset = convert_to_dataset(df_train, batch_size=batch_size)
val_dataset = convert_to_dataset(df_val, batch_size=batch_size)

# Visualizzazione spettrogramma e audio
fig = plt.figure(figsize=(8, 5))
for batch in train_dataset.take(1):
    spectrogram = batch[0][0].numpy()
    label = batch[1][0]
    spectrogram = np.array([np.trim_zeros(x) for x in np.transpose(spectrogram)])
    label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")

    ax = plt.subplot(2, 1, 1)
    ax.imshow(spectrogram, aspect="auto", origin="lower")
    ax.set_title(label)
    ax.axis("off")

    audio_array = id_to_audio[int(df_train.iloc[0]["Unnamed: 0"])]
    plt.subplot(2, 1, 2)
    plt.plot(audio_array["array"])
    plt.title("Signal Wave")
    plt.xlim(0, len(audio_array["array"]))
    display(Audio(audio_array["array"], rate=16000))
plt.show()

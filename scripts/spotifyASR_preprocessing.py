from matplotlib import pyplot as plt
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import librosa
from datasets import load_dataset
import pickle
from tqdm import tqdm
from utils_rnn import text_to_sequence


# === audio -> MFCC (input RNN) ===
def extract_mfcc(audio_array, sampling_rate, n_mfcc=13):
    """
        audio_array: audio dell'entry
        sampling_rate: sampling rate dell'audio
        n_mfcc: numero di MFCC da estrarre. 

        Funzione che estra i MFCC dall'audio.
        I MFCC di un segnale audio sono un piccolo insieme di caratteristiche 
        (di solito circa 10-20) che descrivono la forma complessiva dello spettro.
    """
    mfcc = librosa.feature.mfcc(
        y=audio_array,
        sr=sampling_rate,
        n_mfcc=n_mfcc,
        hop_length=512,
        n_fft=1024
    )
    return mfcc.T  # (timesteps, features)

def extract_spectrogram(audio_array, sampling_rate):

    # Parametri per STFT
    frame_length = 256
    frame_step = 160
    fft_length = 384

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

    return spectrogram


def pad_data(X_list, y_list):
    """
        X_list: array MFCC (dimensioni variabili nel tempo)
        y_list: sequenze di indici caratteri (di lunghezza variabile)

        Le sequenze in X_list e y_list hanno lunghezze diverse.
    """
    X_padded = tf.keras.preprocessing.sequence.pad_sequences(
        X_list, padding='post', dtype='float32'
    )
    y_padded = tf.keras.preprocessing.sequence.pad_sequences(
        y_list, padding='post', value=0
    )
    return X_padded, y_padded


def plot_dataset(woman, man, aave, sae, chicano_english, spanglish, other_dialect_accent):

    woman_arr = np.array(woman)
    man_arr   = np.array(man)
    sae_arr   = np.array(sae)
    aave_arr  = np.array(aave)
    chi_arr   = np.array(chicano_english)
    spa_arr   = np.array(spanglish)
    oth_arr   = np.array(other_dialect_accent)

    # Calcolo dei conteggi
    gender_counts = [np.sum(man_arr), np.sum(woman_arr)]
    gender_classes = ['Male', 'Female']

    dialect_arrays = [sae_arr, aave_arr, chi_arr, spa_arr, oth_arr]
    dialect_counts = [np.sum(arr) for arr in dialect_arrays]
    dialect_classes = ['SAE', 'AAVE', 'Chicano English', 'Spanglish', 'Other']

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(wspace=0.4)

    # Genere
    axs[0].bar(gender_classes, gender_counts)
    axs[0].set_xlabel('Genere')
    axs[0].set_title('Distribuzione del genere')
    axs[0].tick_params(axis='x', rotation=45)

    # Dialetto
    axs[1].bar(dialect_classes, dialect_counts)
    axs[1].set_xlabel('Dialetto')
    axs[1].set_title('Distribuzione dei dialetti')
    axs[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("dataset_split/distribuzione_dataset.png")
    # plt.show()


def plot_dataset_by_dialect_and_gender(woman, man, aave, sae, chicano_english, spanglish, other_dialect_accent):
    # Convertiamo in array
    woman_arr = np.array(woman)
    man_arr   = np.array(man)
    sae_arr   = np.array(sae)
    aave_arr  = np.array(aave)
    chi_arr   = np.array(chicano_english)
    spa_arr   = np.array(spanglish)
    oth_arr   = np.array(other_dialect_accent)

    # Liste di array e nomi dei dialetti
    dialect_arrays = [sae_arr, aave_arr, chi_arr, spa_arr, oth_arr]
    dialect_names  = ['SAE', 'AAVE', 'Chicano English', 'Spanglish', 'Other']

    # Costruiamo matrici di conteggi [len(dialects) x 2]
    # colonna 0 = maschi, colonna 1 = femmine
    counts = []
    for arr in dialect_arrays:
        male_count   = np.sum(arr * man_arr)
        female_count = np.sum(arr * woman_arr)
        counts.append([male_count, female_count])
    counts = np.array(counts)

    # Parametri per il plot
    x = np.arange(len(dialect_names))   # posizioni dei gruppi
    width = 0.35                        # larghezza delle barre

    fig, ax = plt.subplots(figsize=(10, 6))
    # Barre maschi
    ax.bar(x - width/2, counts[:, 0], width, label='Male')
    # Barre femmine
    ax.bar(x + width/2, counts[:, 1], width, label='Female', color='pink')

    # Etichette e titolo
    ax.set_xlabel('Dialetto')
    ax.set_title('Distribuzione di genere per dialetto')
    ax.set_xticks(x)
    ax.set_xticklabels(dialect_names, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig("dataset_split/distribuzione_genere_per_dialetto.png")
    # plt.show()



def plot_binary_split(name, train_arr, val_arr, test_arr, class_labels):
    """
    name         : titolo del plot (es. "Genere")
    train_arr    : lista binaria 0/1 sul train
    val_arr      : idem per val
    test_arr     : idem per test
    class_labels : ['Classe0', 'Classe1']
    """
    # Conversione in np.array
    train_arr = np.array(train_arr)
    val_arr   = np.array(val_arr)
    test_arr  = np.array(test_arr)

    counts = {
        'Train': [(1 - train_arr).sum(), train_arr.sum()],
        'Val': [(1 - val_arr).sum(), val_arr.sum()],
        'Test': [(1 - test_arr).sum(), test_arr.sum()],
    }

    fig, ax = plt.subplots(1, 3, figsize=(15,4), sharey=True)
    for i, split in enumerate(['Train','Val','Test']):
        ax[i].bar(class_labels, counts[split])
        ax[i].set_title(f"{split} - {name}")
        ax[i].set_ylabel("Numero di esempi" if i==0 else "")
        ax[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig("dataset_split/genere.png")
    # plt.show()


def plot_multiclass_split(name, train_arrs, val_arrs, test_arrs, class_labels):
    """
    name         : titolo del plot (es. "Dialetto")
    train_arrs   : lista di array binari [arr_cl0, arr_cl1, ..., arr_clK]
    val_arrs     : idem per val
    test_arrs    : idem per test
    class_labels : lista di nomi di lunghezza K+1
    """
    counts = {
        'Train': [np.sum(arr) for arr in train_arrs],
        'Val':   [np.sum(arr) for arr in val_arrs],
        'Test':  [np.sum(arr) for arr in test_arrs],
    }

    fig, ax = plt.subplots(1, 3, figsize=(18,4), sharey=True)
    for i, split in enumerate(['Train','Val','Test']):
        ax[i].bar(class_labels, counts[split])
        ax[i].set_title(f"{split} - {name}")
        ax[i].set_ylabel("Numero di esempi" if i==0 else "")
        ax[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig("dataset_split/dialetti.png")
    # plt.show()



if __name__ == "__main__":
    # Creazione della directory per il preprocessing del dataset
    os.makedirs("dataset_split", exist_ok=True)

    # === Caricamento del dataset da Hugging Face ===
    dataset = load_dataset("SALT-NLP/spotify_podcast_ASR", split="train")

    # === Definizione vocabolario ===
    vocab = sorted(set("abcdefghijklmnopqrstuvwxyz '"))  # lower-case + spazio + apostrofo
    # L'indice 0 per nessun carattere / padding
    char2idx = {c: i + 1 for i, c in enumerate(vocab)}
    idx2char = {i: c for c, i in char2idx.items()}
    vocab_size = len(char2idx) + 1  # +1 per padding

    # === Lista per raccogliere dati preprocessati ===
    X = []
    y = []
    woman = []
    man = []
    aave = []
    sae = []
    chicano_english = []
    spanglish = []
    other_dialect_accent = []


    for entry in tqdm(dataset):
        audio = entry['audio']['array']
        sr = entry['audio']['sampling_rate']
        text = entry['transcription']

        # Estrazione delle feature
        spectrogram = extract_spectrogram(audio, sr)  # shape: (T, 13)
        # Creazione della label
        label = text_to_sequence(text, char2idx)  # lista di interi

        X.append(spectrogram)
        y.append(label)
        # --- Genere ---
        woman.append(entry['women'])
        man.append(entry['men'])
        # --- Dialetto ---
        aave.append(entry['aave'])
        sae.append(entry['sae'])
        chicano_english.append(entry['chicano_english'])
        spanglish.append(entry['spanglish'])
        other_dialect_accent.append(entry['other_dialect_accent'])

    plot_dataset(woman, man, aave, sae, chicano_english, spanglish, other_dialect_accent)
    plot_dataset_by_dialect_and_gender(woman, man, aave, sae, chicano_english, spanglish, other_dialect_accent)

    # === Split in train / val / test ===
    # Split 80% train + 20% temp (val+test)
    (
        X_train, X_temp, y_train, y_temp, woman_train, woman_temp, man_train, man_temp, 
        aave_train, aave_temp, sae_train, sae_temp, chicano_english_train, chicano_english_temp, 
        spanglish_train, spanglish_temp, other_dialect_accent_train, other_dialect_accent_temp
    ) = train_test_split(X, y, woman, man, aave, sae, chicano_english, spanglish, other_dialect_accent, test_size=0.2, random_state=42)

    # Split temp in 50% val e 50% test (10% val + 10% test)
    (
        X_val, X_test, y_val, y_test, woman_val, woman_test, man_val, man_test,
        aave_val, aave_test, sae_val, sae_test, chicano_english_val, chicano_english_test,
        spanglish_val, spanglish_test, other_dialect_accent_val, other_dialect_accent_test
    ) = train_test_split(X_temp, y_temp, woman_temp, man_temp, aave_temp, sae_temp, chicano_english_temp,
                        spanglish_temp, other_dialect_accent_temp, test_size=0.5, random_state=42, shuffle=True)

    # --- GENERE ---
    # usa le tue liste: man_train, man_val, man_test; woman_train, woman_val, woman_test
    plot_binary_split(
        name="Genere",
        train_arr=woman_train,  # 1=female, 0=male
        val_arr=woman_val,
        test_arr=woman_test,
        class_labels=['Male', 'Female']
    )

    # --- DIALETTI ---
    # lista di array binari per ogni dialetto (1 se appartiene)
    train_dialects = [sae_train, aave_train, chicano_english_train, spanglish_train, other_dialect_accent_train]
    val_dialects   = [sae_val, aave_val, chicano_english_val, spanglish_val, other_dialect_accent_val]
    test_dialects  = [sae_test, aave_test, chicano_english_test, spanglish_test, other_dialect_accent_test]

    plot_multiclass_split(
        name="Dialetto",
        train_arrs=train_dialects,
        val_arrs=val_dialects,
        test_arrs=test_dialects,
        class_labels=['SAE', 'AAVE', 'Chicano English', 'Spanglish', 'Other']
    )

    # === Funzione di padding ===
    X_train_padded, y_train_padded = pad_data(X_train, y_train)
    X_val_padded, y_val_padded = pad_data(X_val, y_val)
    X_test_padded, y_test_padded = pad_data(X_test, y_test)

    # === Salvataggio dataset preprocessed ===
    np.savez_compressed("dataset_split/asr_train.npz", X=X_train_padded, y=y_train_padded, sae=sae_train, aave=aave_train,
                        chicano_english=chicano_english_train, spanglish=spanglish_train, other_dialect_accent=other_dialect_accent_train,
                        woman=woman_train, man=man_train)
    np.savez_compressed("dataset_split/asr_val.npz", X=X_val_padded, y=y_val_padded, sae=sae_val, aave=aave_val, chicano_english=chicano_english_val, 
                        spanglish=spanglish_val, other_dialect_accent=other_dialect_accent_val, woman=woman_val, man=man_val
    )
    np.savez_compressed("dataset_split/asr_test.npz", X=X_test_padded, y=y_test_padded, sae=sae_test, aave=aave_test, chicano_english=chicano_english_test, 
                        spanglish=spanglish_test, other_dialect_accent=other_dialect_accent_test, woman=woman_test, man=man_test
    )

    # === Salvataggio vocabolario ===
    with open("dataset_split/char2idx.pkl", "wb") as f:
        pickle.dump(char2idx, f)

    with open("dataset_split/idx2char.pkl", "wb") as f:
        pickle.dump(idx2char, f)
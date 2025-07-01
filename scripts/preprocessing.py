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


if __name__ == "__main__":
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

    for entry in tqdm(dataset):
        audio = entry['audio']['array']
        sr = entry['audio']['sampling_rate']
        text = entry['transcription']
        
        # Estrazione delle feature
        mfcc = extract_mfcc(audio, sr)  # shape: (T, 13)
        # Creazione della label
        label = text_to_sequence(text, char2idx)  # lista di interi

        X.append(mfcc)
        y.append(label)

    # === Split in train / val / test ===
    # Prima split 80% train + 20% temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

    # Poi split temp in 50% val e 50% test (10% val + 10% test)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # === Funzione di padding ===

    X_train_padded, y_train_padded = pad_data(X_train, y_train)
    X_val_padded, y_val_padded = pad_data(X_val, y_val)
    X_test_padded, y_test_padded = pad_data(X_test, y_test)

    # === Salvataggio dataset preprocessed ===
    np.savez_compressed("asr_train.npz", X=X_train_padded, y=y_train_padded)
    np.savez_compressed("asr_val.npz", X=X_val_padded, y=y_val_padded)
    np.savez_compressed("asr_test.npz", X=X_test_padded, y=y_test_padded)

    # === Salvataggio vocabolario ===
    with open("char2idx.pkl", "wb") as f:
        pickle.dump(char2idx, f)

    with open("idx2char.pkl", "wb") as f:
        pickle.dump(idx2char, f)
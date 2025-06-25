# asr_gender_evaluation.py

import torch
import torchaudio
import nltk
from datasets import load_dataset
from huggingface_hub import login
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from nltk import word_tokenize, edit_distance
import numpy as np
import pandas as pd
import os
from jiwer import wer, cer
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils_dataset as utils_dataset


# Calcola le metriche di WER (Word Error Rate) e CER (Character Error Rate) tra 
# la trascrizione di riferimento e quella predetta dal modello 
def calcola_wer_cer(reference, hypothesis):
    return wer(reference, hypothesis), cer(reference, hypothesis)


def carica_asr_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    processor.batch_size = 1
    processor.truncation = "longest_first"
    processor.chunk_length_s = 30
    processor.stride_length_s = (4, 2)

    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.to(device)

    model.eval()
    return processor, model


def get_transcription(audio_array, sampling_rate, processor, model):

    # Converte l'audio in un tensore di tipo float (NO double)
    waveform = torch.tensor(audio_array, dtype=torch.float)

    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        audio_array = resampler(waveform).squeeze().numpy()

    input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding="longest")

    with torch.no_grad():
        logits = model(input_features.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        torch.cuda.empty_cache()
    
    return transcription


def evaluate_by_group(dataset, processor, model, group_name):
    wer_list = []
    cer_list = []

    print(f"\nValutazione gruppo: {group_name} | Totale: {len(dataset)}")

    for i, sample in enumerate(tqdm(dataset)):
        
        try:
            audio = sample["audio"]
            transcription_ref = sample["transcription"].lower()
            transcription_pred = get_transcription(audio["array"], audio["sampling_rate"], processor, model)
            
            transcription_pred = transcription_pred.lower()

            # print(f"True Campione {i+1}/{len(dataset)}: {transcription_ref}")
            # print(f"Predetto Campione {i+1}/{len(dataset)}: {transcription_pred}")

            w, c = calcola_wer_cer(transcription_ref, transcription_pred)
            wer_list.append(w)
            cer_list.append(c)
        except Exception as e:
            print(f"Errore nel campione {i}: {e}")
            continue

    return {
        "Group": group_name,
        "Samples": len(wer_list),
        "WER Mean": round(np.mean(wer_list), 4),
        "WER List": wer_list,
        "CER Mean": round(np.mean(cer_list), 4),
        "CER List": cer_list
    }




def plot_metrics(results, title_prefix, category):
    # Crea la cartella plots se non esiste
    os.makedirs("plots", exist_ok=True)

    sample_step = 1

    # --- WER Trend ---
    plt.figure(figsize=(10, 6))
    for r in results:
        group = r["Group"]
        wer_values = r["WER List"][::sample_step]
        x = np.arange(len(wer_values))
        plt.plot(x, wer_values, marker='o', linestyle='-', label=group)
    plt.title(f"{title_prefix} - Word Error Rate (WER)")
    plt.xlabel(f"Campione (ogni {sample_step})")
    plt.ylabel("WER")
    plt.xticks(rotation=20)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{category.lower()}_wer_trend.png")
    plt.close()

    # --- CER Trend ---
    plt.figure(figsize=(10, 6))
    for r in results:
        group = r["Group"]
        cer_values = r["CER List"][::sample_step]
        x = np.arange(len(cer_values))
        plt.plot(x, cer_values, marker='s', linestyle='-', label=group)
    plt.title(f"{title_prefix} - Character Error Rate (CER)")
    plt.xlabel(f"Campione (ogni {sample_step})")
    plt.ylabel("CER")
    plt.xticks(rotation=20)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{category.lower()}_cer_trend.png")
    plt.close()

def main():

    model_name = "facebook/wav2vec2-base-960h"
    # Numero massimo di campioni da valutare per ogni gruppo considerato
    # num_samples_max = 10  

    ds = utils_dataset.get_dataset()

    # Divisione dei campioni in base al genere
    men = ds.filter(lambda x: x["men"] == 1)
    women = ds.filter(lambda x: x["women"] == 1)

    # Divisione dei campioni in base ai dialetti
    aave = ds.filter(lambda x: x["aave"] == 1)
    sae = ds.filter(lambda x: x["sae"] == 1)
    spanglish = ds.filter(lambda x: x["spanglish"] == 1)
    chicano = ds.filter(lambda x: x["chicano_english"] == 1)

    processor, model = carica_asr_model(model_name)


    results = []
    results.append(evaluate_by_group(men, processor, model, "Men"))
    results.append(evaluate_by_group(women, processor, model, "Women"))

    df = pd.DataFrame(results)
    df = df[["Group", "Samples", "WER Mean", "CER Mean"]]
    
    print("\nRisultati per gender:")
    print(df.to_string(index=False))

    plot_metrics(results, "Valutazione per genere", "gender")

    results = []
    results.append(evaluate_by_group(aave, processor, model, "African American Vernacular English (AAVE)"))
    results.append(evaluate_by_group(sae, processor, model, "Standard American English (SAE)"))
    results.append(evaluate_by_group(spanglish, processor, model, "Spanglish"))
    results.append(evaluate_by_group(chicano, processor, model, "Chicano English"))

    df = pd.DataFrame(results)
    df = df[["Group", "Samples", "WER Mean", "CER Mean"]]

    print("\nRisultati per dialetti:")
    print(df.to_string(index=False))
    
    plot_metrics(results, "Valutazione per dialetti", "dialetto")


if __name__ == "__main__":
    main()


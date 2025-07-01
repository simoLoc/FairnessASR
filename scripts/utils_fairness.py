import librosa
import numpy as np
import pandas as pd
import os
import json
from jiwer import wer, cer
import matplotlib.pyplot as plt
from tqdm import tqdm

# Calcola le metriche di WER (Word Error Rate) e CER (Character Error Rate) tra 
# la trascrizione di riferimento e quella predetta dal modello 
def calcola_wer_cer(reference, hypothesis):
    return wer(reference, hypothesis), cer(reference, hypothesis)


# # Funzione per il plot delle metriche WER e CER per categoria considerata e modello utilizzato
# def plot_metrics(results, title_prefix, category, model):
#     # Crea la cartella plots se non esiste
#     os.makedirs(f"plots/{model}", exist_ok=True)

#     sample_step = 1

#     # --- Plot di WER ---
#     plt.figure(figsize=(10, 6))
#     for r in results:
#         group = r["Group"]
#         wer_values = r["WER List"][::sample_step]
#         x = np.arange(len(wer_values))
#         plt.plot(x, wer_values, marker='o', linestyle='-', label=group)
#     plt.title(f"{title_prefix} - Word Error Rate (WER)")
#     plt.xlabel(f"Campione (ogni {sample_step})")
#     plt.ylabel("WER")
#     plt.xticks(rotation=20)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"plots/{model}/{category.lower()}_wer.png")
#     plt.close()

#     # --- Plot di CER ---
#     plt.figure(figsize=(10, 6))
#     for r in results:
#         group = r["Group"]
#         cer_values = r["CER List"][::sample_step]
#         x = np.arange(len(cer_values))
#         plt.plot(x, cer_values, marker='s', linestyle='-', label=group)
#     plt.title(f"{title_prefix} - Character Error Rate (CER)")
#     plt.xlabel(f"Campione (ogni {sample_step})")
#     plt.ylabel("CER")
#     plt.xticks(rotation=20)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"plots/{model}/{category.lower()}_cer.png")
#     plt.close()

def plot_metrics(results, title_prefix, category, model):
    os.makedirs(f"plots/{model}", exist_ok=True)

    bins=np.arange(0, 1.3, 0.1)
    groups = [r["Group"] for r in results]
    wer_histograms = []
    cer_histograms = []

    # Calcola gli istogrammi (conteggi) per ciascun gruppo
    for r in results:
        wer_hist, _ = np.histogram(r["WER List"], bins=bins)
        cer_hist, _ = np.histogram(r["CER List"], bins=bins)
        wer_histograms.append(wer_hist)
        cer_histograms.append(cer_hist)

    bin_labels = [f"{round(bins[i], 1)}" for i in range(len(bins) - 1)]
    x = np.arange(len(bin_labels))  # posizioni delle barre
    bar_width = 0.35

    # Colori per i gruppi
    cmap = plt.get_cmap("Set2")
    colors = [cmap(i) for i in range(len(groups))]

    
    # --- Barplot WER ---
    plt.figure(figsize=(12, 6))
    for i, hist in enumerate(wer_histograms):
        plt.bar(x + i * bar_width, hist, width=bar_width, label=groups[i], color=colors[i])
    plt.title(f"{title_prefix} - Distribuzione WER")
    plt.xlabel("WER")
    plt.ylabel("Numero campioni")
    plt.xticks(x + bar_width * (len(groups)-1) / 2, bin_labels, rotation=45)
    plt.grid(True, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{model}/{category.lower()}_wer.png")
    plt.close()

    # --- Barplot CER ---
    plt.figure(figsize=(12, 6))
    for i, hist in enumerate(cer_histograms):
        plt.bar(x + i * bar_width, hist, width=bar_width, label=groups[i], color=colors[i])
    plt.title(f"{title_prefix} - Distribuzione CER")
    plt.xlabel("CER")
    plt.ylabel("Numero campioni")
    plt.xticks(x + bar_width * (len(groups)-1) / 2, bin_labels, rotation=45)
    plt.grid(True, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{model}/{category.lower()}_cer.png")
    plt.close()


def save_results(results, model_name, category):

    # Crea la cartella results se non esiste
    os.makedirs(f"results/{model_name}", exist_ok=True)        

    # Salva il dizionario results in un file txt formattato
    txt_path = f"results/{model_name}/{category}_results.txt"
    with open(txt_path, "w") as f:
        json.dump(results, f, indent=4)

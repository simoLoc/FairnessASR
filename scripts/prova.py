from datasets import load_dataset
from huggingface_hub import login
from transformers import Wav2Vec2Processor, TFWav2Vec2ForCTC, WhisperProcessor, WhisperForConditionalGeneration
import tensorflow as tf
import librosa
import numpy as np
import pandas as pd
import os
import json
from jiwer import wer, cer
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils_dataset as utils_dataset


# Calcola le metriche di WER (Word Error Rate) e CER (Character Error Rate) tra 
# la trascrizione di riferimento e quella predetta dal modello 
def calcola_wer_cer(reference, hypothesis):
    return wer(reference, hypothesis), cer(reference, hypothesis)


def load_model(model_name):

    if "whisper" in model_name:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
    else:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = TFWav2Vec2ForCTC.from_pretrained(model_name)

        # Parametri del processor specifici per Wav2Vec2
        processor.batch_size = 1
        processor.truncation = "longest_first"
        processor.chunk_length_s = 30
        processor.stride_length_s = (4, 2)

    # TensorFlow rileva automaticamente GPU/CPU
    # Mettiamo il modello in "inference mode"
    model.trainable = False

    return processor, model

def get_transcription(audio_array, sampling_rate, processor, model, model_name):

    if sampling_rate != 16000:
        audio_array = librosa.resample(audio_array.astype(float),
                                       orig_sr=sampling_rate,
                                       target_sr=16000)
    
    if "whisper" in model_name:
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"]

        generated_ids = model.generate(input_features)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    else:
        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="tf",
            padding="longest"
        )

        outputs = model(inputs.input_values, training=False)
        
        logits = outputs.logits
        predicted_ids = tf.argmax(logits, axis=-1)
        
        transcription = processor.batch_decode(predicted_ids)[0]
        
    return transcription


def evaluate_by_group(dataset, processor, model, group_name, model_name):
    wer_list = []
    cer_list = []

    print(f"\nValutazione gruppo: {group_name} | Totale: {len(dataset)}")

    for i, sample in enumerate(tqdm(dataset)):
        
        try:
            audio = sample["audio"]
            transcription_ref = sample["transcription"].lower()
            transcription_pred = get_transcription(audio["array"], audio["sampling_rate"], processor, model, model_name)
            
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



def plot_metrics(results, title_prefix, category, model):
    # Crea la cartella plots se non esiste
    os.makedirs(f"plots/{model}", exist_ok=True)

    sample_step = 1

    # --- Plot di WER ---
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
    plt.savefig(f"plots/{model}/{category.lower()}_wer.png")
    plt.close()

    # --- Plot di CER ---
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
    plt.savefig(f"plots/{model}/{category.lower()}_cer.png")
    plt.close()


def get_results_by_category(dataset, processor, model, category, model_name):
    
    # Numero massimo di campioni da valutare per ogni gruppo considerato
    num_samples_max = 10  

    if category == "overall":
        # Valutazione complessiva del dataset
        return [evaluate_by_group(dataset.select(range(num_samples_max)), processor, model, "Overall", model_name)]

    elif category == "gender":
        # Divisione dei campioni in base al genere
        men = dataset.filter(lambda x: x["men"] == 1 and x["women"] == 0).select(range(num_samples_max))
        women = dataset.filter(lambda x: x["women"] == 1 and x["men"] == 0).select(range(num_samples_max))

        results = []
        results.append(evaluate_by_group(men, processor, model, "Men", model_name))
        results.append(evaluate_by_group(women, processor, model, "Women", model_name))
    elif category == "dialect":
        # Divisione dei campioni in base ai dialetti
        aave = dataset.filter(lambda x: x["aave"] == 1 and x["sae"] == 0 and x["spanglish"] == 0 and x["chicano_english"] == 0).select(range(num_samples_max))
        sae = dataset.filter(lambda x: x["sae"] == 1 and x["aave"] == 0 and x["spanglish"] == 0 and x["chicano_english"] == 0).select(range(num_samples_max))
        spanglish = dataset.filter(lambda x: x["spanglish"] == 1 and x["aave"] == 0 and x["sae"] == 0 and x["chicano_english"] == 0).select(range(num_samples_max))
        chicano = dataset.filter(lambda x: x["chicano_english"] == 1 and x["aave"] == 0 and x["sae"] == 0 and x["spanglish"] == 0).select(range(num_samples_max))

        results = []
        results.append(evaluate_by_group(aave, processor, model, "African American Vernacular English (AAVE)", model_name))
        results.append(evaluate_by_group(sae, processor, model, "Standard American English (SAE)", model_name))
        results.append(evaluate_by_group(spanglish, processor, model, "Spanglish", model_name))
        results.append(evaluate_by_group(chicano, processor, model, "Chicano English", model_name))
    elif category == "gender_dialect":
        # Divisione dei campioni in base a dialetto e genere
        aave_men = dataset.filter(lambda x: x["aave"] == 1 and x["men"] == 1 and x["sae"] == 0 and x["spanglish"] == 0 and x["chicano_english"] == 0 and x["women"] == 0).select(range(num_samples_max))
        aave_women = dataset.filter(lambda x: x["aave"] == 1 and x["women"] == 1 and x["sae"] == 0 and x["spanglish"] == 0 and x["chicano_english"] == 0 and x["men"] == 0).select(range(num_samples_max))

        sae_men = dataset.filter(lambda x: x["sae"] == 1 and x["men"] == 1 and x["aave"] == 0 and x["spanglish"] == 0 and x["chicano_english"] == 0 and x["women"] == 0).select(range(num_samples_max))
        sae_women = dataset.filter(lambda x: x["sae"] == 1 and x["women"] == 1 and x["aave"] == 0 and x["spanglish"] == 0 and x["chicano_english"] == 0 and x["men"] == 0).select(range(num_samples_max))

        spanglish_men = dataset.filter(lambda x: x["spanglish"] == 1 and x["men"] == 1 and x["aave"] == 0 and x["sae"] == 0 and x["chicano_english"] == 0 and x["women"] == 0).select(range(num_samples_max))
        spanglish_women = dataset.filter(lambda x: x["spanglish"] == 1 and x["women"] == 1 and x["aave"] == 0 and x["sae"] == 0 and x["chicano_english"] == 0 and x["men"] == 0).select(range(num_samples_max))

        chicano_men = dataset.filter(lambda x: x["chicano_english"] == 1 and x["men"] == 1 and x["aave"] == 0 and x["sae"] == 0 and x["spanglish"] == 0 and x["women"] == 0).select(range(num_samples_max))
        chicano_women = dataset.filter(lambda x: x["chicano_english"] == 1 and x["women"] == 1 and x["aave"] == 0 and x["sae"] == 0 and x["spanglish"] == 0 and x["men"] == 0).select(range(num_samples_max))

        results = []
        results.append(evaluate_by_group(aave_men, processor, model, "AAVE Men", model_name))
        results.append(evaluate_by_group(aave_women, processor, model, "AAVE Women", model_name))
        results.append(evaluate_by_group(sae_men, processor, model, "SAE Man", model_name))
        results.append(evaluate_by_group(sae_women, processor, model, "SAE Women", model_name))
        results.append(evaluate_by_group(spanglish_men, processor, model, "Spanglish Men", model_name))
        results.append(evaluate_by_group(spanglish_women, processor, model, "Spanglish Women", model_name))
        results.append(evaluate_by_group(chicano_men, processor, model, "Chicano English Men", model_name))
        results.append(evaluate_by_group(chicano_women, processor, model, "Chicano English Women", model_name))

    return results

def main():

    # I due modelli utilizzati sono Wav2Vec2 Base e Whisper Medium 
    model_list = ["openai/whisper-medium",
                  "facebook/wav2vec2-base-960h" ]

    # Numero massimo di campioni da valutare per ogni gruppo considerato
    # num_samples_max = 10  

    # Caricamento del dataset
    ds = utils_dataset.get_dataset()

    dataset = ds["train"] 

    # for model_name in model_list:
    #     print(f"\nCaricamento del modello: {model_name}")
    #     processor, model = load_model(model_name)

    #     # Crea una cartella per ogni modello se non esiste
    #     model_dir = model_name.split("/")[-1]
    #     os.makedirs(f"results/{model_dir}", exist_ok=True)
        
    #     categories = ["overall", "gender", "dialect", "gender_dialect"]

    #     for category in categories:
    #         results = get_results_by_category(data, processor, model, category, model_dir)

    #         df = pd.DataFrame(results)
    #         df = df[["Group", "Samples", "WER Mean", "CER Mean"]]
        
    #         print(f"\nRisultati per {category}:")
    #         print(df.to_string(index=False))

    #         # Salva il dizionario results in un file txt formattato
    #         txt_path = f"results/{model_dir}/{category}_results.txt"
    #         with open(txt_path, "w") as f:
    #             json.dump(results, f, indent=4)

    #         plot_metrics(results, f"Valutazione per {category}", category, model_dir)

    men = dataset.filter(lambda x: x["men"] == 1 and x["women"] == 0)
    women = dataset.filter(lambda x: x["women"] == 1 and x["men"] == 0)

    # Divisione dei campioni in base ai dialetti
    aave = dataset.filter(lambda x: x["aave"] == 1 and x["sae"] == 0 and x["spanglish"] == 0 and x["chicano_english"] == 0)
    sae = dataset.filter(lambda x: x["sae"] == 1 and x["aave"] == 0 and x["spanglish"] == 0 and x["chicano_english"] == 0)
    spanglish = dataset.filter(lambda x: x["spanglish"] == 1 and x["aave"] == 0 and x["sae"] == 0 and x["chicano_english"] == 0)
    chicano = dataset.filter(lambda x: x["chicano_english"] == 1 and x["aave"] == 0 and x["sae"] == 0 and x["spanglish"] == 0)
    others = dataset.filter(lambda x: x["aave"] == 0 and x["sae"] == 0 and x["spanglish"] == 0 and x["chicano_english"] == 0 and x["other_dialect_accent"] == 1)

    # Divisione dei campioni in base a dialetto e genere
    aave_men = dataset.filter(lambda x: x["aave"] == 1 and x["men"] == 1 and x["sae"] == 0 and x["spanglish"] == 0 and x["chicano_english"] == 0 and x["women"] == 0)
    aave_women = dataset.filter(lambda x: x["aave"] == 1 and x["women"] == 1 and x["sae"] == 0 and x["spanglish"] == 0 and x["chicano_english"] == 0 and x["men"] == 0)

    sae_men = dataset.filter(lambda x: x["sae"] == 1 and x["men"] == 1 and x["aave"] == 0 and x["spanglish"] == 0 and x["chicano_english"] == 0 and x["women"] == 0)
    sae_women = dataset.filter(lambda x: x["sae"] == 1 and x["women"] == 1 and x["aave"] == 0 and x["spanglish"] == 0 and x["chicano_english"] == 0 and x["men"] == 0)

    spanglish_men = dataset.filter(lambda x: x["spanglish"] == 1 and x["men"] == 1 and x["aave"] == 0 and x["sae"] == 0 and x["chicano_english"] == 0 and x["women"] == 0)
    spanglish_women = dataset.filter(lambda x: x["spanglish"] == 1 and x["women"] == 1 and x["aave"] == 0 and x["sae"] == 0 and x["chicano_english"] == 0 and x["men"] == 0)

    chicano_men = dataset.filter(lambda x: x["chicano_english"] == 1 and x["men"] == 1 and x["aave"] == 0 and x["sae"] == 0 and x["spanglish"] == 0 and x["women"] == 0)
    chicano_women = dataset.filter(lambda x: x["chicano_english"] == 1 and x["women"] == 1 and x["aave"] == 0 and x["sae"] == 0 and x["spanglish"] == 0 and x["men"] == 0)

    print("Numero di campioni men only:", len(men))
    print("Numero di campioni women only:", len(women))
    print("Numero di campioni AAVE:", len(aave))
    print("Numero di campioni SAE:", len(sae))
    print("Numero di campioni Spanglish:", len(spanglish))
    print("Numero di campioni Chicano English:", len(chicano))
    print("Numero di campioni Others:", len(others))
    print("Numero di campioni AAVE Men:", len(aave_men))
    print("Numero di campioni AAVE Women:", len(aave_women))
    print("Numero di campioni SAE Men:", len(sae_men))
    print("Numero di campioni SAE Women:", len(sae_women))
    print("Numero di campioni Spanglish Men:", len(spanglish_men))
    print("Numero di campioni Spanglish Women:", len(spanglish_women))
    print("Numero di campioni Chicano Men:", len(chicano_men))
    print("Numero di campioni Chicano Women:", len(chicano_women))


if __name__ == "__main__":
    main()


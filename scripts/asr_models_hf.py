import torch
import torchaudio
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2ForCTC
)
import numpy as np
import pandas as pd
import os
import json
from jiwer import wer, cer
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils_dataset as utils_dataset
import utils_fairness as utils_fairness


def load_model(model_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "whisper" in model_name:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    else:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)

    # Mettiamo il modello in "inference mode"
    model.eval()

    return processor, model, device


def get_transcription(audio_array, sampling_rate, processor, model, model_name, device):

    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        audio_array = resampler(torch.tensor(audio_array, dtype=torch.float32)).numpy()

    if "whisper" in model_name:
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(device)

        with torch.no_grad():
            generated_ids = model.generate(input_features)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding="longest")
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

    return transcription


def evaluate_by_group(dataset, processor, model, group_name, model_name, device):
    wer_list = []
    cer_list = []

    print(f"\nValutazione gruppo: {group_name} | Totale: {len(dataset)}")

    for i, sample in enumerate(tqdm(dataset)):
        
        try:
            audio = sample["audio"]
            transcription_ref = sample["transcription"].lower()
            transcription_pred = get_transcription(audio["array"], audio["sampling_rate"], processor, model, model_name, device)
            
            transcription_pred = transcription_pred.lower()

            # print(f"True Campione {i+1}/{len(dataset)}: {transcription_ref}")
            # print(f"Predetto Campione {i+1}/{len(dataset)}: {transcription_pred}")

            w, c = utils_fairness.calcola_wer_cer(transcription_ref, transcription_pred)
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


def get_results_by_category(dataset, processor, model, category, model_name, device):
    
    if category == "overall":
        # Valutazione complessiva del dataset
        return [evaluate_by_group(dataset, processor, model, "Overall", model_name, device)]

    elif category == "gender":
        # Divisione dei campioni in base al genere
        men = dataset.filter(lambda x: x["men"] == 1 and x["women"] == 0)
        women = dataset.filter(lambda x: x["women"] == 1 and x["men"] == 0)

        results = []
        results.append(evaluate_by_group(men, processor, model, "Men", model_name, device))
        results.append(evaluate_by_group(women, processor, model, "Women", model_name, device))

    elif category == "dialect":
        # Divisione dei campioni in base ai dialetti
        aave = dataset.filter(lambda x: x["aave"] == 1)
        sae = dataset.filter(lambda x: x["sae"] == 1)
        spanglish = dataset.filter(lambda x: x["spanglish"] == 1)
        chicano = dataset.filter(lambda x: x["chicano_english"] == 1)
        others = dataset.filter(lambda x: x["other_dialect_accent"] == 1)

        results = []
        results.append(evaluate_by_group(aave, processor, model, "African American Vernacular English (AAVE)", model_name, device))
        results.append(evaluate_by_group(sae, processor, model, "Standard American English (SAE)", model_name, device))
        results.append(evaluate_by_group(spanglish, processor, model, "Spanglish", model_name, device))
        results.append(evaluate_by_group(chicano, processor, model, "Chicano English", model_name, device))
        results.append(evaluate_by_group(others, processor, model, "Other Dialects", model_name, device))

    elif category == "gender_dialect":
        # Divisione dei campioni in base a dialetto e genere
        aave_men = dataset.filter(lambda x: x["aave"] == 1 and x["men"] == 1 and x["women"] == 0)
        aave_women = dataset.filter(lambda x: x["aave"] == 1 and x["women"] == 1 and x["men"] == 0)

        sae_men = dataset.filter(lambda x: x["sae"] == 1 and x["men"] == 1 and x["women"] == 0)
        sae_women = dataset.filter(lambda x: x["sae"] == 1 and x["women"] == 1 and x["men"] == 0)

        spanglish_men = dataset.filter(lambda x: x["spanglish"] == 1 and x["men"] == 1 and x["women"] == 0)
        spanglish_women = dataset.filter(lambda x: x["spanglish"] == 1 and x["women"] == 1 and x["men"] == 0)

        chicano_men = dataset.filter(lambda x: x["chicano_english"] == 1 and x["men"] == 1 and x["women"] == 0)
        chicano_women = dataset.filter(lambda x: x["chicano_english"] == 1 and x["women"] == 1 and x["men"] == 0)

        other_men = dataset.filter(lambda x: x["other_dialect_accent"] == 1 and x["men"] == 1 and x["women"] == 0)
        other_women = dataset.filter(lambda x: x["other_dialect_accent"] == 1 and x["women"] == 1 and x["men"] == 0)


        results = []
        results.append(evaluate_by_group(aave_men, processor, model, "AAVE Men", model_name, device))
        results.append(evaluate_by_group(aave_women, processor, model, "AAVE Women", model_name, device))
        results.append(evaluate_by_group(sae_men, processor, model, "SAE Man", model_name, device))
        results.append(evaluate_by_group(sae_women, processor, model, "SAE Women", model_name, device))
        results.append(evaluate_by_group(spanglish_men, processor, model, "Spanglish Men", model_name, device))
        results.append(evaluate_by_group(spanglish_women, processor, model, "Spanglish Women", model_name, device))
        results.append(evaluate_by_group(chicano_men, processor, model, "Chicano English Men", model_name, device))
        results.append(evaluate_by_group(chicano_women, processor, model, "Chicano English Women", model_name, device))
        results.append(evaluate_by_group(other_men, processor, model, "Other Dialects Men", model_name, device))
        results.append(evaluate_by_group(other_women, processor, model, "Other Dialects Women", model_name, device))

    return results

def main_asr_model_hf():

    # I due modelli utilizzati sono Wav2Vec2 Base e Whisper Medium 
    model_list = ["facebook/wav2vec2-base-960h", 
                  "openai/whisper-medium"]

    # Caricamento del dataset
    data = utils_dataset.get_dataset()
    
    for model_name in model_list:
        print(f"\nCaricamento del modello: {model_name}")
        processor, model, device = load_model(model_name)

        # Crea una cartella per ogni modello se non esiste
        model_dir = model_name.split("/")[-1]
        
        categories = ["overall", "gender", "dialect", "gender_dialect"]

        for category in categories:
            results = get_results_by_category(data, processor, model, category, model_dir, device)

            df = pd.DataFrame(results)
            df = df[["Group", "Samples", "WER Mean", "CER Mean"]]
        
            print(f"\nRisultati per {category}:")
            print(df.to_string(index=False))
            
            utils_fairness.save_results(results, model_dir, category)

            utils_fairness.plot_metrics(results, f"Valutazione per {category}", category, model_dir)


if __name__ == "__main__":
    main_asr_model_hf()

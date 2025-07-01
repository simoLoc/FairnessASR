from datasets import load_dataset
from huggingface_hub import login

def get_dataset():
    # Login to Hugging Face Hub
    login("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")

    # Load the dataset
    ds = load_dataset("SALT-NLP/spotify_podcast_ASR")
    data = ds['train']

    return data


def print_dataset_info(data):
    # Numero elementi del dataset
    num_elements = len(data)
    print(f"Numero di elementi nel dataset: {num_elements}")

    # Mostra i primi 3 esempi 
    for i in range(3):
        print(f"\nEsempio {i+1}:")
        print(data[i])    

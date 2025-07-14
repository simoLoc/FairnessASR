# FairnessASR

FairnessASR è un progetto di ricerca che analizza l'equità nei sistemi di riconoscimento automatico del parlato (ASR), con particolare attenzione alle potenziali disparità nelle 
performance dei modelli rispetto al **genere** e al **dialetto** degli speaker. L'obiettivo principale è valutare se e in che misura i modelli ASR presentano **bias sistematici**,
penalizzando alcune categorie di parlanti nella trascrizione automatica. Il progetto mira a contribuire allo sviluppo di sistemi più equi e rappresentativi dal punto di vista 
sociolinguistico.

Sono state adottate due strategie principali:
1. **Progettazione di modelli RNN**: implementazioni con LSTM e GRU per l’ASR;
2. **Inferenza con modelli preaddestrati** della libreria Hugging Face:
   - [`openai/whisper-medium`](https://huggingface.co/openai/whisper-medium)
   - [`facebook/wav2vec2-base`](https://huggingface.co/facebook/wav2vec2-base)
   
L’analisi si è focalizzata sui modelli preaddestrati, mentre quella relativa alle RNN è stata rimandata a sviluppi futuri, a causa dell’elevato costo computazionale richiesto per il loro 
addestramento. Le prestazioni sono state valutate in base a genere, dialetto e la combinazione dei due fattori, mediante le metriche Word Error Rate (WER) e Character Error Rate (CER). 
**Whisper-medium** ha superato **Wav2Vec2-base** in tutte le metriche, grazie alla sua architettura multilingue e alla maggiore robustezza nei confronti della variabilità linguistica. 
Lo **Standard American English (SAE)** ha registrato le migliori performance, probabilmente per la sua maggiore rappresentazione nel dataset. Non sono emerse disparità significative tra
i generi, anche se le voci femminili tendono a ottenere risultati leggermente migliori.

## Dataset utilizzato
[`spotify_podcast_ASR`](https://huggingface.co/datasets/SALT-NLP/spotify_podcast_ASR), disponibile sulla piattaforma HuggingFace.

Per utilizzarlo è necessario importarlo nel seguente modo
```
   from datasets import load_dataset
   ds = load_dataset("SALT-NLP/spotify_podcast_ASR")
```

## Struttura della pipeline
<img width="3840" height="957" alt="DL_pipeline (2)" src="https://github.com/user-attachments/assets/8763eee4-9d89-49cc-b934-cf83cef0cfed" />

## Requisiti
L'esecuzione del progetto richiede l'installazione delle dipendenze, da eseguirsi tramite il comando
```
   pip install -r requirements.txt
```

## Autori e Contatti
| Autore              | Indirizzo email                |
|---------------------|--------------------------------|
| Simona Lo Conte     | s.loconte2@studenti.unisa.it   |
| Marta Napolillo     | m.napolillo1@studenti.unisa.it |

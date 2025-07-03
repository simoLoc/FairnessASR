from asr_models_hf import main_asr_model_hf
from train_rnn import main_rnn


if __name__ == "__main__":
    print("### PARTE 1")
    print("=== ESECUZIONE MODELLI HUGGINGFACE ===")
    main_asr_model_hf()

    print("### PARTE 2")
    print("=== ESECUZIONE MODELLI RNN ===")
    main_rnn()
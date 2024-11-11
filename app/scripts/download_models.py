from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoForCausalLM, GPT2Tokenizer
import os

MODELS = {
    'gpt-neo': 'EleutherAI/gpt-neo-1.3B',
    'phi-2': 'microsoft/phi-2',
    'neural-chat': 'Intel/neural-chat-7b-v3-1'
}

def download_models():
    cache_dir = "/home/ubuntu/.cache/huggingface"
    os.makedirs(cache_dir, exist_ok=True)
    
    for model_id, model_name in MODELS.items():
        print(f"Downloading {model_name}...")
        
        if model_id == 'gpt-neo':
            model = GPTNeoForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
            tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        print(f"Downloaded {model_name}")

if __name__ == "__main__":
    download_models() 
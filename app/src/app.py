from flask import Flask, request, jsonify, render_template
import sys
import logging
import requests
from tqdm import tqdm
import os
import torch
from transformers import (
    GPTNeoForCausalLM, 
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Tokenizer
)
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set up the Flask app with explicit template folder
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
app = Flask(__name__, 
    template_folder=template_dir,
    static_folder=static_dir)

# Model configurations
MODELS = {
    'gpt-neo': {
        'name': 'EleutherAI/gpt-neo-1.3B',
        'display_name': 'GPT-Neo 1.3B',
        'description': 'Open-source language model with 1.3B parameters',
        'max_length': 100,
        'temperature': 0.7
    },
    'phi-2': {
        'name': 'microsoft/phi-2',
        'display_name': 'Microsoft Phi-2',
        'description': 'Compact and efficient 2.7B parameter model',
        'max_length': 256,
        'temperature': 0.7
    },
    'neural-chat': {
        'name': 'Intel/neural-chat-7b-v3-1',
        'display_name': 'Intel Neural Chat 7B',
        'description': 'CPU-optimized conversational model',
        'max_length': 512,
        'temperature': 0.8
    }
}

# Global model cache
loaded_models = {}
loaded_tokenizers = {}

def load_model_and_tokenizer(model_id):
    if model_id not in loaded_models:
        config = MODELS[model_id]
        model_name = config['name']
        
        print(f"Loading model: {model_name}")
        
        if model_id == 'gpt-neo':
            model = GPTNeoForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model.eval()
        loaded_models[model_id] = model
        loaded_tokenizers[model_id] = tokenizer
    
    return loaded_models[model_id], loaded_tokenizers[model_id]

@app.route('/')
def home():
    return render_template('index.html', models=MODELS)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data or 'model' not in data:
            return jsonify({'error': 'Prompt and model selection required'}), 400

        prompt = data['prompt']
        model_id = data['model']
        
        if model_id not in MODELS:
            return jsonify({'error': 'Invalid model selection'}), 400

        model, tokenizer = load_model_and_tokenizer(model_id)
        config = MODELS[model_id]

        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        outputs = model.generate(
            inputs.input_ids,
            max_length=config['max_length'],
            temperature=config['temperature'],
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True
        )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({
            'response': response_text,
            'model': config['display_name'],
            'prompt': prompt
        })

    except Exception as e:
        logger.exception("Generation error")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "available_models": list(MODELS.keys())
    })

if __name__ == '__main__':
    # CPU Optimization settings
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    app.run(host='0.0.0.0', port=5000, debug=True)
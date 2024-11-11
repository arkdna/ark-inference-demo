from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoForCausalLM, GPT2Tokenizer
import torch
import logging

logger = logging.getLogger(__name__)

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

def setup_tokenizer(tokenizer, model_name):
    """Configure tokenizer based on model type"""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model_and_tokenizer(model_id):
    """Load and cache model and tokenizer"""
    logger.info(f"Checking cache for model: {model_id}")
    logger.info(f"Currently loaded models: {list(loaded_models.keys())}")
    
    if model_id in loaded_models:
        logger.info(f"Using cached model: {model_id}")
        return loaded_models[model_id], loaded_tokenizers[model_id]
    
    logger.info(f"Model {model_id} not in cache, loading from scratch...")
    config = MODELS[model_id]
    model_name = config['name']
    
    try:
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
        
        tokenizer = setup_tokenizer(tokenizer, model_name)
        model.eval()  # Set to evaluation mode
        
        # Cache the model and tokenizer
        loaded_models[model_id] = model
        loaded_tokenizers[model_id] = tokenizer
        logger.info(f"Successfully cached model: {model_id}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise

def generate_text(prompt, model_id="phi-2", max_length=None):
    """Generate text using the specified model"""
    try:
        if model_id not in MODELS:
            raise ValueError(f"Invalid model selection: {model_id}")
        
        config = MODELS[model_id]
        max_length = max_length or config['max_length']
        
        model, tokenizer = load_model_and_tokenizer(model_id)
        
        # Set model-specific tokenizer parameters
        if model_id == 'neural-chat':
            tokenizer.model_max_length = config['max_length']  # Use the configured max length
            
        # Create input with explicit parameters
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=max_length,  # Explicitly pass max_length to tokenizer
            return_attention_mask=True
        )
        
        # Model-specific generation settings
        if model_id == 'gpt-neo':
            attention_mask = (inputs.input_ids != tokenizer.eos_token_id).long()
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=config['temperature'],
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=config['temperature'],
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating text with {model_id}: {str(e)}")
        raise
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoForCausalLM, GPT2Tokenizer, BitsAndBytesConfig, AutoConfig
import torch
import logging
import torch.backends.cudnn as cudnn
from transformers import set_seed
import numpy as np
from typing import Iterator
import json

logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
    'gpt-neo': {
        'name': 'EleutherAI/gpt-neo-1.3B',
        'display_name': 'GPT-Neo 1.3B',
        'description': 'EleutherAI GPT-Neo model',
        'max_length': 256,
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 50,
        'num_beams': 4,
        'requires_padding': True,
        'model_type': 'gpt-neo'  # Add model type for special handling
    },
    'phi-2': {
        'name': 'microsoft/phi-2',
        'display_name': 'Microsoft Phi-2',
        'description': 'Compact and efficient 2.7B parameter model',
        'max_length': 256,
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 50,
        'num_beams': 4,
        'requires_padding': True
    },
    'neural-chat': {
        'name': 'Intel/neural-chat-7b-v3-1',
        'display_name': 'Intel Neural Chat 7B',
        'description': 'CPU-optimized conversational model',
        'max_length': 512,
        'temperature': 0.8,
        'top_p': 0.9,
        'top_k': 50,
        'num_beams': 4,
        'requires_padding': True
    }
}

# Global model cache
loaded_models = {}
loaded_tokenizers = {}

torch.set_num_threads(4)  # Limit CPU threads
torch.set_grad_enabled(False)  # Disable gradients
cudnn.benchmark = True  # Enable cudnn benchmarking

def setup_tokenizer(tokenizer, model_id):
    """Configure tokenizer based on model type"""
    logger.info(f"Setting up tokenizer for {model_id}")
    
    if model_id == 'gpt-neo':
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif model_id == 'phi-2':
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif model_id == 'neural-chat':
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Double check pad token is set
    if tokenizer.pad_token is None:
        logger.warning(f"Pad token still None for {model_id}, setting to eos_token as fallback")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"Tokenizer setup complete for {model_id}. Pad token: {tokenizer.pad_token}")
    return tokenizer

def load_model_and_tokenizer(model_id):
    """Load and cache model and tokenizer with optimizations"""
    if model_id not in loaded_models:
        config = MODELS[model_id]
        model_name = config['name']
        
        try:
            # Special handling for GPT-Neo
            if config.get('model_type') == 'gpt-neo':
                from transformers import GPTNeoForCausalLM, GPT2Tokenizer
                
                tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                # GPT-Neo specific: Set pad token to eos token
                tokenizer.pad_token = tokenizer.eos_token
                
                model = GPTNeoForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                model.config.pad_token_id = model.config.eos_token_id
                
            else:
                # Regular handling for other models
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    device_map='auto'
                )
            
            if hasattr(torch, 'compile'):
                model = torch.compile(model)
            model.eval()
            
            loaded_models[model_id] = model
            loaded_tokenizers[model_id] = tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    return loaded_models[model_id], loaded_tokenizers[model_id]

def generate_text(prompt, model_id="phi-2", max_length=None):
    """Generate text without streaming"""
    try:
        model, tokenizer = load_model_and_tokenizer(model_id)
        config = MODELS[model_id]
        max_length = max_length or config.get('max_length', 256)
        
        # Create attention mask for the input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True
        )
        
        # For GPT-Neo, ensure attention mask is properly set
        if config.get('model_type') == 'gpt-neo':
            attention_mask = torch.ones_like(inputs.input_ids)
            inputs['attention_mask'] = attention_mask
        
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                temperature=config.get('temperature', 0.7),
                top_p=config.get('top_p', 0.9),
                top_k=config.get('top_k', 50),
                num_beams=config.get('num_beams', 4),
                no_repeat_ngram_size=3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        # Decode only the new tokens
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating text with {model_id}: {str(e)}")
        raise
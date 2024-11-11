from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoForCausalLM, GPT2Tokenizer
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
            if model_id == 'gpt-neo':
                model = GPTNeoForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
                tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Configure tokenizer
            tokenizer = setup_tokenizer(tokenizer, model_id)
            
            # Set model to eval mode
            model.eval()
            
            # Update model config to match tokenizer
            model.config.pad_token_id = tokenizer.pad_token_id
            
            # Cache in memory
            loaded_models[model_id] = model
            loaded_tokenizers[model_id] = tokenizer
            
            logger.info(f"Successfully loaded and configured {model_id}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    return loaded_models[model_id], loaded_tokenizers[model_id]

def generate_text(prompt, model_id="phi-2", max_length=None):
    """Generate text without streaming (for compatibility)"""
    try:
        config = MODELS[model_id]
        max_length = max_length or config['max_length']
        
        model, tokenizer = load_model_and_tokenizer(model_id)
        
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=max_length,
            return_attention_mask=True
        )
        
        with torch.inference_mode():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=config['temperature'],
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating text with {model_id}: {str(e)}")
        raise

def generate_stream(prompt, model_id="phi-2", max_length=None) -> Iterator[str]:
    """Generate text with streaming"""
    try:
        config = MODELS[model_id]
        max_length = max_length or config['max_length']
        
        model, tokenizer = load_model_and_tokenizer(model_id)
        
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=max_length,
            return_attention_mask=True
        )
        
        input_length = len(inputs.input_ids[0])
        
        with torch.inference_mode():
            # Different generation settings for GPT-Neo
            if model_id == 'gpt-neo':
                attention_mask = (inputs.input_ids != tokenizer.eos_token_id).long()
                generation_kwargs = {
                    'attention_mask': attention_mask,
                    'pad_token_id': tokenizer.eos_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                }
            else:
                generation_kwargs = {
                    'pad_token_id': tokenizer.pad_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                }
            
            generation_kwargs.update({
                'max_length': max_length,
                'temperature': config['temperature'],
                'num_return_sequences': 1,
                'no_repeat_ngram_size': 2,
                'do_sample': True,
                'use_cache': True,
                'num_beams': 1,
                'return_dict_in_generate': True,
            })
            
            outputs = model.generate(inputs.input_ids, **generation_kwargs)
            
            # Stream the output token by token
            generated_sequence = outputs.sequences[0][input_length:]
            for i in range(len(generated_sequence)):
                token = generated_sequence[i:i+1]
                text = tokenizer.decode(token, skip_special_tokens=True)
                if text:
                    data = json.dumps({"token": text})
                    yield f"data: {data}\n\n"
                    
    except Exception as e:
        logger.error(f"Error generating text with {model_id}: {str(e)}")
        error_data = json.dumps({"error": str(e)})
        yield f"data: {error_data}\n\n"
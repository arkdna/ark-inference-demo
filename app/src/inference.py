from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoForCausalLM, GPT2Tokenizer
import torch
import logging

logger = logging.getLogger(__name__)

# Initialize storage for loaded models and tokenizers
loaded_models = {}
loaded_tokenizers = {}

MODELS = {
    'gpt-neo': {
        'name': 'EleutherAI/gpt-neo-1.3B',
        'display_name': 'GPT-Neo 1.3B',
        'description': 'EleutherAI GPT-Neo model',
        'max_length': 256,
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 50,
        'model_type': 'gpt-neo'
    },
    'phi-2': {
        'name': 'microsoft/phi-2',
        'display_name': 'Microsoft Phi-2',
        'description': 'Compact and efficient 2.7B parameter model',
        'max_length': 256,
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 50,
        'model_type': 'phi-2'
    },
    'neural-chat': {
        'name': 'Intel/neural-chat-7b-v3-1',
        'display_name': 'Intel Neural Chat 7B',
        'description': 'CPU-optimized conversational model',
        'max_length': 512,
        'temperature': 0.8,
        'top_p': 0.9,
        'top_k': 50,
        'model_type': 'neural-chat'
    }
}

def load_model_and_tokenizer(model_id):
    if model_id not in loaded_models:
        config = MODELS[model_id]
        model_name = config['name']
        
        try:
            logger.info(f"Loading model and tokenizer for {model_name}")
            
            # Load tokenizer first
            if config.get('model_type') == 'gpt-neo':
                tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            if config.get('model_type') == 'gpt-neo':
                model = GPTNeoForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    device_map='auto'
                )
            
            # Set pad token ID in model config
            model.config.pad_token_id = tokenizer.pad_token_id
            
            if hasattr(torch, 'compile'):
                model = torch.compile(model)
            model.eval()
            
            # Store both model and tokenizer
            loaded_models[model_id] = model
            loaded_tokenizers[model_id] = tokenizer
            
            logger.info(f"Successfully loaded model and tokenizer for {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    return loaded_models[model_id], loaded_tokenizers[model_id]

def generate_text(prompt, model_id="phi-2", max_length=None):
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
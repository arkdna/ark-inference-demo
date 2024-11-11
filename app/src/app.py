from flask import Flask, request, jsonify, render_template
import logging
import os
import torch
import psutil
import platform
from inference import generate_text, MODELS, loaded_models

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set up the Flask app with explicit template folder
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
app = Flask(__name__, 
    template_folder=template_dir,
    static_folder=static_dir)

# Add startup logging
logger.info("Starting application...")
logger.info(f"Template directory: {template_dir}")
logger.info(f"Static directory: {static_dir}")

@app.before_first_request
def before_first_request():
    logger.info("Initializing before first request...")
    # Any initialization code here

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
        max_length = data.get('max_length', None)
        
        if model_id not in MODELS:
            return jsonify({'error': 'Invalid model selection'}), 400

        response_text = generate_text(
            prompt=prompt,
            model_id=model_id,
            max_length=max_length
        )
        
        return jsonify({
            'response': response_text,
            'model': MODELS[model_id]['display_name'],
            'prompt': prompt
        })

    except Exception as e:
        logger.exception("Generation error")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    # Get CPU information
    cpu_count = psutil.cpu_count()
    cpu_physical = psutil.cpu_count(logical=False)
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Get memory information
    memory = psutil.virtual_memory()
    memory_total_gb = memory.total / (1024 ** 3)  # Convert to GB
    memory_used_gb = memory.used / (1024 ** 3)
    memory_percent = memory.percent
    
    # Get loaded models information
    loaded_model_info = {
        model_id: {
            'name': MODELS[model_id]['display_name'],
            'parameters': '1.3B' if model_id == 'gpt-neo' else '2.7B' if model_id == 'phi-2' else '7B'
        }
        for model_id in loaded_models.keys()
    }
    
    return jsonify({
        "status": "healthy",
        "system_info": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__
        },
        "cpu": {
            "total_cores": cpu_count,
            "physical_cores": cpu_physical,
            "usage_percent": cpu_percent,
            "torch_threads": torch.get_num_threads()
        },
        "memory": {
            "total_gb": round(memory_total_gb, 2),
            "used_gb": round(memory_used_gb, 2),
            "usage_percent": memory_percent
        },
        "models": {
            "loaded": loaded_model_info,
            "available": MODELS
        },
        "optimizations": {
            "torch_grad_enabled": torch.is_grad_enabled(),
            "device": "cpu",
            "low_memory_mode": True
        }
    })

# Add a new route for detailed stats
@app.route('/stats')
def stats():
    # Get process-specific information
    process = psutil.Process()
    
    return jsonify({
        "process": {
            "cpu_percent": process.cpu_percent(),
            "memory_percent": process.memory_percent(),
            "threads": process.num_threads(),
            "open_files": len(process.open_files()),
            "memory_maps": len(process.memory_maps())
        },
        "disk": {
            "usage": psutil.disk_usage('/').percent,
            "io_counters": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else None
        },
        "network": {
            "connections": len(psutil.net_connections()),
            "io_counters": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else None
        }
    })

if __name__ == '__main__':
    # CPU Optimization settings
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    app.run(host='0.0.0.0', port=5000, debug=True)
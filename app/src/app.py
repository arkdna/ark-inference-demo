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

# Add startup logging
logger.info("Starting application...")
logger.info(f"Template directory: {template_dir}")
logger.info(f"Static directory: {static_dir}")

# CPU Configuration
num_physical_cores = psutil.cpu_count(logical=False)  # Physical cores
num_logical_cores = psutil.cpu_count(logical=True)    # Logical cores (including hyperthreading)

# Optimize thread settings for your 40 CPU setup
os.environ['OMP_NUM_THREADS'] = str(num_physical_cores)  # OpenMP threads
os.environ['MKL_NUM_THREADS'] = str(num_physical_cores)  # MKL threads
torch.set_num_threads(num_physical_cores)                # PyTorch threads
torch.set_num_interop_threads(min(4, num_physical_cores))  # Inter-op parallelism

# Log CPU information
logger.info(f"Number of CPU cores: Physical={num_physical_cores}, Logical={num_logical_cores}")
logger.info(f"PyTorch threads: {torch.get_num_threads()}")

app = Flask(__name__, 
    template_folder=template_dir,
    static_folder=static_dir)

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
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    memory = psutil.virtual_memory()
    
    return jsonify({
        'status': 'healthy',
        'cpu': {
            'total_cores': psutil.cpu_count(),
            'usage_per_core': cpu_percent,
            'average_usage': sum(cpu_percent) / len(cpu_percent)
        },
        'memory': {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent
        },
        'torch_threads': torch.get_num_threads(),
        'model_cache_size': len(loaded_models)
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
    app.run(host='0.0.0.0', port=5000, debug=True)
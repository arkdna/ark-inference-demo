from flask import Flask, request, jsonify, render_template
import logging
import os
import torch
from inference import generate_text, MODELS

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set up the Flask app with explicit template folder
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
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
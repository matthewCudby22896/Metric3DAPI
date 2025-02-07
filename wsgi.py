from flask import Flask, request, jsonify, send_file
from markupsafe import escape
from metric3d_inference import metric3d_inference_generator
import numpy as np
import cv2
import logging
import time
from numpy.typing import NDArray
import io

# Basic logger definition
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()  # Logs to console
    ]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_OPTIONS = {
    'small' : 'metric3d_vit_small',
    'large' : 'metric3d_vit_large',
    'giant' : 'metric3d_vit_giant2',
}

generator = metric3d_inference_generator()

@app.route("/inference/<string:version>", methods=['POST'])
def run_inference(version: str):
    """
    flask --app wsgi run --host=0.0.0.0
    gunicorn --bind 0.0.0.0:5000 wsgi:app
    """
    if version not in MODEL_OPTIONS:
        return f"version={escape(version)} is not one of the available options {list(MODEL_OPTIONS.keys())}", 400
    
    if 'image' not in request.files:
        return jsonify({'error' : 'No image in the request'})

    image_bytes = request.files['image'].read() # byte file
    
    npimg = np.frombuffer(image_bytes, np.uint8) # convert bytes into a numpy array
    
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR) # converts into format that opencv can process
    
    # Run inference (Metric3D)
    start = time.time()
    depth_estimation : NDArray[np.float32] = generator.estimate_depth(org_rgb=img, version=version)
    time_elapsed = time.time() - start
    logger.info(f"Metric3D version='{version}' inference took {time_elapsed} seconds")
    
    # Create an in-memory buffer
    buffer = io.BytesIO()
    
    # Load the np array into the buffer
    np.save(buffer, depth_estimation)
    # Move stream position to the start of the buffer s.t. it can be read
    buffer.seek(0)
    
    return send_file(
        buffer, 
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name="depth_map.npy"
        )
    
    



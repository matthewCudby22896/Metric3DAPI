from flask import Flask, request, jsonify, Response
from markupsafe import escape
from metric3d_inference import metric3d_inference_generator
import numpy as np
import cv2
import logging
import time

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
    flask --app wsgi run --host-0.0.0.0
    
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
    depth_estimation : np.ndarray = generator.estimate_depth(org_rgb=img, version=version)
    time_elapsed = time.time() - start
    logger.info(f"Metric3D version='{version}' inference took {time_elapsed} seconds")
    
    # Normalise
    depth_estimation = ((depth_estimation / np.max(depth_estimation)) * 255).astype(np.uint8)
    
    # Encode numpy array as a png image
    _, buffer = cv2.imencode('.png', depth_estimation)
    image_bytes = buffer.tobytes()
    



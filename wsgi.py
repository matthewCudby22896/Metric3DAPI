from flask import Flask, request, jsonify, send_file
from markupsafe import escape
import io
import metric3d_inference
import numpy as np
import cv2
import logging
from metric3d_inference import Image

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

@app.route("/inference/<string:version>", methods=['POST'])
def run_inference(version: str):
    """
    flask --app wsgi run --host=0.0.0.0
    gunicorn --bind 0.0.0.0:5000 wsgi:app
    """
    
    if version not in MODEL_OPTIONS:
        return f"version={escape(version)} is not one of the available options {list(MODEL_OPTIONS.keys())}", 400
    
    if 'image' not in request.files:
        return jsonify({'error' : 'No image in the request'}), 400
    
    # Extract the request body
    json_data : dict = request.form.to_dict()
    
    # Attempt to extract focal length from the request body
    focal_length = json_data.get('focal_length', None)
    if focal_length is None:
        return jsonify({'error' : 'No focal_length in request body'}), 400
    
    logger.info(f"{focal_length=}")
    logger.info(f"{version=}")
    
    # Access and decode the sent image intoan OpenCV format
    image_bytes = request.files['image'].read() # byte file
    npimg = np.frombuffer(image_bytes, np.uint8) # convert bytes into a numpy array
    img : Image = cv2.imdecode(npimg, cv2.IMREAD_COLOR) # converts into format that opencv can process

    depth_map = metric3d_inference.estimate_depth(version, org_rgb=img, focal_length_px=focal_length)
    
    # Save depth map to a binary buffer
    buffer = io.BytesIO()
    np.save(buffer, depth_map)
    buffer.seek(0)  # Move to the beginning of the buffer

    return send_file(buffer, as_attachment=True, download_name='depth_map.npy', mimetype='application/octet-stream')

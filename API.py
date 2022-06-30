from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import numpy as np
import cv2
from recognize import recognize_face
import os

app = Flask(__name__)
#pipreqs --encoding utf-8 "./" --force
@app.route('/', methods=['POST'])
def predict():
    payload = request.get_json()
    faces = []
    for i in range(len(payload['images'])):
        im_b64 = payload['images'][i]
        im_binary = base64.b64decode(im_b64)
        buf = io.BytesIO(im_binary)
        img = Image.open(buf)
        faces.append(np.asarray(img))

    captured_image_b64 = payload['image']
    captured_image_binary = base64.b64decode(captured_image_b64)
    buf = io.BytesIO(captured_image_binary)
    captured_image = cv2.cvtColor(np.array(Image.open(buf)), cv2.COLOR_RGB2BGR)
    dict = recognize_face(captured_image, faces)
    id = None
    if (dict['identity'] != None):
        id = payload['id'][dict['identity']]
    return jsonify({'status' :str(dict['status']), 'id':str(id)})

app.run(host='0.0.0.0', port='5000')
    

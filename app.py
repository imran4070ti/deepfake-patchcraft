from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import cv2
import os
import numpy as np
import random
import json
from uuid import uuid4
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from dotenv import load_dotenv
from preprocess import preprocess, extract_frames, load_model
from postprocess import majority_vote, get_output

# Load environment variables from the '.dev' file
load_dotenv('.dev')

# Import your model and preprocessing functions
from model import RichPoorTextureContrastModel
from preprocessing.patch_generator import smash_n_reconstruct
import preprocessing.filters as f

app = Flask(__name__)

# -----------------------------------------------------------------------------
# JWT CONFIGURATION
# -----------------------------------------------------------------------------
# Set the secret key from an environment variable for improved security.
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')  # Ensure this is set in your .dev file
jwt = JWTManager(app)

# -----------------------------------------------------------------------------
# AUTHENTICATION ENDPOINTS
# -----------------------------------------------------------------------------

@app.route('/deepfake/register', methods=['POST'])
def register():
    """
    Register a new user.
    Expects JSON payload:
      {
         "username": "example",
         "password": "password123"
      }
    The password is hashed before saving, and user data is stored in 'users.json'.
    """
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'msg': 'Username and password are required.'}), 400

    username = data['username']
    password = data['password']

    users_file = 'users.json'
    users = []
    # Load existing users if the file exists
    if os.path.exists(users_file):
        with open(users_file, 'r') as f:
            try:
                users = json.load(f)
            except json.JSONDecodeError:
                users = []

    # Check if the username already exists
    if any(user['username'] == username for user in users):
        return jsonify({'msg': 'User already exists.'}), 400

    # Hash the password for secure storage
    hashed_password = generate_password_hash(password)
    new_user = {
        'id': str(uuid4()),
        'username': username,
        'password': hashed_password
    }
    users.append(new_user)

    # Save the updated user list back to the JSON file
    with open(users_file, 'w') as f:
        json.dump(users, f)

    return jsonify({'msg': 'User registered successfully.'}), 200

@app.route('/deepfake/login', methods=['POST'])
def login():
    """
    Login a user.
    Expects JSON payload:
      {
         "username": "example",
         "password": "password123"
      }
    If credentials are valid, returns a JWT access token.
    """
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'msg': 'Username and password are required.'}), 400

    username = data['username']
    password = data['password']

    users_file = 'users.json'
    if not os.path.exists(users_file):
        return jsonify({'msg': 'No users registered.'}), 400

    with open(users_file, 'r') as f:
        try:
            users = json.load(f)
        except json.JSONDecodeError:
            return jsonify({'msg': 'Error reading users file.'}), 500

    # Locate the user with the matching username
    user = next((u for u in users if u['username'] == username), None)
    if user is None or not check_password_hash(user['password'], password):
        return jsonify({'msg': 'Invalid username or password.'}), 401

    # Create a JWT access token using the user's ID as the identity
    access_token = create_access_token(identity=user['id'])
    return jsonify({'access_token': access_token}), 200

# -----------------------------------------------------------------------------
# PROTECTED DEEPFAKE PREDICTION ENDPOINT
# -----------------------------------------------------------------------------

@app.route('/deepfake/predict', methods=['POST'])
@jwt_required()  # This decorator enforces that a valid JWT must be sent with the request.
def predict():
    """
    DeepFake detection endpoint.
    This endpoint now requires a valid JWT token.
    Accepts either an image or video file via the 'file' form-data field.
    """
    # Optionally, get the current user identity (e.g., the user id) from the token
    current_user = get_jwt_identity()
    print("Current User ID:", current_user)

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    filename = file.filename

    # Ensure the uploads directory exists
    uploads_dir = 'uploads'
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    # Load your pre-trained deepfake detection model
    device = 'cpu'
    model = load_model(device=device)

    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Handle image input
        image = Image.open(file).convert('RGB')
        rti, pti = preprocess(image)

        label = get_output(model, rti, pti, device)
        output = 'Fake' if label == 1 else 'Real'

        return jsonify({'result': output})

    elif filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Handle video input
        video_path = os.path.join(uploads_dir, filename)
        file.save(video_path)

        try:
            frames = extract_frames(video_path)
        except ValueError as e:
            os.remove(video_path)
            return jsonify({'error': str(e)}), 400

        predictions = []
        for frame in frames:
            rti, pti = preprocess(frame)
            label = get_output(model, rti, pti, device)
            predictions.append(label)

        os.remove(video_path)  # Clean up the saved video file
        print("Frame predictions:", predictions)

        final_label = majority_vote(predictions)
        output = 'Fake' if final_label == 1 else 'Real'

        return jsonify({'result': output})
    else:
        return jsonify({'error': 'Unsupported file format.'}), 400

# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)

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
from flask_cors import CORS
from datetime import timedelta
import re
from authentication_validation import is_valid_email, is_strong_password

# Load environment variables from the .env file
load_dotenv('.env')

# Import your model and preprocessing functions
from model import RichPoorTextureContrastModel
from preprocessing.patch_generator import smash_n_reconstruct
import preprocessing.filters as f

app = Flask(__name__)
CORS(app)

# -----------------------------------------------------------------------------
# JWT CONFIGURATION
# -----------------------------------------------------------------------------
# Set the secret key from an environment variable for improved security.
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')  # Ensure this is set in your .env file
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24) #minutes=1440
jwt = JWTManager(app)

# -----------------------------------------------------------------------------
# AUTHENTICATION ENDPOINTS
# -----------------------------------------------------------------------------

@app.route('/register', methods=['POST'])
def register():
    """
    Register a new user.
    Expects JSON payload:
      {
         "username": "example",
         "email": "user@example.com",
         "password": "Password@123",
         "confirm_password": "Password@123"
      }
    """
    data = request.get_json()

    # Validate required fields
    if not data or not all(key in data for key in ['username', 'email', 'password', 'confirm_password']):
        return jsonify({'msg': 'Username, email, password, and confirmation are required.'}), 400

    username = data['username'].strip()
    email = data['email'].strip().lower()
    password = data['password']
    confirm_password = data['confirm_password']

    # Validate email format
    if not is_valid_email(email):
        return jsonify({'msg': 'Invalid email format. Use example@email.com'}), 400

    # Check password strength
    if not is_strong_password(password):
        return jsonify({'msg': 'Password must be at least 8 characters, include an uppercase, lowercase, number, special character, and have no spaces.'}), 400

    # Confirm passwords match
    if password != confirm_password:
        return jsonify({'msg': 'Passwords do not match.'}), 400

    users_file = 'users.json'
    users = []

    # Load existing users if the file exists
    if os.path.exists(users_file):
        with open(users_file, 'r') as f:
            try:
                users = json.load(f)
            except json.JSONDecodeError:
                users = []

    # Ensure username and email are unique
    if any(user['username'] == username for user in users):
        return jsonify({'msg': 'Username already exists.'}), 400
    if any(user['email'] == email for user in users):
        return jsonify({'msg': 'Email already registered.'}), 400

    # Hash the password for secure storage
    hashed_password = generate_password_hash(password)
    new_user = {
        'id': str(uuid4()),
        'username': username,
        'email': email,
        'password': hashed_password
    }
    users.append(new_user)

    # Save the updated user list back to the JSON file
    with open(users_file, 'w') as f:
        json.dump(users, f)

    return jsonify({'msg': 'User registered successfully.'}), 200

@app.route('/login', methods=['POST'])
def login():
    """
    Login a user using either username or email.
    Expects JSON payload:
      {
         "id": "example OR user@example.com",
         "password": "Password@123"
      }
    """
    data = request.get_json()

    if not data or not data.get('user_id') or not data.get('password'):
        return jsonify({'msg': 'Username/Email and password are required.'}), 400

    identifier = data['user_id'].strip()  # Can be username or email
    password = data['password'].strip()

    # If the identifier looks like an email, validate format
    if '@' in identifier and not is_valid_email(identifier):
        return jsonify({'msg': 'Invalid email format. Use example@email.com'}), 400

    users_file = 'users.json'
    if not os.path.exists(users_file):
        return jsonify({'msg': 'Invalid username/email or password.'}), 401

    with open(users_file, 'r') as f:
        try:
            users = json.load(f)
        except json.JSONDecodeError:
            return jsonify({'msg': 'Error reading users file.'}), 500

    # Locate the user using either username or email
    user = next((u for u in users if u['username'].lower() == identifier or u['email'] == identifier), None)

    # Check if user exists and password is correct
    if user is None or not check_password_hash(user['password'], password):
        return jsonify({'msg': 'Invalid username/email or password.'}), 401  # Generic error message

    # Create a JWT access token using the user's ID as the identity
    access_token = create_access_token(identity=user['id'])
    return jsonify({'access_token': access_token}), 200

# -----------------------------------------------------------------------------
# PROTECTED DEEPFAKE PREDICTION ENDPOINT
# -----------------------------------------------------------------------------

@app.route('/predict', methods=['POST'])
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
            frames = extract_frames(video_path, num_frames=3)
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
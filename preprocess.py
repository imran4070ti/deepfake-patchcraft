import torch
from torchvision import transforms
from PIL import Image
import cv2
import os
import numpy as np
import random
from model import RichPoorTextureContrastModel
from preprocessing.patch_generator import smash_n_reconstruct
import preprocessing.filters as f

# Define the preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Helper function to preprocess an image
def preprocess(image):
    rt, pt = smash_n_reconstruct(image)
    frt = f.apply_all_filters(rt)
    fpt = f.apply_all_filters(pt)
    # Ensure tensors are in float32
    frt = torch.tensor(frt, dtype=torch.float32).unsqueeze(-1)
    fpt = torch.tensor(fpt, dtype=torch.float32).unsqueeze(-1)
    return frt, fpt


# Helper function to extract frames from video
def extract_frames(video_path, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        raise ValueError("Video has fewer frames than the required number of frames.")

    frame_indices = sorted(random.sample(range(total_frames), num_frames))
    frames = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)

    cap.release()
    return frames

# Load the model
def load_model(device = 'cpu'):
    checkpoint_path = 'models_cnnspot_10p_data'
    model = RichPoorTextureContrastModel()
    state_dict = torch.load(os.path.join(checkpoint_path, 'best.pt'), map_location=device)['state_dict']
    model.load_state_dict(state_dict)
    return model


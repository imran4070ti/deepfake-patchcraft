import torch
import numpy as np

# Helper function for majority voting
def majority_vote(predictions):
    return max(set(predictions), key=predictions.count)


def get_output(model, rich_texture, poor_texture, device):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        
        # Convert inputs to tensors and move to the specified device
        if isinstance(rich_texture, np.ndarray):
            rich_texture = torch.tensor(rich_texture).permute(2, 0, 1).unsqueeze(0).float().to(device)
        else:
            rich_texture = rich_texture.permute(2, 0, 1).unsqueeze(0).to(device)
        
        if isinstance(poor_texture, np.ndarray):
            poor_texture = torch.tensor(poor_texture).permute(2, 0, 1).unsqueeze(0).float().to(device)
        else:
            poor_texture = poor_texture.permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Forward pass through the model
        probability = model(rich_texture, poor_texture)
        probability = probability.item()  # Convert tensor to scalar
        
        # Threshold to determine label
        label = 1 if probability >= 0.5 else 0
        return label
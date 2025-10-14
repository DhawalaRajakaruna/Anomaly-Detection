import torch
from model import CrossConvolutions, parallel_glow_coupling_layer
import freia_funcs  # make sure this imports ReversibleGraphNet


def load_model(checkpoint_path):
    """
    Safely load the anomaly detection model from the given path.
    Returns the model in evaluation mode on CPU.
    """
    try:
        with torch.serialization.safe_globals([
            freia_funcs.ReversibleGraphNet,
            CrossConvolutions,
            parallel_glow_coupling_layer
        ]):
            model = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.eval()
            print("✅ Model loaded successfully on CPU!")
            return model
    except Exception as e:
        print("❌ Error loading model:", e)
        return None


def print_model_summary(model):
    if model is None:
        print("❌ Model is None, cannot print details.")
        return

    print("=== Model Structure ===")
    print(model)  # prints the full model architecture

    print("\n=== Named Parameters ===")
    for name, param in model.named_parameters():
        print(f"{name} | shape: {param.shape} | requires_grad: {param.requires_grad}")

    print("\n=== Total Parameters ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

def compute_anomaly_score(outputs):
    # outputs: list of z_tensors for each scale
    scores = []
    for z in outputs:
        # Compute per-sample average magnitude over all channels and spatial locations
        s = torch.mean(z**2, dim=(1, 2, 3))
        scores.append(s)

    # Combine scales (sum or mean)
    total_score = torch.mean(torch.stack(scores, dim=1), dim=1)
    return total_score.numpy()


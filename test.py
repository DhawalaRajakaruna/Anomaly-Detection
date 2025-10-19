import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import config as c
from model import load_model, FeatureExtractor
from utils import t2np, concat_maps
import os
import time
from datetime import datetime
import psutil

# Configuration
image_path = "data/images/bottle/test/broken_large/004.png"
model_name = c.modelname  # Using model name from config

print(f"Loading image: {image_path}")

# Check if image exists
if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    print("Please ensure the image exists in the specified path.")
    exit()

# Load and preprocess the image
image = Image.open(image_path).convert('RGB')
original_image = np.array(image)

# Define transformations (same as training)
transform = transforms.Compose([
    transforms.Resize(c.img_size),
    transforms.ToTensor(),
    transforms.Normalize(c.norm_mean, c.norm_std)
])

# Transform image for model input
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
print(f"Image shape after preprocessing: {input_tensor.shape}")

# Load the feature extractor
print("Loading feature extractor...")
feature_extractor = FeatureExtractor()
feature_extractor.to(c.device)
feature_extractor.eval()

# Extract features from the image
print("Extracting features from the image...")
with torch.no_grad():
    input_tensor = input_tensor.to(c.device)
    features = feature_extractor(input_tensor)
    print(f"Number of feature scales: {len(features)}")
    for i, feat in enumerate(features):
        print(f"  Scale {i} feature shape: {feat.shape}")

# Load the trained model
print(f"Loading trained model: {model_name}")
try:
    model = load_model(model_name)
    model.to(c.device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model has been trained and saved.")
    exit()

# Get baseline memory after model is loaded
process = psutil.Process()
baseline_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB

# Run features through the model
print("Running features through the model...")
start_time = time.time()
start_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB

with torch.no_grad():
    z = model(features)
    
    # Calculate anomaly score
    z_concat = t2np(concat_maps(z))
    anomaly_score = np.mean(z_concat ** 2 / 2, axis=(1, 2))
    print(f"Anomaly score: {anomaly_score[0]:.4f}")
    
    # Generate anomaly map (localization)
    # Use the first scale for visualization
    z_grouped = list()
    for i in range(len(z)):
        z_grouped.append(z[i].view(-1, *z[i].shape[1:]))
    
    # Calculate likelihood for the first scale
    likelihood_map = torch.mean(z_grouped[0] ** 2, dim=(1,))
    
    # Convert to numpy
    anomaly_map = t2np(likelihood_map)[0]
    
    # Resize to original image dimensions
    # Convert back to torch tensor for interpolation
    likelihood_tensor = torch.from_numpy(anomaly_map).unsqueeze(0).unsqueeze(0).float()
    anomaly_map_resized = t2np(
        F.interpolate(
            likelihood_tensor, 
            size=original_image.shape[:2], 
            mode='bilinear', 
            align_corners=False
        )
    )[0, 0]

inference_time = time.time() - start_time
end_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
memory_used = end_memory - start_memory  # Memory used only for this image processing
current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Visualization
print("Generating visualizations...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Add title with inference time, memory usage for image processing and date at the top of the figure
info_text = f'Inference Time: {inference_time*1000:.2f}ms | Memory Used: {memory_used:.2f}MB | Date: {current_datetime} | Model: {model_name}'
fig.suptitle(info_text, fontsize=12, fontweight='bold', y=0.98)

# Original image
axes[0].imshow(original_image)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Anomaly map
im1 = axes[1].imshow(anomaly_map_resized, cmap='jet')
axes[1].set_title(f'Anomaly Map\n(Score: {anomaly_score[0]:.4f})')
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# Overlay
axes[2].imshow(original_image)
axes[2].imshow(anomaly_map_resized, cmap='jet', alpha=0.5)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save visualization
output_dir = './viz/test_output/'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'anomaly_visualization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

# Show the plot
plt.show()

print("\nAnalysis complete!")
print(f"Image: {image_path}")
print(f"Anomaly Score: {anomaly_score[0]:.4f}")
print(f"Inference Time: {inference_time*1000:.2f}ms")
print(f"Memory Used for Image Processing: {memory_used:.2f}MB")
print(f"Date/Time: {current_datetime}")
print(f"Higher scores indicate higher probability of anomaly.")

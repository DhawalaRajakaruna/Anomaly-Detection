import time
import torch
from tqdm import tqdm
from model import load_model
from utils import make_dataloaders, load_datasets
import config as c
import torch.nn.functional as F

# === Device selection ===
device = c.device if torch.cuda.is_available() else 'cpu'
# === Load dataset ===
print("Loading dataset and dataloader...")
train_set, test_set = load_datasets(c.dataset_path, c.class_name)
print(len(test_set))
_, test_loader = make_dataloaders(train_set, test_set)
print(f"Test set size: {len(test_set)}")

# === Load CS-Flow model ===
print(f"Loading CS-Flow model: {c.modelname} ...")
model = load_model(c.modelname)
model.to(device)
model.eval()
print(" Model loaded successfully!")

# === Warm-up (for fair timing) ===
print("Warming up the model...")
with torch.no_grad():
    for i, data in enumerate(test_loader):
        # Safely unpack input
        if isinstance(data, (list, tuple)):
            inputs, _ = data
        else:
            inputs = data

        if isinstance(inputs, list):
            inputs = inputs[0]

        inputs = inputs.to(device)

        # Prepare multi-scale input list
        multi_scale_inputs = [inputs]
        for s in range(1, c.n_scales):
            scaled = F.interpolate(inputs, scale_factor=1/(2**s), mode='bilinear', align_corners=False)
            multi_scale_inputs.append(scaled)

        _ = model(multi_scale_inputs)
        break  # warm-up done

# === Measure runtime and perform inference ===
print("\nMeasuring inference time on test set...")
total_time = 0.0
num_images = 0
anomaly_scores = []

with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader)):
        # Safely unpack input
        if isinstance(data, (list, tuple)):
            inputs, labels = data
        else:
            inputs = data
            labels = torch.zeros(inputs.shape[0])  # dummy labels

        if isinstance(inputs, list):
            inputs = inputs[0]

        inputs = inputs.to(device)

        # Prepare multi-scale input list
        multi_scale_inputs = [inputs]
        for s in range(1, c.n_scales):
            scaled = F.interpolate(inputs, scale_factor=1/(2**s), mode='bilinear', align_corners=False)
            multi_scale_inputs.append(scaled)

        start_time = time.time()
        outputs = model(multi_scale_inputs)
        end_time = time.time()

        batch_time = end_time - start_time
        total_time += batch_time
        num_images += inputs.shape[0]

        # Compute anomaly score (mean over scales)
        if isinstance(outputs, list):
            batch_scores = [out.mean(dim=[1, 2, 3]) for out in outputs]
            batch_scores = torch.stack(batch_scores, dim=1).mean(dim=1)
        else:
            batch_scores = outputs.mean(dim=[1, 2, 3])
        anomaly_scores.append(batch_scores.cpu())

# Concatenate all anomaly scores
anomaly_scores = torch.cat(anomaly_scores, dim=0)

# === Runtime summary ===
avg_time_per_image = total_time / num_images
fps = 1.0 / avg_time_per_image

print(f"\n--- Inference Summary ---")
print(f"Total images tested: {num_images}")
print(f"Total inference time: {total_time:.3f} seconds")
print(f"Average time per image: {avg_time_per_image * 1000:.3f} ms")
print(f"Inference speed: {fps:.2f} FPS")
print(f"Device used: {device}")

# Optional: GPU memory usage
if torch.cuda.is_available():
    mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
    mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    print(f"GPU Memory Allocated: {mem_alloc:.2f} MB")
    print(f"GPU Memory Reserved: {mem_reserved:.2f} MB")

# === Optional: Display anomaly detection results ===
print("\n--- Sample Anomaly Scores ---")
print(anomaly_scores[:10])

print("\n End-to-end inference with multi-scale anomaly detection completed successfully!")

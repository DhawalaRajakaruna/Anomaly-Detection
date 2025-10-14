import loadmodel 

import numpy as np
import torch

checkpoint_path = r"D:\UNIVERSITY OF MORATUWA\Semester-5\Anomaly-Detection\models\tmp\bottle_test"


# -------------------------
# Load model
# -------------------------
model = loadmodel.load_model(checkpoint_path)
#loadmodel.print_model_summary(model)

export_dir = "data/features/bottle/"

# Load all test feature scales
f0 = np.load(export_dir + "bottle_scale_0_test.npy")
f1 = np.load(export_dir + "bottle_scale_1_test.npy")
f2 = np.load(export_dir + "bottle_scale_2_test.npy")

# Convert to torch tensors
f0 = torch.tensor(f0, dtype=torch.float32)
f1 = torch.tensor(f1, dtype=torch.float32)
f2 = torch.tensor(f2, dtype=torch.float32)

test_features = [f0, f1, f2]

print([f.shape for f in test_features])

model.eval()
with torch.no_grad():
    outputs = model(test_features)

print(type(outputs))
if isinstance(outputs, (list, tuple)):
    for i, o in enumerate(outputs):
        print(f"Output {i} shape: {o.shape}")
else:
    print("Output shape:", outputs.shape)

with torch.no_grad():
    anomaly_scores = loadmodel.compute_anomaly_score(outputs)

print("Anomaly scores shape:", anomaly_scores.shape)
print("Example scores:", anomaly_scores[:10])

# Example pseudo-code (depends on your model definition)
# z, log_jac_det = model(test_features, rev=False)
# anomaly_score = torch.mean(z**2, dim=1) - log_jac_det
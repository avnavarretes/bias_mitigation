import torch
import diffusers

# Load the model.
model = diffusers.load_model("stable-diffusion-128-v2")

# Print the model version.
print(model.model_version)

import torch
from hearsavi.savi import *

# Load model with weights - located in the root directory of this repo
model = load_model("models/best_val.pth")
print(model)

# Create a batch of 2 white noise clips that are 2-seconds long
# and compute scene embeddings for each clip
audio = torch.rand((2, 2, model.sample_rate * 2))
print(audio.shape[1])
embeddings, timestamps = get_timestamp_embeddings(audio, model)
print(audio.shape)
print(embeddings.shape)
print(timestamps)
embeddings = get_scene_embeddings(audio, model)
print(embeddings.shape)

# Returns an array of embeddings computed every 25ms over the duration of the input audio.
# An array of timestamps corresponding to each embedding is also returned.
embeddings = get_timestamp_embeddings(audio, model)
print(embeddings[0].shape)
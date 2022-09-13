import torch
import hearbaseline

# Load model with weights - located in the root directory of this repo
model = hearbaseline.load_model("../models/naive_baseline.pt")

# Create a batch of 2 white noise clips that are 2-seconds long
# and compute scene embeddings for each clip
audio = torch.rand((2, model.sample_rate * 2))
embeddings = hearbaseline.get_scene_embeddings(audio, model)
print(audio.shape)
print(embeddings.shape)

# Returns an array of embeddings computed every 25ms over the duration of the input audio.
# An array of timestamps corresponding to each embedding is also returned.
embeddings = hearbaseline.get_timestamp_embeddings(audio, model)
print(embeddings[0].shape) # 1000/25=40
print(len(embeddings)) # batch_size = 2
import torch
Input_size=3
EPOCHS=1
LEARNING_RATE=0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
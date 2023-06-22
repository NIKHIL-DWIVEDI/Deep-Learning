import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 10
dis_learning_rate=0.001
gen_learning_rate=0.001
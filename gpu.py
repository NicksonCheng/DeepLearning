import torch
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use the first available GPU
    print("GPU is available")
else:
    device = torch.device("cpu")   # Use CPU if no GPU is available
    print("GPU is not available")

print("Current device:", device)
# Check the number of available GPUs
num_gpus = torch.cuda.device_count()


# Print the number of GPUs
print("Number of GPUs:", num_gpus)
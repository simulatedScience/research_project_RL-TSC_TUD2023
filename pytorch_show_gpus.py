import torch

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} GPU(s) available:")
    for i in range(device_count):
        print(f"\tGPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs available.")

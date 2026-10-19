import torch
import torchvision
import os
from src.data.cifar10 import get_cifar10_dataloaders
from src.data.backdoor import create_asr_dataloader

def dump():
    train, test = get_cifar10_dataloaders("./data", 64, False)
    asr = create_asr_dataloader(test, target_class=0, trigger_size=3, batch_size=1)
    
    # Get one image
    img, label = next(iter(asr))
    img = img[0] # unbatched
    
    # Unnormalize
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    img_unnorm = img * std + mean
    
    os.makedirs("results", exist_ok=True)
    torchvision.utils.save_image(img_unnorm, "results/triggered_sample.png")
    print("Triggered image saved to results/triggered_sample.png")

if __name__ == "__main__":
    dump()

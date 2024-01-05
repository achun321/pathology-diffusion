from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn as nn
import torch
from diffusers import AutoencoderKL
from datasets import load_dataset
from tqdm import tqdm

device = 'cuda'
transform = transforms.Compose([transforms.Resize((512, 512)), 
                                transforms.ToTensor()])
train = load_dataset("../train")

train_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples["image"]]
    examples["image"] = [train_transforms(image) for image in images]
    return examples

train_dataset = train["train"].with_transform(preprocess_train)

#train.set_transform(transform_dataset)
dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Load the pre-trained VAE model
vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="vae").to(device)

optimizer = optim.Adam(vae.parameters(), lr=0.001)
reconstruction_loss_fn = nn.MSELoss()

vae.train()
# Fine-tuning loop
for epoch in range(50):
    for inputs in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{50}"):
        img = inputs["image"].to(device)
        optimizer.zero_grad()
        reconstructed= vae(img).sample
        loss = reconstruction_loss_fn(reconstructed, img)
        loss.backward()
        optimizer.step()
        print("LOSS: ", loss.item())
        
vae.save_pretrained("path-vae-finetuned")

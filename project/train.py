import wandb
import numpy as np
import matplotlib.pyplot as plt
import argparse

import torch
import torchvision
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F

from core.model import Generator, StyleEncoder, FAN, TextEncoder
from dataset import CelebADataset


def tensor_to_image(tensor):
    image = tensor.permute((1, 2, 0))
    mean=[0.5, 0.5, 0.5]
    std=[0.5, 0.5, 0.5]
    return np.clip(image.detach().cpu().numpy() * std + mean, 0, 1)

def show_images(batch, nrow=4):
    plt.figure(figsize=(10, 15))
    plt.imshow(tensor_to_image(torchvision.utils.make_grid(batch, nrow=nrow)))


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of epochs')
args = parser.parse_args()


models = torch.load('expr/checkpoints/celeba_hq/100000_nets_ema.ckpt')
style_encoder = StyleEncoder()
style_encoder.load_state_dict(models['style_encoder'])


transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    
dataset = CelebADataset('./data', transform)
dataloader = data.DataLoader(dataset=dataset,
                       batch_size=8,
                       shuffle=True,
                       num_workers=1,
                       pin_memory=True)



wandb.init(project="stargan-text-encoder")

text_encoder = TextEncoder().to('cuda')
style_encoder = style_encoder.eval().to('cuda')

loss_function = nn.L1Loss()

opt = torch.optim.Adam(text_encoder.parameters())

generator = Generator().to('cuda')
fan = FAN().to('cuda')
generator.load_state_dict(models['generator'], False)
fan.load_state_dict(models['fan'])
def sample_images(loader, save=True):
    x, descriptions = next(iter(loader))
    with torch.no_grad():
        y = descriptions['male']
        embs = descriptions['embedding']
        x, y, embs  = x.to('cuda'), y.to('cuda'), embs.to('cuda')
        target_text_style = text_encoder(embs, y)
        target_image_style = style_encoder(x, y)
        masks = fan.get_heatmap(x)
        p = torch.randperm(x.shape[0])
        source = x
        target = x[p]
        target_gender = y[p]
        target_text_style = target_text_style[p]
        target_image_style = target_image_style[p]
        
        gen_images = generator(x, target_image_style, masks=masks)
        gen_text_images = generator(x, target_text_style, masks=masks)
        plt.figure(figsize=(15, 35))
        for i in range(x.shape[0]):
            plt.subplot(x.shape[0], 4, 1 + i*4)
            original_im = tensor_to_image(source[i])
            target_im = tensor_to_image(target[i])
            gen_im = tensor_to_image(gen_images[i])
            gen_text_im = tensor_to_image(gen_text_images[i])
            #print(f'Caption: {description["caption"][p[i]]}')

            plt.title(f'original ({"male" if y[i] else "female"})')
            plt.imshow(original_im, interpolation="none")
            plt.subplot(x.shape[0], 4, 2 + i*4)
            plt.title(f'target ({"male" if target_gender[i] else "female"})')
            plt.imshow(target_im, interpolation="none")
            plt.subplot(x.shape[0], 4, 3 + i*4)
            plt.title(f'original style encoder')
            plt.imshow(gen_im, interpolation="none")
            plt.subplot(x.shape[0], 4, 4 + i*4)
            plt.title(f'text style encoder')
            plt.imshow(gen_text_im, interpolation="none")
            plt.gca().set_xlabel(f'Caption: {descriptions["caption"][p[i]]}')
        if save is None:
            plt.show()
        else:
            plt.savefig('expr/samples/'+save)



for epoch in range(args.num_epochs):
    for iteration, (x, description) in enumerate(dataloader):
        y = description['male']
        embs = description['embedding']
        x, y, embs  = x.to('cuda'), y.to('cuda'), embs.to('cuda')
        
        with torch.no_grad():
            styles_from_image = style_encoder(x, y)
        
        styles_from_text = text_encoder(embs, y)

        # Calculating the Loss
        loss = loss_function(styles_from_image, styles_from_text)
        opt.zero_grad()
        loss.backward()
        opt.step()
        wandb.log({f"l1 loss": loss.item()})
    if (epoch + 1) % 1 == 0:
        sample_images(dataloader, f'epoch{epoch+1}.png')
    torch.save(text_encoder.state_dict(), f"expr/text_encoder.pt")
        
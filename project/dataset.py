import os
import clip
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, device='cuda'):
        self.image_dir = f'{root_dir}/CelebA-HQ-img'
        self.transform = transform 
        self.annotations = pd.read_csv(f'{root_dir}/new_labels_hq.csv', sep=',').to_numpy()
        self.embeddings = []
        model = clip.load("ViT-B/32", device=device)[0].eval()
        for caption in tqdm(self.annotations[:, 1]):
            tokens = clip.tokenize(caption).to(device)
            with torch.no_grad():
                embeddings = model.encode_text(tokens).float().cpu()
            self.embeddings.append(embeddings)
              
    def __len__(self): 
        return self.annotations.shape[0]

    def __getitem__(self, idx):
        img_name = self.annotations[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)
        male = (self.annotations[idx, 2] + 1) // 2
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        caption = self.annotations[idx, 1]
        emb = self.embeddings[idx]
        return img, {'filename': img_name, 'idx': idx,
                     'male': torch.tensor(int(male)).long(),
                    'caption': caption,
                    'embedding':emb}
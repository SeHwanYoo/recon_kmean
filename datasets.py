import torch 
from torch.utils.data import Dataset 
import os 
import numpy as np
from PIL import Image
from torchvision import transforms

class ImageDataset(Dataset):
    # ⭐ __init__에 마스크 디렉토리 경로 추가
    def __init__(self, img_dir, mask_dir, list_file_path, img_transform=None, mask_transform=None):
        self.img_paths = [] # 기존과 동일
        self.mask_paths = []
        self.mask_dir = mask_dir # ⭐ 마스크 폴더 경로 저장
        self.image_dir = img_dir # 이미지 폴더 경로 저장
        
        with open(list_file_path, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                img_path = line.strip()
                
                if os.path.exists(os.path.join(img_dir, img_path)):
                    self.img_paths.append(os.path.join(img_dir, img_path))
                    self.mask_paths.append(os.path.join(mask_dir, img_path.replace('.tif', '_mask.png')))

        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.img_transform:
            image = self.img_transform(image)
        
        # base_filename = os.path.basename(img_path)
        # filename_without_ext = os.path.splitext(base_filename)[0]
        # mask_path = os.path.join(self.mask_dir, os.path.relpath(img_path, self.image_dir)).replace(base_filename, f"{filename_without_ext}_mask.png")
        
        mask = Image.open(mask_path).convert('L')
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch: return None, None
    images, masks = zip(*batch)
    return torch.stack(images, 0), torch.stack(masks, 0)# ⭐ K-means 클러스터 개수
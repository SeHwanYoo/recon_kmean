import os 
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# Distributed torch 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from datasets import ImageDataset, custom_collate_fn
from models import DualResUNet

import logging
from datetime import datetime
from tqdm import tqdm

def get_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("Trainer")
    if logger.hasHandlers(): return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"run_{time_tag}.log"))
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

# def setup_ddp():
#     # dist.init_process_group(backend='nccl')
#     dist.init_process_group(backend='gloo')
#     torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    
# def cleanup_ddp():
#     dist.destroy_process_group()

# --- 4. 메인 학습 함수 ---
def main():
    # --- ⭐ 설정 (Configuration) - config 딕셔너리 제거, 변수 직접 사용 ---
    BASE_IMG_DIR = '/home/users/ntu/sehwan00/scratch/pathology'
    MASK_DIR = '/home/users/ntu/sehwan00/scratch/pathology_masks'
    LIST_FILE_PATH = '/home/users/ntu/sehwan00/scratch/pathology/list.txt'
    IMAGE_SIZE = 256
    NUM_CLASSES = 4
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    RECON_LOSS_WEIGHT = 1.0
    SEG_LOSS_WEIGHT = 0.5
    CHECKPOINT_DIR = "/home/users/ntu/sehwan00/scratch/recon_kmean/checkpoints_cnn"
    LOG_DIR = "./logs_cnn"
    SAVE_INTERVAL = 5
    RESUME_FROM_CHECKPOINT = None 

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    # args = parser.parse_args()

    is_ddp = 'WORLD_SIZE' in os.environ
    rank, local_rank, world_size, device = 0, 0, 1, None

    if is_ddp:
        dist.init_process_group(backend='nccl')
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger = None
    if rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        logger = get_logger(LOG_DIR)
        logger.info(f"DDP Enabled: {is_ddp}, World Size: {world_size}")

    img_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
        transforms.ToTensor()])
    
    mask_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.NEAREST), 
                                         transforms.ToTensor(), 
                                         transforms.Lambda(lambda x: x.squeeze().long())])
    
    dataset = ImageDataset(BASE_IMG_DIR, MASK_DIR, LIST_FILE_PATH, img_transform, mask_transform)
    sampler = DistributedSampler(dataset) if is_ddp else None
    # train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=(not is_ddp), num_workers=4, collate_fn=custom_collate_fn, pin_memory=True)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=(not is_ddp), num_workers=0, collate_fn=custom_collate_fn, pin_memory=True)

    model = DualResUNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE * world_size)
    start_epoch = 0

    resume_from_checkpoint = None 
    if RESUME_FROM_CHECKPOINT is not None: 
        resume_from_checkpoint = os.path.join(CHECKPOINT_DIR, f'dual_unet_epoch_{RESUME_FROM_CHECKPOINT}.pth')
        # if os.path.exists(resume_from_checkpoint):
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        if rank == 0: logger.info(f"Resuming from epoch {start_epoch}")
    
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    reconstruction_criterion = nn.MSELoss().to(device)
    segmentation_criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(start_epoch, NUM_EPOCHS):
        if is_ddp: sampler.set_epoch(epoch)
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", disable=(rank != 0))

        for batch_data in progress_bar:
            if batch_data[0] is None: continue
            images, masks = batch_data
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            seg_logits, reconstructed_imgs = model(images)
            seg_loss = segmentation_criterion(seg_logits, masks)
            recon_loss = reconstruction_criterion(reconstructed_imgs, images)
            total_loss = (SEG_LOSS_WEIGHT * seg_loss) + (RECON_LOSS_WEIGHT * recon_loss)
            
            total_loss.backward()
            optimizer.step()

        if rank == 0:
            logger.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Total Loss: {total_loss.item():.4f}, Seg Loss: {seg_loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}")
            if (epoch + 1) % SAVE_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
                model_to_save = model.module if is_ddp else model
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"dual_unet_epoch_{epoch+1}.pth")
                torch.save({'epoch': epoch + 1, 'model_state_dict': model_to_save.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")

    if is_ddp: dist.destroy_process_group()
    if rank == 0: logger.info("Training finished.")

if __name__ == '__main__':
    main()
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

def setup_ddp():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    
def cleanup_ddp():
    dist.destroy_process_group()

if __name__ == '__main__':
    # --- 설정 (Configuration) ---
    IMAGE_DIR = '/home/users/ntu/sehwan00/scratch/pathology'
    MASK_DIR = '/home/users/ntu/sehwan00/scratch/pathology_masks' 
    LIST_FILE_PATH = '/home/users/ntu/sehwan00/scratch/pathology_list.txt' 
    K_MEANS_CLASSES = 4 
    IMAGE_SIZE= (256, 256) # 이미지 크기
    NUM_EPOCHS = 100 # 학습 에폭 수
    BATCH_SIZE = 16 # 배치 크기
    
    
    # ⭐ 손실 함수 가중치
    RECON_LOSS_WEIGHT = 1.0 # 복원 손실 가중치
    SEG_LOSS_WEIGHT = 0.5   # 분할 손실 가중치
    KLD_WEIGHT = 0.00025    # KL 손실 가중치

    setup_ddp() 
    rank = int(os.environ['LOCAL_RANK'])    
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    
    # --- 이미지 전처리 변환 ---
    img_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    # ⭐ 마스크용 전처리 (텐서로 변환하고 Long 타입으로 변경)
    mask_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze().long()) # (1, H, W) -> (H, W) LongTensor
    ])
    
    device = torch.device(f"cuda:{local_rank}")
    

    # --- 데이터셋 및 DataLoader 설정 ---
    # ⭐ ImageDataset 초기화 시 mask_dir와 mask_transform 전달
    dataset = ImageDataset( base_img_dir=IMAGE_DIR, mask_dir=MASK_DIR, list_file_path=LIST_FILE_PATH, img_transform=img_transform, mask_transform=mask_transform )    
    # DDP: DistributedSampler를 사용하여 각 GPU에 데이터 분배
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    train_dataloader = DataLoader( dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, collate_fn=custom_collate_fn, pin_memory=True )
 

    model = DDP(DualResUNet(in_channels=3, num_classes=K_MEANS_CLASSES)).to(device) 
    
    reconstruction_criterion = nn.MSELoss()
    segmentation_criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(...)
    
    for epoch in range(NUM_EPOCHS):
        for batch in train_dataloader:
            inputs, masks = batch 
            
            reconstructed, seg_logits, mu, log_var = model(inputs)

            recon_loss = reconstruction_criterion(reconstructed, inputs)
            kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            seg_loss = segmentation_criterion(seg_logits, masks) 

            loss = (RECON_LOSS_WEIGHT * recon_loss) + \
                   (SEG_LOSS_WEIGHT * seg_loss) + \
                   (KLD_WEIGHT * kld_loss)
                  
            loss.backward()
            optimizer.step()
            
        if rank == 0: 
            print(f"Epoch [{epoch}/{NUM_EPOCHS}], Loss: {loss.item():.4f}, "
                  f"Recon Loss: {recon_loss.item():.4f}, "
                  f"Seg Loss: {seg_loss.item():.4f}, "
                  f"KLD Loss: {kld_loss.item():.4f}")
                  
                  
        # BEST SAVE
        if rank == 0 and (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }, f'checkpoint_epoch_{epoch + 1}.pth')          
             
    if rank == 0:
        print("Training complete. Saving final model.")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'final_model.pth') 

    cleanup_ddp() 
import torch
import torchvision
import torchvision.models
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import time
import argparse
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from getdata_com import combineNet, NUM_CLASSES
from varfcn_model import CVAE_SE_FCN_Modulation
import cv2
import sys
import os
import os.path
import time
import random
import argparse
import math
from datetime import datetime
import os 

# Define a fixed seed
seed = 32541241
# Set the seed for reproducibility
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def train(args, model, optimizer, lr_scheduler, criterion, combine_dataloader, device,save_dir):
    is_better = True
    prev_loss = float('inf')
    epoch_kl_losses, epoch_ce_losses_prior, epoch_ce_losses_post, epoch_total_losses = [], [], [], []

    for epoch in range(args.NUM_EPOCHS): 
        train_data = enumerate(combine_dataloader)
        loss_f = 0
        t_start = time.time()
        total_kl_loss, total_ce_loss_prior, total_ce_loss_post = 0, 0, 0

        for batch_idx, data in tqdm(train_data, total=len(combine_dataloader)):
            optimizer.zero_grad()
            
            prior_input = data['prior_image']
            mask = data['mask']
            post_input = data['post_image']
    
            ens_mask = []
            for i in range(args.NUM_ENSEMBLES):
                ens_mask.append(data['mask'])

            ens_mask = torch.stack(ens_mask)
            mask = mask.unsqueeze(0).repeat(args.NUM_ENSEMBLES,1,1,1,1).permute(1,0,2,3,4)

            prior_input = prior_input.to(device)
            mask = mask.to(device)
            post_input = post_input.to(device)
            
            decoded_out_prior, kl_loss= model(prior_input, x_mask = post_input, training = True, num_ensembles= args.NUM_ENSEMBLES, device = device)
            
            decoded_out_prior = nn.Sigmoid()(decoded_out_prior).permute(1,0,2,3,4)
            
            ce_loss_prior = criterion(decoded_out_prior, mask)
            total_loss = 500 * ce_loss_prior + kl_loss 
            total_kl_loss += kl_loss.item()
            total_ce_loss_prior += ce_loss_prior.item()

            total_loss.backward()
            optimizer.step()
            
            loss_f += total_loss.item()

        lr_scheduler.step()

        
        # Calculating average losses for the epoch
        avg_kl_loss = total_kl_loss / len(combine_dataloader)
        avg_ce_loss_prior = total_ce_loss_prior / len(combine_dataloader)
        avg_total_loss = loss_f / len(combine_dataloader)

        # Storing the average losses
        epoch_kl_losses.append(avg_kl_loss)  # Use .item() to get the Python number from tensor
        epoch_ce_losses_prior.append(avg_ce_loss_prior)
        epoch_total_losses.append(avg_total_loss)  # Make sure avg_total_loss is also a tensor

        # Check for improvement
        is_better = avg_total_loss < prev_loss
        if is_better:
            prev_loss = avg_total_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "CVAE_SE_FCN_with_Modulation_SingleGPU.pth"))

        print("Epoch #{}\tKL Loss: {:.8f}\tCE Loss Prior: {:.8f} \tTime: {:2f}s".format(epoch+1, avg_kl_loss, avg_ce_loss_prior, time.time() - t_start))


    # Plotting the losses
    # Setting up a figure for subplots
    plt.figure(figsize=(15, 15))  # Adjust the figure size to accommodate the three plots vertically

    # Subplot for KL Loss
    plt.subplot(3, 1, 1)  # 3 rows, 1 column, 1st subplot
    plt.plot(epoch_kl_losses, label='KL Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('KL Loss')
    plt.title('KL Loss per Epoch')
    plt.legend()

    # Subplot for CE Loss Prior
    plt.subplot(3, 1, 2)  # 3 rows, 1 column, 2nd subplot
    plt.plot(epoch_ce_losses_prior, label='CE Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('CE Loss')
    plt.title('CE Loss per Epoch')
    plt.legend()

    # Subplot for Total Loss
    plt.subplot(3, 1, 3)  # 3 rows, 1 column, 3rd subplot
    plt.plot(epoch_total_losses, label='Total Loss', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.title('Total Loss per Epoch')
    plt.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_loss_subplots.png'))
    plt.close()


def main(args, device):
    # Get the current time and format it as a string
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"trained_model_{time_str}"
    save_dir = os.path.join(args.SAVE_DIR, folder_name)

    img_dir = args.DATASET_PATH +'/images'
    mask_dir = args.DATASET_PATH + '/masks'

    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    num_gpus = 1

    # define the data augmentation pipeline

    img_h = args.img_h_w[0]
    img_w = args.img_h_w[1]

    transform = A.Compose([
                A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT, p=0.5), 
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.0, rotate_limit=0, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomSizedCrop(min_max_height=(400, 520), height=520, width=520, p=0.7),
                ToTensorV2(),
            ])
    combine_dataset = combineNet(img_dir=img_dir,
                                        mask_dir=mask_dir, image_shape=(img_h,img_w), transform=transform)  

    combine_dataloader = DataLoader(combine_dataset, batch_size=args.BATCH_SIZE, shuffle= True, num_workers=8, drop_last=True)
   

    # Move the model to the appropriate device first
    model = CVAE_SE_FCN_Modulation(num_classes =args.NUM_CLASSES).to(device)

    criterion = torch.nn.BCELoss(reduction='mean').to(device)
    # criterion = torch.nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                        lr=args.LEARNING_RATE * np.sqrt(num_gpus)) # , betas=[0.5, 0.999]
    # Set different learning rates for encoder and decoder
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if 'decoder' in name:
            decoder_params.append(param)
        else:
            encoder_params.append(param)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[30, 60, 90], last_epoch= -1)
    train(args, model, optimizer, lr_scheduler, criterion, combine_dataloader, device, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--NUM_INPUT_CHANNELS', type=int, default=3)
    parser.add_argument('--NUM_CLASSES', type=int, default=NUM_CLASSES)
    parser.add_argument('--NUM_EPOCHS', type=int, default= 120)
    parser.add_argument('--LEARNING_RATE', type=float, default=1e-4)
    parser.add_argument('--BATCH_SIZE', type=int, default=6)
    parser.add_argument('--NUM_ENSEMBLES', type=int, default=2)
    parser.add_argument('--DATASET_PATH', type=str, default='./Dataset/train')
    parser.add_argument('--SAVE_DIR', type=str,  default='./Weights')
    parser.add_argument('--img_h_w', type=list, default=[520, 520], help="list containing image [H, W]")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(args, device)
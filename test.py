from __future__ import print_function, division
import os
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision
from tqdm import tqdm
import csv
from itertools import zip_longest
from sklearn.metrics import brier_score_loss
from getdata_com import combineNet, NUM_CLASSES
from varfcn_model import CVAE_SE_FCN_Modulation
from utils import *
import time


# Define a fixed seed
seed = 32541241
# Set the seed for reproducibility
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

def main(args):
    # Corruptions folders 
    folders = ["Brightness", "Contrast", "Defocus_Blur", "Elastic", "Gaussian_Noise", "Impulse_Noise", "Motion_Blur", "Pixelate", "Shot_Noise", "Zoom_Blur", "jpeg_comp"]
    
    # Directories from arguments
    img_dir = args.dataset_dir + '/images'
    mask_dir = args.dataset_dir + '/masks/whole'
    # Create results_dir using model checkpoint's folder
    results_dir = os.path.join(args.results_dir, args.model_checkpoint.split('/')[-2])
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    im_h = args.img_h_w[0]
    im_w = args.img_h_w[1]
    
    # Load model checkpoint
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CVAE_SE_FCN_Modulation(num_classes=NUM_CLASSES).to(device)
    state_dict = torch.load(args.model_checkpoint, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
                ])

    ent1_data = []
    ent2_data = []
    ent3_data = []
    ent4_data = []

    
    iou1_data = []
    iou2_data = []
    iou3_data = []
    iou4_data = []

    
    f11_data = []
    f12_data = []
    f13_data = []
    f14_data = []

    
    brier_1 = []
    brier_2 = []
    brier_3 = []
    brier_4 = []

    
    ECE_1 = []
    ECE_2 = []
    ECE_3 = []
    ECE_4 = []

    for corrup in folders:
        test_dir_out1 = args.dataset_dir + "/severity_lv1/"+corrup+"/"
        test_dir_out2 = args.dataset_dir + "/severity_lv2/"+corrup+"/"
        test_dir_out3= args.dataset_dir + "/severity_lv3/"+corrup+"/"
        test_dir_out4= args.dataset_dir + "/severity_lv4/"+corrup+"/"
        print('Calculating Scores for...',corrup)
        corruption = corrup.lower()
        if '_' in corruption:
            corruption = corruption.split('_')[0]
        
        ori_loader = combineNet(img_dir=img_dir,
                                        mask_dir=mask_dir, image_shape=(im_h,im_w), transform=transform)
        lv1_loader = combineNet(img_dir=test_dir_out1,
                                       mask_dir=mask_dir, image_shape=(im_h,im_w), transform=transform)
        lv2_loader = combineNet(img_dir=test_dir_out2,
                                       mask_dir=mask_dir, image_shape=(im_h,im_w), transform=transform)
        lv3_loader = combineNet(img_dir=test_dir_out3,
                                       mask_dir=mask_dir, image_shape=(im_h,im_w), transform=transform)
        lv4_loader = combineNet(img_dir=test_dir_out4,
                                       mask_dir=mask_dir, image_shape=(im_h,im_w), transform=transform)

        images_dir = os.listdir(img_dir)
        imgs = []
        iou_ori_data = []
        f1_ori_data = []
        ECE_ori = []
        brier_ori = []
        ent_data = []

        images_dir = os.listdir(img_dir)
        for image_name in tqdm(images_dir, total = len(images_dir)):
            name = image_name.split('.')[0]
            ori_image_path = os.path.join(img_dir, name + '.jpg')
            lv1_image_path = test_dir_out1 + name + corruption + '_.jpg'
            lv2_image_path = test_dir_out2 + name + corruption + '_.jpg'
            lv3_image_path = test_dir_out3 + name + corruption + '_.jpg'
            lv4_image_path = test_dir_out4 + name + corruption + '_.jpg'

            mask_path = os.path.join(mask_dir, name + '.bmp')
            mask = gt_mask_loader(mask_path,(im_h,im_w)).astype(np.float32).transpose(2,0,1)

            ori_image = ori_loader.load_test(path=ori_image_path).unsqueeze(dim=0)/255.
            lv1_image = lv1_loader.load_test(path=lv1_image_path).unsqueeze(dim=0)/255.
            lv2_image = lv2_loader.load_test(path=lv2_image_path).unsqueeze(dim=0)/255.
            lv3_image = lv3_loader.load_test(path=lv3_image_path).unsqueeze(dim=0)/255.
            lv4_image = lv4_loader.load_test(path=lv4_image_path).unsqueeze(dim=0)/255.

            ori_input = torch.FloatTensor(ori_image).to(device)
            lv1_input = torch.FloatTensor(lv1_image).to(device)
            lv2_input = torch.FloatTensor(lv2_image).to(device)
            lv3_input = torch.FloatTensor(lv3_image).to(device)
            lv4_input = torch.FloatTensor(lv4_image).to(device)

            num_ensembles = args.num_ensembles
            with torch.no_grad():
                start_time = time.time()
                ori_out = torch.sigmoid(model(ori_input, training = False, num_ensembles= num_ensembles))
                lv1_out = torch.sigmoid(model(lv1_input, training = False, num_ensembles= num_ensembles))
                lv2_out = torch.sigmoid(model(lv2_input, training = False, num_ensembles= num_ensembles))
                lv3_out = torch.sigmoid(model(lv3_input, training = False, num_ensembles= num_ensembles))
                lv4_out = torch.sigmoid(model(lv4_input, training = False, num_ensembles= num_ensembles))
                
                # end_time = time.time()
                # execution_time = end_time - start_time
                # print(f"Execution time of the code: {execution_time} seconds")

            ori_img_out = ori_out.permute(1,0,2,3,4).mean(1).squeeze().cpu().numpy()
            lv1_img_out = lv1_out.permute(1,0,2,3,4).mean(1).squeeze().cpu().numpy()
            lv2_img_out = lv2_out.permute(1,0,2,3,4).mean(1).squeeze().cpu().numpy()
            lv3_img_out = lv3_out.permute(1,0,2,3,4).mean(1).squeeze().cpu().numpy()
            lv4_img_out = lv4_out.permute(1,0,2,3,4).mean(1).squeeze().cpu().numpy()
         

            ECE_ori.append(get_expected_calibration_error(ori_img_out, mask))
            ECE_1.append(get_expected_calibration_error(lv1_img_out, mask))
            ECE_2.append(get_expected_calibration_error(lv2_img_out, mask))
            ECE_3.append(get_expected_calibration_error(lv3_img_out, mask))
            ECE_4.append(get_expected_calibration_error(lv4_img_out, mask))

            brier_ori.append(brier_score_loss(mask.reshape(-1).astype(np.int32), ori_img_out.reshape(-1)))
            brier_1.append(brier_score_loss(mask.reshape(-1).astype(np.int32), lv1_img_out.reshape(-1)))
            brier_2.append(brier_score_loss(mask.reshape(-1).astype(np.int32), lv2_img_out.reshape(-1)))
            brier_3.append(brier_score_loss(mask.reshape(-1).astype(np.int32), lv3_img_out.reshape(-1)))
            brier_4.append(brier_score_loss(mask.reshape(-1).astype(np.int32), lv4_img_out.reshape(-1)))

        #      #crosss entropy
            ori_enp = np.max(calc_entropy(ori_img_out), axis = 0)
            lv1_enp = np.max(calc_entropy(lv1_img_out), axis = 0)
            lv2_enp = np.max(calc_entropy(lv2_img_out), axis = 0)
            lv3_enp = np.max(calc_entropy(lv3_img_out), axis = 0)
            lv4_enp = np.max(calc_entropy(lv4_img_out), axis = 0)


            ent_data.append(np.mean(ori_enp))
            ent1_data.append(np.mean(lv1_enp))
            ent2_data.append(np.mean(lv2_enp))
            ent3_data.append(np.mean(lv3_enp))
            ent4_data.append(np.mean(lv4_enp))


        #     #iou
            ori_iou     =  np.nanmean(cal_iou(mask, ori_img_out)).round(4)
            lv1_img_iou =  np.nanmean(cal_iou(mask, lv1_img_out)).round(4)
            lv2_img_iou =  np.nanmean(cal_iou(mask, lv2_img_out)).round(4)
            lv3_img_iou =  np.nanmean(cal_iou(mask, lv3_img_out)).round(4)
            lv4_img_iou =  np.nanmean(cal_iou(mask, lv4_img_out)).round(4)

            iou_ori_data.append(ori_iou)
            iou1_data.append(lv1_img_iou)
            iou2_data.append(lv2_img_iou)
            iou3_data.append(lv3_img_iou)
            iou4_data.append(lv4_img_iou)

        #     #f1-score
            ori_f1 =     np.nanmean(cal_f1(mask, ori_img_out)).round(4)
            lv1_img_f1 = np.nanmean(cal_f1(mask, lv1_img_out)).round(4)
            lv2_img_f1 = np.nanmean(cal_f1(mask, lv2_img_out)).round(4)
            lv3_img_f1 = np.nanmean(cal_f1(mask, lv3_img_out)).round(4)
            lv4_img_f1 = np.nanmean(cal_f1(mask, lv4_img_out)).round(4)

            f1_ori_data.append(ori_f1)
            f11_data.append(lv1_img_f1)
            f12_data.append(lv2_img_f1)
            f13_data.append(lv3_img_f1)
            f14_data.append(lv4_img_f1)

    # Combine data into a list
    data = [iou_ori_data, iou1_data, iou2_data, iou3_data, iou4_data, \
    f1_ori_data, f11_data, f12_data, f13_data, f14_data, \
    ECE_ori, ECE_1, ECE_2, ECE_3, ECE_4, \
    brier_ori, brier_1, brier_2, brier_3, brier_4, \
    ent_data, ent1_data, ent2_data, ent3_data, ent4_data,    
    ]
    labels = ['IoU Ori', 'IoU Lv1', 'IoU Lv2', 'IoU Lv3', 'IoU Lv4', \
    'F1 Ori', 'F1 Lv1', 'F1 Lv2', 'F1 Lv3', 'F1 Lv4', \
    'ECE Ori', 'ECE Lv1', 'ECE Lv2', 'ECE Lv3', 'ECE Lv4', \
    'Brier Ori', 'Brier Lv1', 'Brier Lv2', 'Brier Lv3', 'Brier Lv4', \
    'Entropy Ori', 'Entropy Lv1', 'Entropy Lv2', 'Entropy Lv3', 'Entropy Lv4']

    # Save results to CSV
    csv_path = results_dir + '/result.csv'
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
        for row in zip_longest(*data, fillvalue='N/A'):
            writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation and Evaluation Script")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to the dataset root directory")
    parser.add_argument('--model_checkpoint', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--results_dir', type=str, default='./Results/', help="Path to save the output CSV file")
    parser.add_argument('--img_h_w', type=list, default=[520, 520], help="list containing image [H, W]")
    parser.add_argument('--num_ensembles', type=int, default= 10, help="Number of Ensemble")
    
    args = parser.parse_args()
    main(args)


'''


python test.py --dataset_dir Dataset/test/ \
                --model_checkpoint ./Weights/trained_model_20240926_214052/CVAE_SE_FCN_with_Modulation_SingleGPU.pth


'''

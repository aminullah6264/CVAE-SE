
import os
import numpy as np
from PIL import Image
from score import getPaths, db_eval_boundary, IoU_bin

def gt_mask_loader(path, mask_size):
    mask = Image.open(path)
    mask = mask.resize((mask_size[0], mask_size[1]))
    mask = np.asarray(mask)
    mask = mask/mask.max()
    imw, imh = mask.shape[0], mask.shape[1]  

    
    background = np.ones((imw, imh))
    Human1 = np.zeros((imw, imh))
    Robot1 = np.zeros((imw, imh))
    Fish1 = np.zeros((imw, imh))
    Reef1 = np.zeros((imw, imh))
    Wreck1 = np.zeros((imw, imh))

    mask_idx = np.where((mask[:,:,0] == 0) & (mask[:,:,1] == 0) & (mask[:,:,2] == 1))
    Human1[mask_idx] = 1
    background[mask_idx] = 0

    mask_idx = np.where((mask[:,:,0] == 1) & (mask[:,:,1] == 0) & (mask[:,:,2] == 0))
    Robot1[mask_idx] = 1
    background[mask_idx] = 0

    mask_idx = np.where((mask[:,:,0] == 1) & (mask[:,:,1] == 1) & (mask[:,:,2] == 0))
    Fish1[mask_idx] = 1
    background[mask_idx] = 0

    mask_idx = np.where((mask[:,:,0] == 1) & (mask[:,:,1] == 0) & (mask[:,:,2] == 1))
    Reef1[mask_idx] = 1
    background[mask_idx] = 0

    mask_idx = np.where((mask[:,:,0] == 0) & (mask[:,:,1] == 1) & (mask[:,:,2] == 1))
    Wreck1[mask_idx] = 1
    background[mask_idx] = 0
    # return np.stack((background, Robot1, Fish1, Human1, Reef1, Wreck1), -1) 
    return np.stack((Robot1, Fish1, Human1, Reef1, Wreck1), -1)


def save_images_subplot(images, filename):
    fig, axs = plt.subplots(4, 5, figsize=(15,12))
    axs = axs.ravel()
    
    titles = ['Input Image', 'Ground Truth', 'Prediction', 'Misclassification', 'Uncertainty Map']
    y_titles = ['Original Data', 'Severity_Lv1', 'Severity_Lv2', 'Severity_Lv3']
    title_idx = 0
    for i in range(0, 20, 5):
        for j in range(5):
            axs[i+j].tick_params(axis='both', which='both', length=0)
            axs[i+j].set_xticks([])
            axs[i+j].set_yticks([])
            axs[i+j].spines['top'].set_visible(False)
            axs[i+j].spines['right'].set_visible(False)
            axs[i+j].spines['bottom'].set_visible(False)
            axs[i+j].spines['left'].set_visible(False)

            if i == 0:
                axs[i+j].set_title(titles[j], size='large', weight='bold')
            if j == 3:
                img = images[i+j].copy()
                img[img > 0] = 255
                img[img <= 0] = 0
                axs[i+j].imshow(img, cmap='gray')
            else:
                if isinstance(images[i+j], np.ndarray) and len(images[i+j].shape) == 2:
                    axs[i+j].imshow(images[i+j], cmap='jet')
                else:
                    axs[i+j].imshow(images[i+j])
        axs[i].set_ylabel(y_titles[int(i/5)], size='large', weight='bold')
    plt.tight_layout()
    fig.align_labels()
    plt.savefig(filename)
    plt.close()



def visualize_gt_pred(in_image, gt_map, in_pred_map, in_entropy_map, incorrectPrediction,  
                        out_image1, out_pred_map1, out_entropy_map1, incorrectPrediction_out_d1,
                        out_image2, out_pred_map2, out_entropy_map2, incorrectPrediction_out_d2,
                        out_image3, out_pred_map3, out_entropy_map3, incorrectPrediction_out_d3,
                        corruption, image_name):
    label_to_color = {
    0: [0,  0, 0],
    1: [255,  0, 0],
    2: [ 255,  255,  0],
    3: [0, 0, 255],
    4: [255, 0, 255],
    5: [0,255,255],
    
    
}

    label_to_color2 =   {
    1: [255,  0, 0],
    2: [ 255,  255,  0],
    3: [0, 0, 255],
    4: [255, 0, 255],
    5: [0,255,255],
    
    
}

    f, axarr = plt.subplots(2,5, figsize=(10, 10))
    img_sample = in_image
    out_img_sample1 = out_image1
    out_img_sample2 = out_image2
    out_img_sample3 = out_image3
    gt_sample = gt_map.argmax(axis=-1).squeeze()
    in_pred_sample = in_pred_map
    out_pred_sample1 = out_pred_map1
    out_pred_sample2 = out_pred_map2
    out_pred_sample3 = out_pred_map3


    img_rgb_pred = np.zeros((in_pred_sample.shape[0], in_pred_sample.shape[1], 3), dtype=np.uint8)
    out_img_rgb_pred1 = np.zeros((out_pred_sample1.shape[0], out_pred_sample1.shape[1], 3), dtype=np.uint8)
    out_img_rgb_pred2 = np.zeros((out_pred_sample2.shape[0], out_pred_sample2.shape[1], 3), dtype=np.uint8)
    out_img_rgb_pred3 = np.zeros((out_pred_sample3.shape[0], out_pred_sample3.shape[1], 3), dtype=np.uint8)

    img_rgb_gt = np.zeros((in_pred_sample.shape[0], in_pred_sample.shape[1], 3), dtype=np.uint8)
    for gray, rgb in label_to_color.items():
        img_rgb_gt[gt_sample == gray, :] = rgb

    for gray, rgb in label_to_color2.items():
        img_rgb_pred[in_pred_sample[:,:,gray-1] == 1, :] = rgb
        out_img_rgb_pred1[out_pred_sample1[:,:,gray-1] == 1, :] = rgb
        out_img_rgb_pred2[out_pred_sample2[:,:,gray-1] == 1, :] = rgb
        out_img_rgb_pred3[out_pred_sample3[:,:,gray-1] == 1, :] = rgb



    images = [img_sample, img_rgb_gt, img_rgb_pred, incorrectPrediction, in_entropy_map,
                out_img_sample1, img_rgb_gt, out_img_rgb_pred1, incorrectPrediction_out_d1, out_entropy_map1,
                out_img_sample2, img_rgb_gt, out_img_rgb_pred2, incorrectPrediction_out_d2, out_entropy_map2,
                out_img_sample3, img_rgb_gt, out_img_rgb_pred3, incorrectPrediction_out_d3, out_entropy_map3,                
                ]

    uncertainity_dir = './Weights/trained_model_20240926_214052/Evaluation/' + corruption + '/'
    if not exists(uncertainity_dir): os.makedirs(uncertainity_dir)
    uncertainity_dir = uncertainity_dir +image_name

    save_images_subplot(images, uncertainity_dir)


def calc_entropy(p):
    eps = 1e-10
    return -(p ) * np.log2(p + eps) - (1 - p ) * np.log2(1 - p + eps)


def right_wrong(out, mask):
    right = []
    wrong  = []
    temp = out.copy()

    temp[temp>0.4] = 1.
    temp[temp<=0.4] = 0.

    right_idx = np.where(temp == mask)
    a,b,c=right_idx
    right_result = [[a[i], b[i], c[i]] for i in range(len(a))]
    wrong_idx = np.where(temp != mask)
    a,b,c=wrong_idx
    wrong_result = [[a[i], b[i], c[i]] for i in range(len(a))]
    
    #import ipdb ; ipdb.set_trace()
    for idx in right_result:
        # import ipdb ; ipdb.set_trace()
        right.append(out[idx[0], idx[1], idx[2]])
    for idx in wrong_result:
        wrong.append(out[idx[0], idx[1], idx[2]])
    right_pix = np.array(right)
    wrong_pix = np.array(wrong)
    return right_pix, wrong_pix

def cal_iou(target, prediction):
    
    prediction[prediction >= 0.4] = 1.
    prediction[prediction < 0.4] = 0.
    # import ipdb; ipdb.set_trace()
    iou_scores = np.zeros(target.shape[0])
    for i in range(target.shape[0]):
        if np.sum(target[i]) == 0:
            iou_scores[i] = np.nan
        else:
            intersection = np.logical_and(target[i], prediction[i])
            union = np.logical_or(target[i], prediction[i])
            iou_scores[i] = 1.0 * np.sum(intersection) / np.sum(union)
    return iou_scores

def cal_f1(target,prediction):

    prediction[prediction >= 0.4] = 1.
    prediction[prediction < 0.4] = 0.

    scores = []
    for i in range(5):
        if np.sum(target[i]) == 0.:
            scores.append(np.nan)
        else:
            precision, recall, f1 = db_eval_boundary(target[i], prediction[i])
            scores.append(f1)
    return scores

def get_expected_calibration_error(y_pred, y_true, num_bins=15):
        
    prob_y = np.max(y_pred, axis=0)
    pred_y = np.argmax(y_pred, axis=0)

    y_true = np.argmax(y_true, axis=0)
    
    correct = (pred_y == y_true).astype(float)
    
    bins = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins, right=False)
    num = 0
    for b in range(1, num_bins + 1):  # 1-based indexing for np.digitize
        mask = (bins == b)
        if np.any(mask):
            num += np.abs(np.sum(correct[mask] - prob_y[mask]))
            
    return num / (y_pred.shape[1] * y_pred.shape[2])

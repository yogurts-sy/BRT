import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['LOCAL_RANK'] = '0'
os.environ["RANK"] = '0'
os.environ["WORLD_SIZE"] = '1'
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = '8064'
import cv2
import time
import torch
import logging
import argparse
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from mmcv.ops.nms import nms
from torch.utils.data import DataLoader
from dataset.semicd import SemiCDDataset
from network.deeplabv3plus import DeepLabV3Plus
from util.dist_helper import setup_distributed
from util.utils import Evaluator, init_log

color_map = np.array([[[0, 0, 0], [0, 0, 255]], [[255, 255, 0], [255, 255, 255]]])
dist_port="80684"


BDN_frames_nums = 0
BDN_frames_times = 0.0


def compute_iou1(tensor1, tensor2):
    mask1 = tensor1.cpu().numpy().astype(int)
    mask2 = tensor2.cpu().numpy().astype(int)

    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)

    intersection_area = cv2.countNonZero(intersection)
    union_area = cv2.countNonZero(union)

    try:
        iou = intersection_area / union_area
    except ZeroDivisionError as e:
        return 0
    return iou


def compute_iou2(contour1, contour2):
    mask1 = np.zeros((256, 256))
    mask2 = np.zeros((256, 256))

    cv2.drawContours(mask1, [contour1], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(mask2, [contour2], -1, 255, thickness=cv2.FILLED)

    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)

    intersection_area = cv2.countNonZero(intersection)
    union_area = cv2.countNonZero(union)

    try:
        iou = intersection_area / union_area
    except ZeroDivisionError as e:
        return 0
    return iou

def filter_inside(mask, dets):
    index = []

    dets_indices = dets.astype(int)[:, :4]
    for i, (x1, y1, x2, y2) in enumerate(dets_indices):
        mask1 = mask[y1:y2, x1:x2]
        ratio = cv2.countNonZero(mask1) / 2304
        if ratio < 0.90:     # 0.90
            index.append(i)
    
    sdets = dets[index]

    return sdets, len(dets), len(sdets)


def find_float_boundary(maskdt, width=3):
    maskdt = torch.Tensor(maskdt).unsqueeze(0).unsqueeze(0)
    boundary_finder = maskdt.new_ones((1, 1, width, width))
    boundary_mask = F.conv2d(maskdt.permute(1, 0, 2, 3), boundary_finder,
                            stride=1, padding=width//2).permute(1, 0, 2, 3)
    bml = torch.abs(boundary_mask - width*width)
    bms = torch.abs(boundary_mask)
    fbmask = torch.min(bml, bms) / (width*width/2)
    return fbmask[0, 0].numpy()

def _force_move_back(sdets, H, W, patch_size):
    sdets = sdets.copy()
    s = sdets[:, 0] < 0
    sdets[s, 0] = 0
    sdets[s, 2] = patch_size

    s = sdets[:, 1] < 0
    sdets[s, 1] = 0
    sdets[s, 3] = patch_size

    s = sdets[:, 2] >= W
    sdets[s, 0] = W - 1 - patch_size
    sdets[s, 2] = W - 1

    s = sdets[:, 3] >= H
    sdets[s, 1] = H - 1 - patch_size
    sdets[s, 3] = H - 1
    return sdets

def get_dets(maskdt, patch_size, iou_thresh=0.3):
    fbmask = find_float_boundary(maskdt)
    ys, xs = np.where(fbmask)
    scores = fbmask[ys, xs]
    dets = np.stack([xs-patch_size//2, ys-patch_size//2,
                     xs+patch_size//2, ys+patch_size//2, scores]).T

    _, inds = nms(np.ascontiguousarray(dets[:, :4], np.float32),
                    np.ascontiguousarray(dets[:, 4], np.float32),
                    iou_thresh)
    sdets = dets[inds]

    H, W = maskdt.shape
    return _force_move_back(sdets, H, W, patch_size), len(dets), len(sdets)

def filtered_intersection(mask, intersection):
    mask[intersection == 255] = 0

    return mask

def filter_patches(patches_pred, predict, dets):
    predict = predict.squeeze(0)
    index = []
    ious = []

    dets_indices = dets.astype(int)[:, :4]
    for pid, (x1, y1, x2, y2) in enumerate(dets_indices):
        patch_mask = patches_pred[pid]
        pred_mask = predict[:, y1:y2, x1:x2]
    
        pred_mask = torch.sigmoid(pred_mask)    
        patch_mask = torch.sigmoid(patch_mask)
        patch_mask = (patch_mask + pred_mask) / 2
        
        patch_mask = patch_mask.argmax(dim=0)
        pred_mask = pred_mask.argmax(dim=0)

        iou = compute_iou1(patch_mask, pred_mask)

        if iou > 0.7:
            index.append(pid)
            ious.append(int(iou * 100))

    s_patches_pred = patches_pred[index]
    sdets = dets[index]

    return s_patches_pred, sdets, ious, len(dets), len(sdets)


def filter_small(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output1 = np.zeros((256, 256))

    for i, (contour1) in enumerate(contours):
        area = cv2.contourArea(contour1)
        if area < 30:
            continue

        cv2.drawContours(output1, [contour1], -1, 255, thickness=cv2.FILLED)

    kernel = np.ones((1, 1), np.uint8)
    output1 = cv2.dilate(output1, kernel, iterations = 1)

    return output1

def crop(img, maskgt, dets):
    dets = dets.astype(int)[:, :4]

    img_patches, gt_patches = None, []
    for x1, y1, x2, y2 in dets:
        if img_patches == None:
            img_patches = img[:, y1:y2, x1:x2].unsqueeze(0)
        else:
            img_patches = torch.cat((img_patches, img[:, y1:y2, x1:x2].unsqueeze(0)))
        gt_patches.append(maskgt[y1:y2, x1:x2])
    return img_patches, gt_patches


def filtered_intersection_small(newmask_refined, AB_intersection, outputs_contours):
    newmask_refined_mask = newmask_refined.clone().squeeze(0).cpu().numpy()
    newmask_refined_mask = (newmask_refined_mask * 255).astype(np.uint8)

    newmask_refined_mask = filtered_intersection(newmask_refined_mask, AB_intersection)

    newmask_refined_contours, _ = cv2.findContours(newmask_refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    saved_contours = []

    for contour in newmask_refined_contours:
        for outputs_contour in outputs_contours:
            iou_saved = compute_iou2(contour, outputs_contour)

            area = cv2.contourArea(outputs_contour)
            if area < 60 and iou_saved > 0.15:
                saved_contours.append(contour)
                break

            if area >= 60 and iou_saved > 0.05:
                saved_contours.append(contour)
                break

    mask_saved = np.zeros((256, 256))
    for contour in saved_contours:
        cv2.drawContours(mask_saved, [contour], -1, (255, 255, 255), cv2.FILLED)

    newmask_refined_contours, hierarchy = cv2.findContours(newmask_refined_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(newmask_refined_contours)):
        if hierarchy[0][i][3] != -1:
            contour = newmask_refined_contours[i]
            area = cv2.contourArea(contour)
            if area > 200:
                cv2.drawContours(mask_saved, [contour], -1, (0, 0, 0), cv2.FILLED)

    mask_saved = torch.from_numpy(np.array(mask_saved))

    mask_saved = mask_saved / 255
    mask_saved = mask_saved.long().unsqueeze(0).cuda()

    return mask_saved

def main():

    parser = argparse.ArgumentParser(description='Boundary Refinement Teacher Change Detection for Remote Sensing Images')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--crop_size', type=int, default=256, help='training dataset crop size')
    parser.add_argument('--dataset', type=str, default='WHU', help='training dataset name')
    parser.add_argument('--save_path', type=str, default='./exp/WHU/40')

    args = parser.parse_args()

    # **Replace your dataset path**

    if args.dataset == 'WHU':
        args.dataset_root = '/data2/suyou/Codes/datasets/WHU-CD256'
        args.val_id_path = './splits' + '/' + args.dataset + '/' + 'test.txt'
    elif args.dataset == 'LEVIR':
        args.dataset_root = '/data2/suyou/Codes/datasets/LEVIR-CD256'
        args.val_id_path = './splits' + '/' + args.dataset + '/' + 'test.txt'
    elif args.dataset == 'DSIFN':
        args.dataset_root = '/data2/suyou/Codes/datasets/DSIFN-CD256-HANet/train/'
        args.val_root = '/data2/suyou/Codes/datasets/DSIFN-CD256-HANet/val/'

    cd_save_path = args.save_path + '/' + 'CD'
    bdn_save_path = args.save_path + '/' + 'BDN'

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=dist_port)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus()

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    model_bdn = DeepLabV3Plus()

    model_bdn = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_bdn)
    model_bdn.cuda()

    model_bdn = torch.nn.parallel.DistributedDataParallel(model_bdn, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    valset = SemiCDDataset(args.dataset, args.dataset_root, 'val', args.crop_size, args.val_id_path)

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler)

    if os.path.exists(os.path.join(cd_save_path, 'best.pth')):
        checkpoint = torch.load(os.path.join(cd_save_path, 'best.pth'))
        model.load_state_dict(checkpoint['model'])

        logger.info('Load best change detection model from best checkpoint.')
    else:
        logger.warning('Not found change detection model.')
        return

    if os.path.exists(os.path.join(bdn_save_path, 'best.pth')):
        checkpoint = torch.load(os.path.join(bdn_save_path, 'best.pth'))
        model_bdn.load_state_dict(checkpoint['model'])
    
        logger.info('Load best boundary refinement teacher model from best checkpoint.\n')
    else:
        logger.warning('Not found boundary refinement teacher model.')
        return

    model.eval()
    model_bdn.eval()

    Eva_test = Evaluator(num_class=2)

    with torch.no_grad():
        for imgA, imgB, mask, id, _, _ in tqdm(valloader):

            id = id[0].split(".")[0]
            # os.makedirs(os.path.join(base_path, id), exist_ok=True)
            # visualize_save_path = os.path.join(base_path, id)

            imgA = imgA.cuda()
            imgB = imgB.cuda()

            predict = model(imgA, imgB)
            pred_out = predict.max(1)[1].cpu().numpy()
            outputs = predict.argmax(dim=1)

            pred = outputs.cpu().numpy()

            pred_A = model(imgA, mode='seg')
            pred_B = model(imgB, mode='seg')

            # --------------# Pre-Processing --------------------
            # 1. Find Contours

            outputs_mask = outputs.clone().squeeze(0).cpu().numpy()
            outputs_mask = (outputs_mask * 255).astype(np.uint8)
            outputs_contours, _ = cv2.findContours(outputs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            pred_A = torch.argmax(pred_A[0].clone(), dim=0, keepdim=True).squeeze(0)
            pred_A_mask = pred_A.clone().cpu().numpy()
            pred_A_mask = (pred_A_mask * 255).astype(np.uint8)
            pred_A_contours, _ = cv2.findContours(pred_A_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            pred_B = torch.argmax(pred_B[0].clone(), dim=0, keepdim=True).squeeze(0)
            pred_B_mask = pred_B.clone().cpu().numpy()
            pred_B_mask = (pred_B_mask * 255).astype(np.uint8)
            pred_B_contours, _ = cv2.findContours(pred_B_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            AB_intersection = cv2.bitwise_and(pred_A_mask, pred_B_mask)
            AB_intersection[outputs_mask == 255] = 0
            AB_intersection = filter_small(AB_intersection)

            # 2. Contour Match (CM)
            matched_contours = []

            for contour in outputs_contours:
                iou_best = 0
                matched_contour = None
                for contour_A in pred_A_contours:
                    iou_A = compute_iou2(contour, contour_A)
                    if iou_A > iou_best:
                        matched_contour = (contour, contour_A, 'A')
                        iou_best = iou_A
                for contour_B in pred_B_contours:
                    iou_B = compute_iou2(contour, contour_B)
                    if iou_B > iou_best:
                        matched_contour = (contour, contour_B, 'B')
                        iou_best = iou_B
                if matched_contour:
                    matched_contours.append(matched_contour)

            dets = []
            img_patches = None
            
            for contour_mask, _, from_where in matched_contours:

                area = cv2.contourArea(contour_mask)
                if area <= 50:
                    continue

                mask1 = np.zeros_like(outputs_mask)
                cv2.drawContours(mask1, [contour_mask], -1, 1, cv2.FILLED)

                mask2 = np.zeros_like(outputs_mask)
                cv2.drawContours(mask2, [contour_mask], -1, (255, 255, 255), cv2.FILLED)

                # 3. Boundary Block Extraction (BBE) & Overlap Filter
                dets_one, _, _ = get_dets(mask1, 48, 0.2)


                # 4. Inside Filter
                dets_one, _, _ = filter_inside(mask1, dets_one)

                img = imgA if from_where == 'A' else imgB
                img = img.clone()[0]
                maskgt = mask.clone().squeeze(0).cpu().numpy()
                img_one_patches, _ = crop(img, maskgt, dets_one)

                if len(dets) == 0:
                    dets = dets_one
                else:
                    dets = np.concatenate((dets, dets_one), axis=0)
                if img_patches == None:
                    img_patches = img_one_patches
                else:
                    img_patches = torch.cat((img_patches, img_one_patches))
            

            # --------------# Post-Processing --------------------

            if img_patches != None and len(img_patches) != 0:

                img_patches = F.interpolate(img_patches, size=(128, 128), mode="bilinear", align_corners=True)
                patches_pred = model_bdn(img_patches, mode='seg') # BDN infer

                patches_pred = F.interpolate(patches_pred, size=(48, 48), mode="bilinear", align_corners=True)

                # 5. Dist. Filter
                patches_pred, dets, _, _, _ = filter_patches(patches_pred, predict.clone(), dets)

                # 6. Assemble
                newmask = predict.clone()  
                newmask = newmask.squeeze(0)
                newmask_refined = torch.zeros((2, 256, 256)).cuda()
                newmask_count = torch.zeros((2, 256, 256)).cuda()

                dets_indices = dets.astype(int)[:, :4]
                for pid, (x1, y1, x2, y2) in enumerate(dets_indices):
                    patch_mask = patches_pred[pid]
                    newmask_refined[:, y1:y2, x1:x2] = patch_mask[:]
                    newmask_count[:, y1:y2, x1:x2] += 1

                s = newmask_count > 0
                newmask_refined[s] /= newmask_count[s]
                mask_mixed = newmask_refined.clone()
                mask_mixed = 0.5 * mask_mixed + 0.5 * newmask
                mask_mixed = mask_mixed.argmax(dim=0).unsqueeze(0)
    
                mask_mixed = filtered_intersection_small(mask_mixed, AB_intersection, outputs_contours)
                pred = mask_mixed.cpu().numpy().astype(int)

                pred_out = pred

            Eva_test.add_batch(mask.cpu().numpy(), pred)

            pred_rgb_out = np.array([color_map[l][p] for l, p in zip(mask.reshape(-1), pred_out.reshape(-1))])
            pred_rgb_out = pred_rgb_out.reshape((256, 256, 3)).astype(np.uint8)
            # cv2.imwrite(f'{visualize_save_path}/{id}_color.png', pred_rgb_out)


    IoU = Eva_test.Intersection_over_Union()
    Pre = Eva_test.Precision()
    Recall = Eva_test.Recall()
    F1 = Eva_test.F1()
    OA = Eva_test.OA()
    Kappa = Eva_test.Kappa()

    if rank == 0:
        logger.info('***** Evaluation ***** >>>> IoU: %.2f, F1: %.2f, Precision: %.2f, Recall: %.2f, OA: %.2f, Kappa: %.4f' % ( IoU[1] * 100, F1[1] * 100, Pre[1] * 100, Recall[1] * 100, OA[1] * 100, Kappa[1]))


if __name__ == '__main__':
    # Test Boundary Refinement Teacher (BRT) **WITH** Bi-temporal Image Boundary Refinement (BIBR)
    main()

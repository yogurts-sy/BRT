import cv2
import os
import torch
import numpy as np
import torch.nn.functional as F

from mmcv.ops.nms import nms


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
    return _force_move_back(sdets, H, W, patch_size)


def filtered_intersection(mask, intersection):
    mask[intersection == 255] = 0

    return mask


def filter_patches(patches_pred, predict, dets):
    predict = predict.squeeze(0)
    index = []
    # s_IMG_patches = []
    ious = []

    dets_indices = dets.astype(int)[:, :4]
    for pid, (x1, y1, x2, y2) in enumerate(dets_indices):
        patch_mask = patches_pred[pid]                      # patches predict [2, 48, 48]
        pred_mask = predict[:, y1:y2, x1:x2]                # CD predict      [2, 48, 48]
    
        pred_mask = torch.sigmoid(pred_mask)    
        patch_mask = torch.sigmoid(patch_mask)
        patch_mask = (patch_mask + pred_mask) / 2
        
        patch_mask = patch_mask.argmax(dim=0)
        pred_mask = pred_mask.argmax(dim=0)

        iou = compute_iou1(patch_mask, pred_mask)

        if iou > 0.7:
            index.append(pid)
            # s_IMG_patches.append(IMG_patches[pid])
            ious.append(int(iou * 100))

    s_patches_pred = patches_pred[index]
    sdets = dets[index]

    return s_patches_pred, sdets, ious # s_IMG_patches, 


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

def filter_inside(mask, dets):
    index = []

    dets_indices = dets.astype(int)[:, :4]
    for i, (x1, y1, x2, y2) in enumerate(dets_indices):
        mask1 = mask[y1:y2, x1:x2]
        ratio = cv2.countNonZero(mask1) / 2304
        if ratio < 0.90:
            index.append(i)
    
    sdets = dets[index]

    return sdets

def crop(img, dets):
    dets = dets.astype(int)[:, :4]

    img_patches = None
    for x1, y1, x2, y2 in dets:
        if img_patches == None:
            img_patches = img[:, y1:y2, x1:x2].unsqueeze(0)
        else:
            img_patches = torch.cat((img_patches, img[:, y1:y2, x1:x2].unsqueeze(0)))

    return img_patches


def filtered_intersection_small(newmask_refined, AB_intersection, outputs_contours):
    newmask_refined_mask = newmask_refined.clone().squeeze(0).cpu().numpy()
    newmask_refined_mask = (newmask_refined_mask * 255).astype(np.uint8)

    newmask_refined_mask = filtered_intersection(newmask_refined_mask, AB_intersection)

    newmask_refined_contours, _ = cv2.findContours(newmask_refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    saved_contours = []

    for contour in newmask_refined_contours:
        mask1 = np.zeros((256, 256))
        cv2.drawContours(mask1, [contour], -1, (255, 255, 255), cv2.FILLED)
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


def boundary_refine(imgA, imgB, br_model, predict, pred_A, pred_B, visualize_save_path, id):
    imgA, imgB = imgA.unsqueeze(0), imgB.unsqueeze(0)
    predict, pred_A, pred_B = predict.unsqueeze(0), pred_A.unsqueeze(0), pred_B.unsqueeze(0)
    outputs = predict.argmax(dim=1)

    outputs_mask = outputs.clone().squeeze(0).cpu().numpy()
    outputs_mask = (outputs_mask * 255).astype(np.uint8)
    outputs_contours, _ = cv2.findContours(outputs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imwrite(f'{visualize_save_path}/{id}_gt_mask.png', outputs_mask)

    pred_A = torch.argmax(pred_A[0].clone(), dim=0, keepdim=True).squeeze(0)
    pred_A_mask = pred_A.clone().cpu().numpy()
    pred_A_mask = (pred_A_mask * 255).astype(np.uint8)
    pred_A_contours, _ = cv2.findContours(pred_A_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imwrite(f'{visualize_save_path}/{id}_predA_mask.png', pred_A_mask)

    pred_B = torch.argmax(pred_B[0].clone(), dim=0, keepdim=True).squeeze(0)
    pred_B_mask = pred_B.clone().cpu().numpy()
    pred_B_mask = (pred_B_mask * 255).astype(np.uint8)
    pred_B_contours, _ = cv2.findContours(pred_B_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imwrite(f'{visualize_save_path}/{id}_predB_mask.png', pred_B_mask)

    AB_intersection = cv2.bitwise_and(pred_A_mask, pred_B_mask)
    # cv2.imwrite(f'{visualize_save_path}/{id}_AB_intersection.png', AB_intersection)


    AB_intersection[outputs_mask == 255] = 0
    AB_intersection = filter_small(AB_intersection)
    # cv2.imwrite(f'{visualize_save_path}/{id}_AB_intersection_filtered.png', AB_intersection)

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
        if area <= 50: # small boundary not to refine
            continue

        mask1 = np.zeros_like(outputs_mask)
        cv2.drawContours(mask1, [contour_mask], -1, 1, cv2.FILLED)

        mask2 = np.zeros_like(outputs_mask)
        cv2.drawContours(mask2, [contour_mask], -1, (255, 255, 255), cv2.FILLED)
        # cv2.imwrite(f'{visualize_save_path}/{id}_contours_{ii}_match_{from_where}.png', mask2)

        dets_one = get_dets(mask1, 48, 0.2)

        dets_one = filter_inside(mask1, dets_one)

        img = imgA if from_where == 'A' else imgB
        img = img.clone()[0]
        img_one_patches = crop(img, dets_one)

        if len(dets) == 0:
            dets = dets_one
        else:
            dets = np.concatenate((dets, dets_one), axis=0)
        if img_patches == None:
            img_patches = img_one_patches
        else:
            img_patches = torch.cat((img_patches, img_one_patches))

    if img_patches != None and len(img_patches) != 0 and len(img_patches) != 1:

        img_patches = F.interpolate(img_patches, size=(128, 128), mode="bilinear", align_corners=True)

        patches_pred = br_model(img_patches, mode='seg')  # [b, 2, 128, 128]

        patches_pred = F.interpolate(patches_pred, size=(48, 48), mode="bilinear", align_corners=True)

        patches_pred, dets, _ = filter_patches(patches_pred, predict.clone(), dets)

        if dets.shape[0] == 0:
            return outputs

        predict_ori = predict.clone()                          # [2, 256, 256]
        predict_ori = predict_ori.squeeze(0)
        predict_refined = torch.zeros((2, 256, 256)).cuda()    # [2, 256, 256]
        predict_count = torch.zeros((2, 256, 256)).cuda()      # [2, 256, 256]

        dets_indices = dets.astype(int)[:, :4]
        for pid, (x1, y1, x2, y2) in enumerate(dets_indices):
            patch_mask = patches_pred[pid]
            predict_refined[:, y1:y2, x1:x2] = patch_mask[:]
            predict_count[:, y1:y2, x1:x2] += 1

        s = predict_count > 0
        predict_refined[s] /= predict_count[s]

        predict_mixed = predict_refined.clone()
        predict_mixed = (predict_mixed + predict_ori) / 2

        mask_mixed = predict_mixed.argmax(dim=0).unsqueeze(0)
        mask_mixed = filtered_intersection_small(mask_mixed, AB_intersection, outputs_contours)

        return mask_mixed
    
    else:
        return outputs


def boundary_refine_v2(imgA, imgB, br_model, predict, pred_A, pred_B):
    # --------------# Pre-Processing --------------------

    imgA, imgB = imgA.unsqueeze(0), imgB.unsqueeze(0)
    predict, pred_A, pred_B = predict.unsqueeze(0), pred_A.unsqueeze(0), pred_B.unsqueeze(0)
    outputs = predict.argmax(dim=1)

    # 1. Find Contours

    outputs_mask = outputs.clone().squeeze(0).cpu().numpy()
    outputs_mask = (outputs_mask * 255).astype(np.uint8)
    outputs_contours, _ = cv2.findContours(outputs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imwrite(f'{visualize_save_path}/{id}_gt_mask.png', outputs_mask)

    pred_A = torch.argmax(pred_A[0].clone(), dim=0, keepdim=True).squeeze(0)
    pred_A_mask = pred_A.clone().cpu().numpy()
    pred_A_mask = (pred_A_mask * 255).astype(np.uint8)
    pred_A_contours, _ = cv2.findContours(pred_A_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imwrite(f'{visualize_save_path}/{id}_predA_mask.png', pred_A_mask)

    pred_B = torch.argmax(pred_B[0].clone(), dim=0, keepdim=True).squeeze(0)
    pred_B_mask = pred_B.clone().cpu().numpy()
    pred_B_mask = (pred_B_mask * 255).astype(np.uint8)
    pred_B_contours, _ = cv2.findContours(pred_B_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imwrite(f'{visualize_save_path}/{id}_predB_mask.png', pred_B_mask)


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
        if area <= 50: # small boundary not to refine
            continue

        mask1 = np.zeros_like(outputs_mask)
        cv2.drawContours(mask1, [contour_mask], -1, 1, cv2.FILLED)

        mask2 = np.zeros_like(outputs_mask)
        cv2.drawContours(mask2, [contour_mask], -1, (255, 255, 255), cv2.FILLED)
        # cv2.imwrite(f'{visualize_save_path}/{id}_contours_{ii}_match_{from_where}.png', mask2)

        # 3. Boundary Block Extraction (BBE) & Overlap Filter
        dets_one = get_dets(mask1, 48, 0.2)

        # 4. Inside Filter
        dets_one = filter_inside(mask1, dets_one)

        img = imgA if from_where == 'A' else imgB
        img = img.clone()[0]
        img_one_patches = crop(img, dets_one)

        if len(dets) == 0:
            dets = dets_one
        else:
            dets = np.concatenate((dets, dets_one), axis=0)
        if img_patches == None:
            img_patches = img_one_patches
        else:
            img_patches = torch.cat((img_patches, img_one_patches))
    
    # --------------# Post-Processing --------------------

    if img_patches != None and len(img_patches) != 0 and len(img_patches) != 1:

        img_patches = F.interpolate(img_patches, size=(128, 128), mode="bilinear", align_corners=True)

        patches_pred = br_model(img_patches, mode='seg') # BDN

        patches_pred = F.interpolate(patches_pred, size=(48, 48), mode="bilinear", align_corners=True)

        # 5. Dist. Filter
        patches_pred, dets, _ = filter_patches(patches_pred, predict.clone(), dets)

        if dets.shape[0] == 0:
            return predict
        
        # 6. Assemble

        predict_ori = predict.clone()  
        predict_ori = predict_ori.squeeze(0)
        predict_refined = torch.zeros((2, 256, 256)).cuda()
        predict_count = torch.zeros((2, 256, 256)).cuda()

        dets_indices = dets.astype(int)[:, :4]
        for pid, (x1, y1, x2, y2) in enumerate(dets_indices):
            patch_mask = patches_pred[pid]
            predict_refined[:, y1:y2, x1:x2] = patch_mask[:]
            predict_count[:, y1:y2, x1:x2] += 1

        s = predict_count > 0
        predict_refined[s] /= predict_count[s]

        predict_mixed = predict_refined.clone()
        predict_mixed = (predict_mixed + predict_ori) / 2

        return predict_mixed.unsqueeze(0)
    
    else:
        return predict
    
def boundaries_refine(imgA, imgB, br_model, predict, pred_A, pred_B, ids, base_path):
    batch_size = imgA.shape[0]
    mask_refined = torch.zeros((0)).cuda()
    for i in range(batch_size):
        id = ids[i].split(".")[0]
        # os.makedirs(os.path.join(base_path, id), exist_ok=True)
        visualize_save_path = os.path.join(base_path, id)
        mask_mixed = boundary_refine(imgA[i], imgB[i], br_model, predict[i], pred_A[i], pred_B[i], visualize_save_path, id)

        if mask_refined.shape == (0,):
            mask_refined = mask_mixed
        else:
            mask_refined = torch.cat([mask_refined, mask_mixed], axis=0)
    
    return mask_refined

def boundaries_refine_v2(imgA, imgB, br_model, predict, pred_A, pred_B):
    batch_size = imgA.shape[0]
    predicts_refined = torch.zeros((0)).cuda()
    for i in range(batch_size):
        predict_mixed = boundary_refine_v2(imgA[i], imgB[i], br_model, predict[i], pred_A[i], pred_B[i]).clone()

        if predicts_refined.shape == (0,):
            predicts_refined = predict_mixed
        else:
            predicts_refined = torch.cat([predicts_refined, predict_mixed], axis=0)
    
    return predicts_refined

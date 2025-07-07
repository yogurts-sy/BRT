import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import utils
from torchvision.utils import save_image


vis_root = './visualize'

def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5

def _visualize_pred(pred):
    pred = torch.argmax(pred, dim=1, keepdim=True)
    pred_vis = pred * 255
    return pred_vis

def make_numpy_grid(tensor_data, pad_value=0, padding=0):

    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis

def visualize_patches(tensor, tensor_gt, base_path, name=''):
    img_margin_inner = np.zeros([48, 2, 3])
    batch_name = os.path.join(base_path, name + '.png')

    tensor_gt = torch.from_numpy(tensor_gt).long().unsqueeze(0)
    tensor_gt = make_numpy_grid(tensor_gt)

    vis = np.concatenate([tensor, img_margin_inner, tensor_gt], axis=1)

    vis = np.clip(vis, a_min=0.0, a_max=1.0)
    plt.imsave(batch_name, vis)

def visualize_pred_patches(tensor, tensor_pred, tensor_gt, base_path, name=''):
    img_margin_inner = np.zeros([48, 2, 3])
    batch_name = os.path.join(base_path, name + '.png')

    tensor_pred = tensor_pred * 255
    tensor_pred = make_numpy_grid(tensor_pred)

    tensor_gt = tensor_gt * 255
    tensor_gt = make_numpy_grid(tensor_gt)

    vis = np.concatenate([tensor, img_margin_inner, tensor_pred, img_margin_inner, tensor_gt], axis=1)

    vis = np.clip(vis, a_min=0.0, a_max=1.0)
    plt.imsave(batch_name, vis)

def visualize_pred_patches_two(tensor, tensor_pred, base_path, name=''):
    img_margin_inner = np.zeros([48, 2, 3])
    batch_name = os.path.join(base_path, name + '.png')

    tensor_pred = tensor_pred * 255
    tensor_pred = make_numpy_grid(tensor_pred)

    vis = np.concatenate([tensor, img_margin_inner, tensor_pred], axis=1)

    vis = np.clip(vis, a_min=0.0, a_max=1.0)
    plt.imsave(batch_name, vis)

def visualize_batch_patches(tensor, predict, tensor_gt, base_path, batch_size, name=''):
    img_margin_inner = np.zeros([128, 2, 3])
    predict = _visualize_pred(predict)
    batch_name = os.path.join(base_path, name + '.png')

    # batch_size = 16
    batch_img = np.array([])
    for i in range(batch_size):
        I = make_numpy_grid(tensor[i])
        P = make_numpy_grid(predict[i])
        M = make_numpy_grid(tensor_gt[i])

        vis = np.concatenate([I, img_margin_inner, P, img_margin_inner, M], axis=1)
        if batch_img.shape == (0,):
            batch_img = vis
        else:
            batch_img = np.concatenate([batch_img, vis], axis=0)

    batch_img = np.clip(batch_img, a_min=0.0, a_max=1.0)
    plt.imsave(batch_name, batch_img)

def visualize(tensor_A, tensor_B, tensor_gt):
    tensor_A = _visualize_pred(tensor_A)
    tensor_B = _visualize_pred(tensor_B)
    img_margin_inner = np.zeros([256, 5, 3])
    batch_name = os.path.join(vis_root, 'AB.png')

    batch_size = 1
    batch_img = np.array([])
    for i in range(batch_size):
        A = make_numpy_grid(tensor_A[i])
        B = make_numpy_grid(tensor_B[i])
        M = make_numpy_grid(tensor_gt[i])
        vis = np.concatenate([A, img_margin_inner, B, img_margin_inner, M], axis=1)
        #vis = np.concatenate([A, B], axis=1)
        if batch_img.shape == (0,):
            batch_img = vis
        else:
            # img_margin = np.zeros([10, 1039, 3])
            batch_img = np.concatenate([batch_img, vis], axis=0)

    batch_img = np.clip(batch_img, a_min=0.0, a_max=1.0)
    plt.imsave(batch_name, batch_img) # RGB的值必须是0-1之间

def visualize2(tensor_A, tensor_B, tensor_gt):
    save_image(tensor_A[0], 'img1.png')


def visualize_pred(tensor_A, tensor_B, predict, tensor_gt=None, batch_size=3, name=''):
    img_margin_inner = np.zeros([256, 5, 3])
    batch_name = os.path.join(vis_root, name + '.png')
    predict = _visualize_pred(predict)

    batch_img = np.array([])
    for i in range(batch_size):

        A = make_numpy_grid(tensor_A[i])
        B = make_numpy_grid(tensor_B[i])
        P = make_numpy_grid(predict[i])
        if tensor_gt != None:
            M = make_numpy_grid(tensor_gt[i])
        vis = np.concatenate([A, img_margin_inner, B, img_margin_inner, P, img_margin_inner, M], axis=1)
        if batch_img.shape == (0,):
            batch_img = vis
        else:
            # img_margin = np.zeros([10, 1039, 3])
            batch_img = np.concatenate([batch_img, vis], axis=0)

    batch_img = np.clip(batch_img, a_min=0.0, a_max=1.0)
    plt.imsave(batch_name, batch_img) # RGB的值必须是0-1之间


def visualize_cutmixed(tensor_A, tensor_B, predict, tensor_gt=None, batch_size=3, name='', dist_dir='unimatch-all-2'):
    img_margin_inner = np.zeros([256, 5, 3])
    batch_name = os.path.join(vis_root, dist_dir, name + '.png')
    predict = _visualize_pred(predict)

    batch_img = np.array([])
    for i in range(batch_size):

        A = make_numpy_grid(tensor_A[i])
        B = make_numpy_grid(tensor_B[i])
        P = make_numpy_grid(predict[i])
        #M_cutmixed = make_numpy_grid(cutmixed_gt[i])
        M = make_numpy_grid(tensor_gt[i])
        vis = np.concatenate([A, img_margin_inner, B, img_margin_inner, P, img_margin_inner, M], axis=1)
        if batch_img.shape == (0,):
            batch_img = vis
        else:
            # img_margin = np.zeros([10, 1039, 3])
            batch_img = np.concatenate([batch_img, vis], axis=0)

    batch_img = np.clip(batch_img, a_min=0.0, a_max=1.0)
    plt.imsave(batch_name, batch_img) # RGB的值必须是0-1之间


def visualize_cutmixed_v2(tensor_A, tensor_B, predict, tensor_gt=None, batch_size=3, name='', dist_dir='unimatch-all-2'):
    batch_name = os.path.join(vis_root, dist_dir, name + '.png')
    # predict = _visualize_pred(predict)

    plt.figure(figsize=(10, 4), dpi=340)

    for i in range(batch_size):
        A = make_numpy_grid(tensor_A[i])
        A = np.clip(A, a_min=0.0, a_max=1.0)
        plt.subplot(batch_size, 4, i+1)
        plt.axis("off")
        plt.imshow(A)

        B = make_numpy_grid(tensor_B[i])
        B = np.clip(B, a_min=0.0, a_max=1.0)
        plt.subplot(batch_size, 4, i+2)
        plt.axis("off")
        plt.imshow(B)

        P = make_numpy_grid(predict)
        P = np.clip(P, a_min=0.0, a_max=1.0)
        plt.subplot(batch_size, 4, i+3)
        plt.axis("off")
        plt.imshow(P)

        M = make_numpy_grid(tensor_gt)
        M = np.clip(M, a_min=0.0, a_max=1.0)
        plt.subplot(batch_size, 4, i+4)
        plt.axis("off")
        plt.imshow(M)

    plt.savefig(batch_name)
    plt.show()


def visualize_labeled(tensor_A, tensor_B, predict, tensor_gt=None, ids=[], batch_size=3, name='', dist_dir='unimatch-all-2'):
    batch_name = os.path.join(vis_root, dist_dir, name + '.png')
    predict = _visualize_pred(predict)

    plt.figure(figsize=(10, 6), dpi=340)
    index = 1

    for i in range(batch_size):
        A = make_numpy_grid(tensor_A[i])
        A = np.clip(A, a_min=0.0, a_max=1.0)
        plt.subplot(batch_size, 4, index+0)
        plt.title(ids[i])
        plt.axis("off")
        plt.imshow(A)

        B = make_numpy_grid(tensor_B[i])
        B = np.clip(B, a_min=0.0, a_max=1.0)
        plt.subplot(batch_size, 4, index+1)
        plt.title(ids[i])
        plt.axis("off")
        plt.imshow(B)

        P = make_numpy_grid(predict[i])
        P = np.clip(P, a_min=0.0, a_max=1.0)
        plt.subplot(batch_size, 4, index+2)
        plt.title(ids[i])
        plt.axis("off")
        plt.imshow(P)

        M = make_numpy_grid(tensor_gt[i])
        M = np.clip(M, a_min=0.0, a_max=1.0)
        plt.subplot(batch_size, 4, index+3)
        plt.title(ids[i])
        plt.axis("off")
        plt.imshow(M)

        index = index + 4

    plt.savefig(batch_name)
    plt.close()


def visualize_eval(tensor_A, tensor_B, predict, tensor_gt, base_path, name=''):
    img_margin_inner = np.zeros([256, 5, 3])
    batch_name = os.path.join(base_path, name + '.png')
    batch_size = 1
    predict = _visualize_pred(predict)

    batch_img = np.array([])
    for i in range(batch_size):

        A = make_numpy_grid(tensor_A[i])
        B = make_numpy_grid(tensor_B[i])
        P = make_numpy_grid(predict)
        M = make_numpy_grid(tensor_gt)
        vis = np.concatenate([A, img_margin_inner, B, img_margin_inner, P, img_margin_inner, M], axis=1)
        if batch_img.shape == (0,):
            batch_img = vis
        else:
            batch_img = np.concatenate([batch_img, vis], axis=0)

    batch_img = np.clip(batch_img, a_min=0.0, a_max=1.0)
    plt.imsave(batch_name, batch_img)

def visualize_b2f(tensor_A, tensor_B, predict, tensor_gt, base_path, name=''):
    img_margin_inner = np.zeros([256, 5, 3])
    batch_name = os.path.join(base_path, name + '.png')
    batch_size = 1

    predict = predict * 255

    batch_img = np.array([])
    for i in range(batch_size):

        A = make_numpy_grid(tensor_A[i])
        B = make_numpy_grid(tensor_B[i])
        P = make_numpy_grid(predict)
        M = make_numpy_grid(tensor_gt)
        vis = np.concatenate([A, img_margin_inner, B, img_margin_inner, P, img_margin_inner, M], axis=1)
        if batch_img.shape == (0,):
            batch_img = vis
        else:
            batch_img = np.concatenate([batch_img, vis], axis=0)

    batch_img = np.clip(batch_img, a_min=0.0, a_max=1.0)
    plt.imsave(batch_name, batch_img)

def visualize_single(tensor_A, base_path, name=''):
    batch_name = os.path.join(base_path, name + '.png')
    batch_size = 1

    batch_img = np.array([])
    for i in range(batch_size):

        A = make_numpy_grid(tensor_A[i])
        vis = np.concatenate([A], axis=1)
        if batch_img.shape == (0,):
            batch_img = vis
        else:
            batch_img = np.concatenate([batch_img, vis], axis=0)

    batch_img = np.clip(batch_img, a_min=0.0, a_max=1.0)
    plt.imsave(batch_name, batch_img)

def visualize_single_pred(predict, base_path, name=''):
    batch_name = os.path.join(base_path, name + '.png')
    predict = _visualize_pred(predict)
    batch_size = 1

    batch_img = np.array([])
    for i in range(batch_size):

        P = make_numpy_grid(predict[i])
        vis = np.concatenate([P], axis=1)
        if batch_img.shape == (0,):
            batch_img = vis
        else:
            batch_img = np.concatenate([batch_img, vis], axis=0)

    batch_img = np.clip(batch_img, a_min=0.0, a_max=1.0)
    plt.imsave(batch_name, batch_img)

def visualize_bibr(tensor_A, tensor_B, predict, predict_refined, predict_saved, tensor_gt, base_path, name=''):
    img_margin_inner = np.zeros([256, 5, 3])
    batch_name = os.path.join(base_path, name + '.png')
    batch_size = 1
    predict = _visualize_pred(predict)

    predict_refined = torch.argmax(predict_refined, dim=1, keepdim=True)
    predict_refined = predict_refined * 255

    batch_img = np.array([])
    for i in range(batch_size):

        A = make_numpy_grid(tensor_A[i])
        B = make_numpy_grid(tensor_B[i])
        P = make_numpy_grid(predict)
        PR = make_numpy_grid(predict_refined)
        PS = make_numpy_grid(predict_saved)
        M = make_numpy_grid(tensor_gt)
        vis = np.concatenate([A, img_margin_inner, B, img_margin_inner, P, img_margin_inner, PR, img_margin_inner, PS, img_margin_inner, M], axis=1)
        if batch_img.shape == (0,):
            batch_img = vis
        else:
            batch_img = np.concatenate([batch_img, vis], axis=0)

    batch_img = np.clip(batch_img, a_min=0.0, a_max=1.0)
    plt.imsave(batch_name, batch_img)

def visualize_bibr(tensor_A, tensor_B, predict, predict_saved, base_path, name=''):
    img_margin_inner = np.zeros([256, 5, 3])
    batch_name = os.path.join(base_path, name + '.png')
    batch_size = 1
    predict = _visualize_pred(predict)

    batch_img = np.array([])
    for i in range(batch_size):

        A = make_numpy_grid(tensor_A[i])
        B = make_numpy_grid(tensor_B[i])
        P = make_numpy_grid(predict)
        PS = make_numpy_grid(predict_saved)
        vis = np.concatenate([A, img_margin_inner, B, img_margin_inner, P, img_margin_inner, PS], axis=1)
        if batch_img.shape == (0,):
            batch_img = vis
        else:
            batch_img = np.concatenate([batch_img, vis], axis=0)

    batch_img = np.clip(batch_img, a_min=0.0, a_max=1.0)
    plt.imsave(batch_name, batch_img)

def visualize_bibr_batch(tensor_A, tensor_B, predict, predict_saved, base_path, name=''):
    img_margin_inner = np.zeros([256, 5, 3])
    batch_name = os.path.join(base_path, name + '.png')
    batch_size = 8
    predict = _visualize_pred(predict)

    batch_img = np.array([])
    for i in range(batch_size):

        A = make_numpy_grid(tensor_A[i])
        B = make_numpy_grid(tensor_B[i])
        P = make_numpy_grid(predict[i])
        PS = make_numpy_grid(predict_saved[i])
        vis = np.concatenate([A, img_margin_inner, B, img_margin_inner, P, img_margin_inner, PS], axis=1)
        if batch_img.shape == (0,):
            batch_img = vis
        else:
            batch_img = np.concatenate([batch_img, vis], axis=0)

    batch_img = np.clip(batch_img, a_min=0.0, a_max=1.0)
    plt.imsave(batch_name, batch_img)

def visualize_bibr2(tensor_A, tensor_B, predict, predict_saved, tensor_gt, base_path, name=''):
    img_margin_inner = np.zeros([256, 5, 3])
    batch_name = os.path.join(base_path, name + '.png')
    batch_size = 1

    batch_img = np.array([])
    for i in range(batch_size):

        A = make_numpy_grid(tensor_A[i])
        B = make_numpy_grid(tensor_B[i])
        P = make_numpy_grid(predict)
        PS = make_numpy_grid(predict_saved)
        M = make_numpy_grid(tensor_gt)
        vis = np.concatenate([A, img_margin_inner, B, img_margin_inner, P, img_margin_inner, PS, img_margin_inner, M], axis=1)
        if batch_img.shape == (0,):
            batch_img = vis
        else:
            batch_img = np.concatenate([batch_img, vis], axis=0)

    batch_img = np.clip(batch_img, a_min=0.0, a_max=1.0)
    plt.imsave(batch_name, batch_img)

def visualize_test(tensor_A, tensor_B, pred, gt, boundary_pred, boundary_gt, base_path, name=''):
    img_margin_inner = np.zeros([256, 5, 3])
    batch_name = os.path.join(base_path, name + '.png')
    batch_size = 1

    batch_img = np.array([])
    for i in range(batch_size):

        A = make_numpy_grid(tensor_A[i])
        B = make_numpy_grid(tensor_B[i])
        P = make_numpy_grid(pred)
        M = make_numpy_grid(gt)
        BP = make_numpy_grid(torch.tensor(boundary_pred).unsqueeze(0))
        BG = make_numpy_grid(torch.tensor(boundary_gt).unsqueeze(0))
        vis = np.concatenate([A, img_margin_inner, B, img_margin_inner, P, img_margin_inner, M, img_margin_inner, BP, img_margin_inner, BG], axis=1)
        if batch_img.shape == (0,):
            batch_img = vis
        else:
            batch_img = np.concatenate([batch_img, vis], axis=0)

    batch_img = np.clip(batch_img, a_min=0.0, a_max=1.0)
    plt.imsave(batch_name, batch_img)


def visualize_pseudo_label_confidence(tensor_A, tensor_B, predict, conf, tensor_gt, batch_size=3, names=[], dist_dir='unimatch-conf'):
    batch_name = os.path.join(vis_root, dist_dir, names[0])
    predict = _visualize_pred(predict)

    plt.figure(figsize=(50, 20), dpi=340)

    index = 1

    for i in range(batch_size):
        A = make_numpy_grid(tensor_A[i])
        A = np.clip(A, a_min=0.0, a_max=1.0)
        plt.subplot(batch_size, 5, index+0)
        plt.axis("off")
        plt.imshow(A)

        B = make_numpy_grid(tensor_B[i])
        B = np.clip(B, a_min=0.0, a_max=1.0)
        plt.subplot(batch_size, 5, index+1)
        plt.axis("off")
        plt.imshow(B)

        P = make_numpy_grid(predict[i])
        P = np.clip(P, a_min=0.0, a_max=1.0)
        plt.subplot(batch_size, 5, index+2)
        plt.axis("off")
        plt.imshow(P)

        C = conf[i].cpu().detach().numpy()
        plt.subplot(batch_size, 5, index+3)
        plt.imshow(C, cmap='viridis', interpolation='nearest', aspect='auto')
        plt.colorbar(label='Confidence')

        M = make_numpy_grid(tensor_gt[i])
        M = np.clip(M, a_min=0.0, a_max=1.0)
        plt.subplot(batch_size, 5, index+4)
        plt.axis("off")
        plt.imshow(M)

        index = index + 5
    plt.tight_layout()
    plt.savefig(batch_name)
    plt.close()
    


def visualize_pseudo_label(tensor_A, tensor_B, predict, tensor_gt, H, batch_size=3, name='', dist_dir='unimatch-all-2', is_auged=None):
    batch_name = os.path.join(vis_root, dist_dir, name + '.png')
    predict = _visualize_pred(predict)

    plt.figure(figsize=(30, 10), dpi=340)
    flag = False
    index = 1

    for i in range(batch_size):
        A = make_numpy_grid(tensor_A[i])
        A = np.clip(A, a_min=0.0, a_max=1.0)
        plt.subplot(batch_size, 5, index+0)
        plt.axis("off")
        plt.imshow(A)

        B = make_numpy_grid(tensor_B[i])
        B = np.clip(B, a_min=0.0, a_max=1.0)
        plt.subplot(batch_size, 5, index+1)
        plt.axis("off")
        plt.imshow(B)

        P = make_numpy_grid(predict[i])
        P = np.clip(P, a_min=0.0, a_max=1.0)
        plt.subplot(batch_size, 5, index+2)
        plt.axis("off")
        plt.imshow(P)

        M = make_numpy_grid(tensor_gt[i])
        M = np.clip(M, a_min=0.0, a_max=1.0)
        plt.subplot(batch_size, 5, index+3)
        plt.axis("off")
        plt.imshow(M)

        # pseudo_label = pseudo_labels[i]
        # pseudo_label_1 = pseudo_label[0].cpu().detach().numpy()
        # pseudo_label_2 = pseudo_label[1].cpu().detach().numpy()
        # pred = pred / 255	                                    # 预测结果需要归一化到0~1的区间内
        # eps = 1e-8
        # pseudo_label = np.clip(pseudo_label, eps, 1 - eps)	# 防止计算熵时log0出错
        # H = -pseudo_label * np.log2(pseudo_label) - (1 - pseudo_label) * np.log2(1 - pseudo_label)

        #pseudo_label_1 = -pseudo_label_1
        #pseudo_label_2 = -pseudo_label_2
        # ----------------------------------------------
        # pred_u = pseudo_label[0].detach()
        # logits_u_aug, label_u_aug = torch.max(pred_u, dim=1)

        # eps = 1e-8
        # pred_u = torch.clip(pred_u, eps, 1-eps)

        # H_3 = -(pred_u * torch.log(pred_u))
        # H_3 /= np.log(2)
        # H_3 = H_3.cpu().detach().numpy()
        # # ----------------------------------------------

        H_3 = H[i].cpu().detach().numpy()

        plt.subplot(batch_size, 5, index+4)
        if is_auged[i] == 1:
            plt.title("auged")
        # plt.axis("off")
        # plt.imshow(M)
        plt.imshow(H_3, cmap='hot', interpolation='nearest')
        plt.tick_params(axis='both', which='both', length=0, labelsize=0, labelcolor='w')
        plt.colorbar(shrink=0.6)

        index = index + 5

    # if flag:
    plt.savefig(batch_name)
    #plt.savefig('hotmap.png')
    plt.close()



def visualize_hotmap(predict, name='', dist_dir='unimatch-all-2'):
    hotmap_name_0 = os.path.join(vis_root, dist_dir, name + '-hotmap0.png')
    hotmap_name_1 = os.path.join(vis_root, dist_dir, name + '-hotmap1.png')

    for i in range(2):
        predict = predict.squeeze(0)
        pred = predict[i].cpu().detach().numpy()
        # pred = pred / 255	                # 预测结果需要归一化到0~1的区间内
        eps = 1e-8	
        pred = np.clip(pred, eps, 1 - eps)	# 防止计算熵时log0出错
        H = -pred * np.log2(pred) - (1 - pred) * np.log2(1 - pred)
        t = torch.log(predict + 1e-10)
        t1 = predict * t

        H_2 = -torch.sum(t1, dim=1)

        fig, ax = plt.subplots()
        im = ax.imshow(H, cmap='hot', interpolation='nearest')
        ax.tick_params(axis='both', which='both', length=0, labelsize=0, labelcolor='w')
        plt.colorbar(im, shrink=0.6)

        if i == 0:
            plt.savefig(hotmap_name_0, bbox_inches='tight')
        else:
            plt.savefig(hotmap_name_1, bbox_inches='tight')
        # print("Entropy", H.sum() / pred.size)


def tensor2np(input_image, if_normalize=True):
    """
    :param input_image: C*H*W / H*W
    :return: ndarray, H*W*C / H*W
    """
    if isinstance(input_image, torch.Tensor):  # get the data from a variable
        image_tensor = input_image.data
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array

    else:
        image_numpy = input_image
    if image_numpy.ndim == 2:
        return image_numpy
    elif image_numpy.ndim == 3:
        C, H, W = image_numpy.shape
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        #  如果输入为灰度图C==1，则输出array，ndim==2；
        if C == 1:
            image_numpy = image_numpy[:, :, 0]
        if if_normalize and C == 3:
            image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
            #  add to prevent extreme noises in visual images
            image_numpy[image_numpy<0]=0
            image_numpy[image_numpy>255]=255
            image_numpy = image_numpy.astype(np.uint8)
    return image_numpy


def visulize_features(features, normalize=False):
    """
    可视化特征图，各维度make grid到一起
    """
    from torchvision.utils import make_grid
    assert features.ndim == 4
    b,c,h,w = features.shape
    features = features.view((b*c, 1, h, w))
    # if normalize:
    #     features = norm_tensor(features)
    grid = make_grid(features)
    visualize_tensors(grid)

def visualize_tensors(*tensors):
    """
    可视化tensor，支持单通道特征或3通道图像
    :param tensors: tensor: C*H*W, C=1/3
    :return:
    """
    import matplotlib.pyplot as plt
    # from misc.torchutils import tensor2np
    images = []
    for tensor in tensors:
        assert tensor.ndim == 3 or tensor.ndim==2
        if tensor.ndim ==3:
            assert tensor.shape[0] == 1 or tensor.shape[0] == 3
        images.append(tensor2np(tensor))
    nums = len(images)
    if nums>1:
        fig, axs = plt.subplots(1, nums)
        for i, image in enumerate(images):
            axs[i].imshow(image, cmap='jet')
        plt.show()
    elif nums == 1:
        fig, ax = plt.subplots(1, nums)
        for i, image in enumerate(images):
            ax.imshow(image, cmap='jet')
        plt.show()
    plt.savefig('visualize_tensors.png')

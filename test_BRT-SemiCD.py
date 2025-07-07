import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import torch
import logging
import argparse
import torch.nn.functional as F


from torch.utils.data import DataLoader
from dataset.semicd import SemiCDDataset
from network.deeplabv3plus import DeepLabV3Plus
from util.utils import Evaluator, init_log
from util.dist_helper import setup_distributed

from util.visualize import visualize_test
from tqdm import tqdm


def evaluate(model, loader):
    model.eval()

    evaluator_test = Evaluator(num_class=2)

    with torch.no_grad():
        for imgA, imgB, mask, id, A, B in tqdm(loader):
            id = id[0].split(".")[0]

            imgA = imgA.cuda()
            imgB = imgB.cuda()

            outputs = model(imgA, imgB).argmax(dim=1)

            evaluator_test.add_batch(mask.cpu().numpy(), outputs.cpu().numpy())

    return evaluator_test

def unlabeled_loss(pred_u, mask_u_w_cutmixed, conf_u_w_cutmixed, ignore_mask_cutmixed, criterion_u):
    loss_u = criterion_u(pred_u, mask_u_w_cutmixed) 
    loss_u = loss_u * ((conf_u_w_cutmixed >= 0.95) & (ignore_mask_cutmixed != 255))
    loss_u = loss_u.sum() / (ignore_mask_cutmixed != 255).sum().item()
    return loss_u

def main():
    parser = argparse.ArgumentParser(description='Boundary Refinement Teacher Change Detection for Remote Sensing Images')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
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

    args.save_path = args.save_path + '/' + 'CD'

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    os.environ['LOCAL_RANK'] = '0'
    os.environ["RANK"] = '0'
    os.environ["WORLD_SIZE"] = '1'
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '8064'

    rank, world_size = setup_distributed(port=8084)

    local_rank = int(os.environ["LOCAL_RANK"])

    model = DeepLabV3Plus().cuda()

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    
    valset = SemiCDDataset(args.dataset, args.dataset_root, 'val', args.crop_size, args.val_id_path)

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler)

    if os.path.exists(os.path.join(args.save_path, 'best.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'best.pth'))
        model.load_state_dict(checkpoint['model'])

        if rank == 0:
            logger.info('Load from Change Detection Model checkpoint.')
    else:
        logger.info('Not found Change Detection Model checkpoint.')

    evaluator_test = evaluate(model, valloader)

    IoU = evaluator_test.Intersection_over_Union()[1]
    ACC = evaluator_test.OA()[1]
    Pre = evaluator_test.Precision()[1]
    Recall = evaluator_test.Recall()[1]
    F1 = evaluator_test.F1()[1]
    Kappa = evaluator_test.Kappa()[1]

    if rank == 0:
        logger.info('***** Evaluation ***** >>>> IoU: %.2f, F1: %.2f, Precision: %.2f, Recall: %.2f, OA: %.2f, Kappa: %.4f' % ( IoU * 100, F1 * 100, Pre * 100, Recall * 100, ACC * 100, Kappa))


if __name__ == "__main__":
    # Test Boundary Refinement Teacher (BRT) **WITHOUT** Bi-temporal Image Boundary Refinement (BIBR)
    main()

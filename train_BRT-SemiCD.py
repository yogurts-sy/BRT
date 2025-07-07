import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import torch
import pprint
import logging
import argparse

from torch import nn
from copy import deepcopy
from torch.optim import SGD
from torch.utils.data import DataLoader
from dataset.semicd import SemiCDDataset
from network.deeplabv3plus import DeepLabV3Plus
from network.BoundaryRefinement import boundaries_refine, boundaries_refine_v2
from util.utils import Evaluator, init_log, AverageMeter
from util.dist_helper import setup_distributed


def evaluate(model, loader):
    model.eval()

    evaluator_test = Evaluator(num_class=2)

    with torch.no_grad():
        for imgA, imgB, mask, _, _, _ in loader:

            imgA = imgA.cuda()
            imgB = imgB.cuda()

            outputs = model(imgA, imgB).argmax(dim=1)

            evaluator_test.add_batch(mask.cpu().numpy(), outputs.cpu().numpy())

    return evaluator_test

def evaluate_STPU(model, cd_teacher, cd_teacher_preds, cd_teacher_updated, loader):
    model.eval()
    cd_teacher.eval()

    if len(cd_teacher_preds) == 0:
        cd_teacher_updated = True

    evaluator_test = Evaluator(num_class=2)

    with torch.no_grad():
        for e_id, (imgA, imgB, _, _, _, _) in enumerate(loader):

            imgA = imgA.cuda()
            imgB = imgB.cuda()

            outputs = model(imgA, imgB).argmax(dim=1)

            if cd_teacher_updated:
                mask = cd_teacher(imgA, imgB).argmax(dim=1)
                cd_teacher_preds[str(e_id)] = mask
            else:
                mask = cd_teacher_preds[str(e_id)]

            evaluator_test.add_batch(mask.cpu().numpy(), outputs.cpu().numpy())

    IoU = evaluator_test.Intersection_over_Union()[1]

    return IoU, cd_teacher_preds


def unlabeled_loss(pred_u, mask_u_w_cutmixed, conf_u_w_cutmixed, ignore_mask_cutmixed, criterion_u):
    loss_u = criterion_u(pred_u, mask_u_w_cutmixed) 
    loss_u = loss_u * ((conf_u_w_cutmixed >= 0.95) & (ignore_mask_cutmixed != 255))
    loss_u = loss_u.sum() / (ignore_mask_cutmixed != 255).sum().item()
    return loss_u

def main():
    parser = argparse.ArgumentParser(description='Boundary Refinement Teacher Change Detection for Remote Sensing Images')
    parser.add_argument('--epochs', type=int, default=80, help='epochs')
    parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--crop_size', type=int, default=256, help='training dataset crop size')
    parser.add_argument('--train_ratio', type=float, default=0.40, help='Proportion of the labeled images')
    parser.add_argument('--update', type=str, default='STPU', help='Teacher parameter update scheme')
    parser.add_argument('--update_epoch', type=int, default=2, help='Epoch to Teacher parameter update scheme')
    parser.add_argument('--dataset', type=str, default='WHU', help='training dataset name')
    parser.add_argument('--boundary_path', type=str, default='./exp/WHU/40', help='BRT weights path')
    parser.add_argument('--save_path', type=str, default='./output/')

    args = parser.parse_args()

    # **Replace your dataset path**

    if args.dataset == 'LEVIR':
        # follows SemiCD splits
        args.dataset_root = '/data2/suyou/Codes/datasets/LEVIR-CD256'
        args.train_ratio = str(int(args.train_ratio * 100))
        args.labeled_id_path = './splits' + '/' + args.dataset + '/' + args.train_ratio + '%' + '/' + 'labeled.txt'
        args.unlabeled_id_path = './splits' + '/' + args.dataset + '/' + args.train_ratio + '%' + '/' + 'unlabeled.txt'
        args.val_id_path = './splits' + '/' + args.dataset + '/' + 'test.txt'
    elif args.dataset == 'WHU':
        # follows SemiCD splits
        args.dataset_root = '/data2/suyou/Codes/datasets/WHU-CD256'
        args.train_ratio = str(int(args.train_ratio * 100))
        args.labeled_id_path = './splits' + '/' + args.dataset + '/' + args.train_ratio + '%' + '/' + 'labeled.txt'
        args.unlabeled_id_path = './splits' + '/' + args.dataset + '/' + args.train_ratio + '%' + '/' + 'unlabeled.txt'
        args.val_id_path = './splits' + '/' + args.dataset + '/' + 'test.txt'
    elif args.dataset == 'DSIFN':
        args.dataset_root = '/data2/suyou/Codes/datasets/DSIFN'
        args.train_ratio = str(int(args.train_ratio * 100))
        args.labeled_id_path = './splits' + '/' + args.dataset + '/' + args.train_ratio + '%' + '/' + 'labeled.txt'
        args.unlabeled_id_path = './splits' + '/' + args.dataset + '/' + args.train_ratio + '%' + '/' + 'unlabeled.txt'
        args.val_id_path = './splits' + '/' + args.dataset + '/' + 'val.txt'

    args.save_path = args.save_path + args.dataset + '/' + args.train_ratio + '/'
    args.boundary_path = args.boundary_path + '/' + 'BDN'

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    os.makedirs(args.save_path, exist_ok=True)

    os.environ['LOCAL_RANK'] = '0'
    os.environ["RANK"] = '0'
    os.environ["WORLD_SIZE"] = '1'
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '8061'
    distributed_port = 8081

    rank, world_size = setup_distributed(port=distributed_port)

    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

    local_rank = int(os.environ["LOCAL_RANK"])

    model = DeepLabV3Plus().cuda()

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': args.lr}], lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    br_teacher = DeepLabV3Plus().cuda()  # Boundry Refinement Teacher

    br_teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(br_teacher)
    br_teacher = torch.nn.parallel.DistributedDataParallel(br_teacher, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)
    
    cd_teacher = DeepLabV3Plus().cuda()  # Change Detection Teacher

    cd_teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(cd_teacher)
    cd_teacher = torch.nn.parallel.DistributedDataParallel(cd_teacher, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda()
    criterion_u = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda()

    trainset_u = SemiCDDataset(args.dataset, args.dataset_root, 'train_u', args.crop_size, args.unlabeled_id_path)
    trainset_l = SemiCDDataset(args.dataset, args.dataset_root, 'train_l', args.crop_size, args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiCDDataset(args.dataset, args.dataset_root, 'val', args.crop_size, args.val_id_path)

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * args.epochs
    previous_best_iou, previous_best_acc = 0.0, 0.0
    previous_best_iou_stpu = 0.0
    epoch = -1
    cd_teacher_preds = dict()
    cd_teacher_updated = True

    if os.path.exists(os.path.join(args.boundary_path, 'best.pth')):
        checkpoint = torch.load(os.path.join(args.boundary_path, 'best.pth'))
        br_teacher.load_state_dict(checkpoint['model'])

        if rank == 0:
            logger.info('Load best Boundry Refinement Teacher model from best checkpoint.')
    else:
        if rank == 0:
            logger.warning('Not found Boundry Refinement Teacher model.')
        return


    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']

        previous_best_iou = checkpoint['previous_best_iou']
        previous_best_acc = checkpoint['previous_best_acc']

        if rank == 0:
            logger.info('Load from Change Detection Model checkpoint at epoch %i' % epoch)

    if os.path.exists(os.path.join(args.save_path, 'best.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'best.pth'))
        cd_teacher.load_state_dict(checkpoint['model'])
        best_epoch = checkpoint['epoch']

        if rank == 0:
            logger.info('Load from Change Detection Teacher checkpoint at epoch %i\n' % best_epoch)

    for epoch in range(epoch + 1, args.epochs):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best Changed IoU: {:.2f}, Overall Accuracy: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best_iou * 100, previous_best_acc * 100))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        use_refined = True if epoch > 40 else False

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, ((imgA_x, imgB_x, mask_x),
                (imgA_u_w, imgB_u_w, imgA_u_s1, imgB_u_s1,
                 imgA_u_s2, imgB_u_s2, ignore_mask, cutmix_box1, cutmix_box2, _),
                (imgA_u_w_mix, imgB_u_w_mix, imgA_u_s1_mix,
                 imgB_u_s1_mix, imgA_u_s2_mix, imgB_u_s2_mix, ignore_mask_mix, _, _, _)) in enumerate(loader):
            
            imgA_x, imgB_x, mask_x = imgA_x.cuda(), imgB_x.cuda(), mask_x.cuda()
            imgA_u_w, imgB_u_w = imgA_u_w.cuda(), imgB_u_w.cuda()
            imgA_u_s1, imgB_u_s1 = imgA_u_s1.cuda(), imgB_u_s1.cuda()
            imgA_u_s2, imgB_u_s2 = imgA_u_s2.cuda(), imgB_u_s2.cuda()
            ignore_mask = ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            imgA_u_w_mix, imgB_u_w_mix = imgA_u_w_mix.cuda(), imgB_u_w_mix.cuda()
            imgA_u_s1_mix, imgB_u_s1_mix = imgA_u_s1_mix.cuda(), imgB_u_s1_mix.cuda()
            imgA_u_s2_mix, imgB_u_s2_mix = imgA_u_s2_mix.cuda(), imgB_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            with torch.no_grad():
                if use_refined:
                    cd_teacher.eval()

                    pred_u_w, pred_u_w_mix = cd_teacher(torch.cat((imgA_u_w, imgA_u_w_mix)), torch.cat((imgB_u_w, imgB_u_w_mix))).chunk(2)

                    pred_u_w_A, pred_u_w_B = cd_teacher(torch.cat((imgA_u_w, imgB_u_w)), mode='seg').chunk(2)
                    pred_u_w = boundaries_refine_v2(imgA_u_w, imgB_u_w, br_teacher, pred_u_w.detach(), pred_u_w_A, pred_u_w_B)

                    pred_u_w_mix_A, pred_u_w_mix_B = cd_teacher(torch.cat((imgA_u_w_mix, imgB_u_w_mix)), mode='seg').chunk(2)
                    pred_u_w_mix = boundaries_refine_v2(imgA_u_w_mix, imgB_u_w_mix, br_teacher, pred_u_w_mix.detach(), pred_u_w_mix_A, pred_u_w_mix_B)
                else:
                    model.eval()

                    pred_u_w_mix = model(imgA_u_w_mix, imgB_u_w_mix).detach()

                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            imgA_u_s1[cutmix_box1.unsqueeze(1).expand(imgA_u_s1.shape) == 1] = \
                imgA_u_s1_mix[cutmix_box1.unsqueeze(1).expand(imgA_u_s1.shape) == 1]
            imgB_u_s1[cutmix_box1.unsqueeze(1).expand(imgB_u_s1.shape) == 1] = \
                imgB_u_s1_mix[cutmix_box1.unsqueeze(1).expand(imgB_u_s1.shape) == 1]
            imgA_u_s2[cutmix_box2.unsqueeze(1).expand(imgA_u_s2.shape) == 1] = \
                imgA_u_s2_mix[cutmix_box2.unsqueeze(1).expand(imgA_u_s2.shape) == 1]
            imgB_u_s2[cutmix_box2.unsqueeze(1).expand(imgB_u_s2.shape) == 1] = \
                imgB_u_s2_mix[cutmix_box2.unsqueeze(1).expand(imgB_u_s2.shape) == 1]

            model.train()

            num_lb, num_ulb = imgA_x.shape[0], imgA_u_w.shape[0]

            preds, preds_fp = model(torch.cat((imgA_x, imgA_u_w)), torch.cat((imgB_x, imgB_u_w)), True)
            if use_refined:
                pred_x, _ = preds.split([num_lb, num_ulb])
            else:
                pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            pred_u_s1, pred_u_s2 = model(torch.cat((imgA_u_s1, imgA_u_s2)), torch.cat((imgB_u_s1, imgB_u_s2))).chunk(2)

            mask_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), ignore_mask.clone()
            conf_u_w_cutmixed1 = conf_u_w.clone()
            conf_u_w_cutmixed2 = conf_u_w.clone()

            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            loss_x = criterion_l(pred_x, mask_x)
            loss_u_s1 = unlabeled_loss(pred_u_s1, mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1, criterion_u)
            loss_u_s2 = unlabeled_loss(pred_u_s2, mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2, criterion_u)
            loss_u_w_fp = unlabeled_loss(pred_u_w_fp, mask_u_w, conf_u_w, ignore_mask, criterion_u)                
            
            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())

            iters = epoch * len(trainloader_u) + i
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}'
                            .format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg, total_loss_w_fp.avg))

        evaluator_test = evaluate(model, valloader)

        IoU = evaluator_test.Intersection_over_Union()[1]
        ACC = evaluator_test.OA()[1]
        Pre = evaluator_test.Precision()[1]
        Recall = evaluator_test.Recall()[1]
        F1 = evaluator_test.F1()[1]
        Kappa = evaluator_test.Kappa()[1]

        if rank == 0:
            logger.info('***** Evaluation ***** >>>> IoU: %.2f, F1: %.2f, Precision: %.2f, Recall: %.2f, OA: %.2f, Kappa: %.4f\n' % ( IoU * 100, F1 * 100, Pre * 100, Recall * 100, ACC * 100, Kappa ))

        if args.update == 'STPU' and epoch > args.update_epoch:
            IoU_STPU, cd_teacher_preds = evaluate_STPU(model, cd_teacher, cd_teacher_preds, cd_teacher_updated, valloader)
        else:
            IoU_STPU = IoU

        if IoU > previous_best_iou:
            is_best = True
        elif IoU_STPU > previous_best_iou_stpu:
            is_best = True
        else:
            is_best = False
        previous_best_iou = max(IoU, previous_best_iou)
        previous_best_iou_stpu = max(IoU_STPU, previous_best_iou_stpu)
        if is_best:
            previous_best_acc = ACC
            cd_teacher = deepcopy(model)
            cd_teacher_updated = True
        else:
            cd_teacher_updated = False

        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best_iou': previous_best_iou,
                'previous_best_acc': previous_best_acc,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))

if __name__ == "__main__":
    # Train Boundary Refinement Teacher (BRT) with Bi-temporal Image Boundary Refinement (BIBR)
    main()  
    # python train_BRT-SemiCD.py 2>&1 | tee WHU_40.log

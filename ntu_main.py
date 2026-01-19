import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, epoch_saving, load_checkpoint, generate_text
from datasets.build_ntu120 import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending4
from utils.config import get_config
from models import mspnet_ntu120 as MSPNet
import torchvision.transforms as transforms
from ipdb import set_trace as st
from torch.cuda.amp import GradScaler, autocast

# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=25658 ntu120_main.py \
# --config configs/NTU/NTU120_XSet.yaml \
# --distributed True \
# --accumulation-steps 2 \
# --output /data2/XCLIP/mspnet/output/1211_ntu_loss \
# > /data2/XCLIP/mspnet/output/1211_ntu_loss/1001_fd4_ntu_loss.log 2>&1


#train_idset = [0, 1, 5, 6, 7, 10, 14, 15, 16, 17, 18, 21, 22, 23, 24, 26, 27, 36, 40, 41, 42, 43, 44, 47, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 66, 74, 75, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 90, 91, 92, 93, 94, 95]
def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    #print(torch.full((x.size()[0], num_classes), off_value, device=device))
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', type = str, default = 'configs/NTU/NTU120_XSet.yaml')
    parser.add_argument(
        "--opts",
        help = "Modify config options by adding 'KEY VALUE' pairs. ",
        default = None,
        nargs = '+',
    )
    parser.add_argument('--output', type = str)
    parser.add_argument('--resume', type = str)
    parser.add_argument('--pretrained', type = str)
    parser.add_argument('--only_test', type = bool, default = False)
    parser.add_argument('--batch-size', type = int)
    parser.add_argument('--accumulation-steps', type = int)
    parser.add_argument("--distributed", type = bool, default = False, help = 'local rank for DistributedDataParallel')
    parser.add_argument("--local_rank", type = int, default = -1, help = 'local rank for DistributedDataParallel')
    args = parser.parse_args()
    config = get_config(args)
    return args, config

def main(config): 

    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    model, _ = MSPNet.load(config.MODEL.PRETRAINED, config.MODEL.ARCH, 
                         device="cpu", jit=False, 
                         T=config.DATA.NUM_FRAMES, 
                         droppath=config.MODEL.DROP_PATH_RATE, 
                         use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
                         use_cache=config.MODEL.FIX_TEXT,
                         logger=logger,
                        )
    model = model.cuda()

    mixup_fn = None
    if config.AUG.MIXUP > 0:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = CutmixMixupBlending4(num_classes1=config.DATA.NUM_CLASSES, 
                                        num_classes2=106,
                                       smoothing=config.AUG.LABEL_SMOOTH, 
                                       mixup_alpha=config.AUG.MIXUP, 
                                       cutmix_alpha=config.AUG.CUTMIX, 
                                       switch_prob=config.AUG.MIXUP_SWITCH_PROB)
    elif config.AUG.LABEL_SMOOTH > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    if config.TRAIN.OPT_LEVEL != 'O0':
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)

    
    start_epoch, max_accuracy = 0, 0.0
    acc1 = 0.0
    if config.MODEL.RESUME:
        start_epoch, max_accuracy = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)

    text_labels = generate_text(train_data)
    
    if config.TEST.ONLY_TEST == True:
        acc1, acc1_id = test(val_loader, text_labels, model, config)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}% test videos_id: {acc1_id:.1f}%")
        return
    
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)    
        train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn)
        
        if epoch > 14 and epoch%2==1:
            acc1,acc1_id = validate(val_loader, text_labels, model, config)
            logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%  test videos_id: {acc1_id:.1f}%")

        is_best = acc1 > max_accuracy
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            epoch_saving(config, epoch, model.module, max_accuracy, optimizer, lr_scheduler, logger, config.OUTPUT, is_best)

    config.defrost()
    config.TEST.NUM_CLIP = 4
    config.TEST.NUM_CROP = 3
    config.freeze()
    
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)

    acc1,acc1_id = validate(val_loader, text_labels, model, config)
    logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}% test videos_id: {acc1_id:.1f}%")

def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn):
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()
    
    start = time.time()
    end = time.time()
    
    texts = text_labels.cuda(non_blocking=True)
    for idx, batch_data in enumerate(train_loader):
        merged_batch = {
            'imgs_1': batch_data[0]['imgs'],
            'imgs_2':batch_data[1]['imgs'],
            'imgs_3':batch_data[2]['imgs'],
            'label': batch_data[0]['label'],
            'subject': batch_data[0]['subject']
        }
        # print(merged_batch["imgs_1"].shape)
        
        images_1 = merged_batch["imgs_1"].cuda(non_blocking=True)
        images_2 = merged_batch["imgs_2"].cuda(non_blocking=True)
        images_3 = merged_batch["imgs_3"].cuda(non_blocking=True)

        label_id = merged_batch["label"].cuda(non_blocking=True)
        label_id = label_id.reshape(-1)
        label_p = merged_batch["subject"].cuda(non_blocking=True)
        label_p = label_p.reshape(-1)



        images_1 = images_1.view((-1,config.DATA.NUM_FRAMES,3)+images_1.size()[-2:]) # [2, 8, 3, 224, 224]
        images_2 = images_2.view((-1,config.DATA.NUM_FRAMES,3)+images_2.size()[-2:])
        images_3 = images_3.view((-1,config.DATA.NUM_FRAMES,3)+images_3.size()[-2:])
       

        if mixup_fn is not None:
            images_1, images_2,images_3, label_id, label_p = mixup_fn(images_1, images_2, images_3,label_id,label_p)

        # label_id = one_hot(label_id, num_classes=120, on_value=1, off_value=0, device=label_id.device)
        # label_p = one_hot(label_p, num_classes=106, on_value=1, off_value=0, device=label_p.device)
        
        if texts.shape[0] == 1:
            texts = texts.view(1, -1)

        #images = torch.randn(1,8,3,224,224)
        #images= resize_video_tensor(images, resize_transform)
        
        #with autocast():

        output_ac, output_p = model(images_1,images_2,images_3,texts)
        # print(criterion(output_ac, label_id))
        total_loss = criterion(output_ac, label_id) + 0.1 * criterion(output_p, label_p)
        # total_loss = criterion(output_ac, label_id) 
        total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        if config.TRAIN.OPT_LEVEL != 'O0':
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        tot_loss_meter.update(total_loss.item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    
@torch.no_grad()
def validate(val_loader, text_labels, model, config):
    model.eval()
    
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    acc1_meter_id, acc5_meter_id = AverageMeter(), AverageMeter()
    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):

            merged_batch = {
                        'imgs_1': batch_data[0]['imgs'].cuda(non_blocking=True),
                        'imgs_2':batch_data[1]['imgs'].cuda(non_blocking=True),
                        'imgs_3':batch_data[2]['imgs'].cuda(non_blocking=True),
                        'label': batch_data[0]['label'].cuda(non_blocking=True),
                        'subject': batch_data[0]['subject'].cuda(non_blocking=True)
                    }

                        
            _image_1 = merged_batch["imgs_1"]
            _image_2 = merged_batch["imgs_2"]
            _image_3 = merged_batch["imgs_3"]
            
            label_id = merged_batch["label"]
            label_id = label_id.reshape(-1)
            label_p = merged_batch["subject"]
            label_p = label_p.reshape(-1)
            b, tn, c, h, w = _image_1.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t

            _image_1 = _image_1.view(b, n, t, c, h, w)
            _image_2 = _image_2.view(b, n, t, c, h, w)
            _image_3 = _image_3.view(b, n, t, c, h, w)

            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            tot_similarity_id = torch.zeros((b, 106)).cuda()
            for i in range(n):   
                image_1 = _image_1[:, i, :, :, :, :] # [b,t,c,h,w]
                image_2 = _image_2[:, i, :, :, :, :] # [b,t,c,h,w]
                image_3 = _image_3[:, i, :, :, :, :] # [b,t,c,h,w]


                label_id = label_id
                label_p = label_p
                image_input_1 = image_1
                image_input_2 = image_2
                image_input_3 = image_3


                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()

                output_ac, output_p = model(image_input_1,image_input_2,image_input_3,text_inputs)

                similarity = output_ac.view(b, -1).softmax(dim=-1)
                similarity_id = output_p.view(b, -1).softmax(dim=-1)
                #tot=total
                tot_similarity += similarity
                tot_similarity_id += similarity_id

            values_1, indices_1 = tot_similarity.topk(1, dim=-1)
            values_5, indices_5 = tot_similarity.topk(5, dim=-1)

            values_1_id, indices_1_id = tot_similarity_id.topk(1, dim=-1)
            values_5_id, indices_5_id = tot_similarity_id.topk(5, dim=-1)
            acc1, acc5 = 0, 0
            acc1_id, acc5_id = 0, 0

            for i in range(b):
                if indices_1[i] == label_id[i]:
                    acc1 += 1
                if label_id[i] in indices_5[i]:
                    acc5 += 1

            for i in range(b):
                if indices_1_id[i] == label_p[i]:
                    acc1_id += 1
                if label_p[i] in indices_5_id[i]:
                    acc5_id += 1

                        
            
            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)
            acc1_meter_id.update(float(acc1_id) / b * 100, b)
            acc5_meter_id.update(float(acc5_id) / b * 100, b)

            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc1_meter.avg:.3f}\t'
                    f'Acc_id@1: {acc1_meter_id.avg:.3f}\t'
                )
    acc1_meter.sync()
    acc5_meter.sync()
    acc1_meter_id.sync()
    acc5_meter_id.sync()
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f} Acc_id@1 {acc1_meter_id.avg:.3f} Acc_id@5 {acc5_meter_id.avg:.3f}')
    return acc1_meter.avg, acc1_meter_id.avg

@torch.no_grad()
def test(val_loader, text_labels, model, config):
    model.eval()
    
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    acc1_meter_id, acc5_meter_id = AverageMeter(), AverageMeter()
    accuracy_dict = {}
    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):

            merged_batch = {
                        'imgs_1': batch_data[0]['imgs'].cuda(non_blocking=True),
                        'imgs_2':batch_data[1]['imgs'].cuda(non_blocking=True),
                        'imgs_3':batch_data[2]['imgs'].cuda(non_blocking=True),
                        'label': batch_data[0]['label'].cuda(non_blocking=True),
                        'subject': batch_data[0]['subject'].cuda(non_blocking=True)
                    }

            _image_1 = merged_batch["imgs_1"]
            _image_2 = merged_batch["imgs_2"]
            _image_3 = merged_batch["imgs_3"]

            label_id = merged_batch["label"]
            label_id = label_id.reshape(-1)
            label_p = merged_batch["subject"]
            label_p = label_p.reshape(-1)

            b, tn, c, h, w = _image_1.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t
            _image_1 = _image_1.view(b, n, t, c, h, w)
            _image_2 = _image_2.view(b, n, t, c, h, w)
            _image_3 = _image_3.view(b, n, t, c, h, w)

            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            tot_similarity_id = torch.zeros((b, 106)).cuda()
            for i in range(n):   
                image_1 = _image_1[:, i, :, :, :, :] # [b,t,c,h,w]
                image_2 = _image_2[:, i, :, :, :, :] # [b,t,c,h,w]
                image_3 = _image_3[:, i, :, :, :, :] # [b,t,c,h,w]

                label_id = label_id.cuda(non_blocking=True)
                label_p = label_p.cuda(non_blocking=True)
                image_input_1 = image_1.cuda(non_blocking=True)
                image_input_2 = image_2.cuda(non_blocking=True)
                image_input_3 = image_3.cuda(non_blocking=True)

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()

                output_ac, output_p = model(image_input_1,image_input_2,image_input_3, text_inputs)

                similarity = output_ac.view(b, -1).softmax(dim=-1)
                similarity_id = output_p.view(b, -1).softmax(dim=-1)
                tot_similarity += similarity
                tot_similarity_id += similarity_id


            values_1, indices_1 = tot_similarity.topk(1, dim=-1)
            values_5, indices_5 = tot_similarity.topk(5, dim=-1)

            values_1_id, indices_1_id = tot_similarity_id.topk(1, dim=-1)
            values_5_id, indices_5_id = tot_similarity_id.topk(5, dim=-1)
            acc1, acc5 = 0, 0
            acc1_id, acc5_id = 0, 0

            for i in range(b):
                if indices_1[i] == label_id[i]:
                    acc1 += 1
                if label_id[i] in indices_5[i]:
                    acc5 += 1

            for i in range(b):
                if indices_1_id[i] == label_p[i]:
                    acc1_id += 1
                if label_p[i] in indices_5_id[i]:
                    acc5_id += 1

            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)

            acc1_meter_id.update(float(acc1_id) / b * 100, b)
            acc5_meter_id.update(float(acc5_id) / b * 100, b)

            for i in range(b):
                action_label = label_id[i].item()
                person_id = label_p[i].item()
                correct = (indices_1[i] == label_id[i]).item()
                key = (action_label, person_id)
                if key not in accuracy_dict:
                    accuracy_dict[key] = {'correct': 0, 'total': 0}
                if correct:
                    accuracy_dict[key]['correct'] += 1
                accuracy_dict[key]['total'] += 1

            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc1_meter.avg:.3f}\t'
                    f'Acc_id@1: {acc1_meter_id.avg:.3f}\t'
                )
        acc1_meter.sync()
        acc5_meter.sync()
        acc1_meter_id.sync()
        acc5_meter_id.sync()
        logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
        logger.info(f' * Acc_id@1 {acc1_meter_id.avg:.3f} Acc_id@5 {acc5_meter_id.avg:.3f}')

        return acc1_meter.avg, acc1_meter_id.avg



if __name__ == '__main__':
    args, config = parse_option()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[args.local_rank])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")
    
    if (dist.get_rank() == 0) and (not os.path.exists(args.output)):
        os.makedirs(args.output)

    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config)


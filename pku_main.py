import os
import torch
import torch.nn as nn 
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, epoch_saving, load_checkpoint, generate_text
from datasets.build_pku import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending4_pku
from utils.config import get_config
from models import mspnet_pku as MSPNet
import torchvision.transforms as transforms
from ipdb import set_trace as st
from torch.cuda.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn


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
        mixup_fn = CutmixMixupBlending4_pku(num_classes1=config.DATA.NUM_CLASSES, 
                                        num_classes2=22,
                                        num_uuu=2,
                                        num_vvv=5,
                                        num_www=2,
                                        num_xxx=9,
                                        num_yyy=12,
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

    ##########将所有分类标签编码嵌入
    text_labels = generate_text(train_data)
    #测试
    if config.TEST.ONLY_TEST == True:
        acc1,acc1_id,acc1_uuu,acc1_vvv,acc1_www,acc1_xxx,acc1_yyy = test(val_loader, text_labels, model, config)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}% test videos_id: {acc1_id:.1f}%")
        return
    #训练
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)    

        train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn)
        #每轮验证
        if epoch > 0 :
            acc1,acc1_id,acc1_uuu,acc1_vvv,acc1_www,acc1_xxx,acc1_yyy = validate(val_loader, text_labels, model, config)
            logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%  test videos_id: {acc1_id:.1f}% Acc_uuu@1: {acc1_uuu:.3f} Acc_vvv@1: {acc1_vvv:.3f} Acc_www@1: {acc1_www:.3f}  Acc_xxx@1: {acc1_xxx:.3f}  Acc_yyy@1: {acc1_yyy:.3f}")

   
        is_best = acc1 > max_accuracy
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)) and epoch%5==1:
            epoch_saving(config, epoch, model.module, max_accuracy, optimizer, lr_scheduler, logger, config.OUTPUT, is_best)

    config.defrost()
    config.TEST.NUM_CLIP = 4
    config.TEST.NUM_CROP = 3
    config.freeze()
    
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    #验证
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
            'subject': batch_data[0]['subject'],
            'uuu': batch_data[0]['uuu'],
            'vvv': batch_data[0]['vvv'],
            'www': batch_data[0]['www'],
            'xxx': batch_data[0]['xxx'],
            'yyy': batch_data[0]['yyy']            
        }
        # print(merged_batch["imgs_1"].shape)
        
        images_1 = merged_batch["imgs_1"].cuda(non_blocking=True)
        images_2 = merged_batch["imgs_2"].cuda(non_blocking=True)
        images_3 = merged_batch["imgs_3"].cuda(non_blocking=True)

        label_id = merged_batch["label"].cuda(non_blocking=True)
        label_id = label_id.reshape(-1)
        label_id_id = merged_batch["subject"].cuda(non_blocking=True)
        label_id_id = label_id_id.reshape(-1)
        # print(label_id_id[0])
        # print(label_id_id[0] in train_idset)

        label_id_uuu = merged_batch["uuu"].cuda(non_blocking=True)
        label_id_uuu = label_id_uuu.reshape(-1)
        label_id_vvv = merged_batch["vvv"].cuda(non_blocking=True)
        label_id_vvv = label_id_vvv.reshape(-1)
        label_id_www = merged_batch["www"].cuda(non_blocking=True)
        label_id_www = label_id_www.reshape(-1)
        label_id_xxx = merged_batch["xxx"].cuda(non_blocking=True)
        label_id_xxx = label_id_xxx.reshape(-1)
        label_id_yyy = merged_batch["yyy"].cuda(non_blocking=True)
        label_id_yyy = label_id_yyy.reshape(-1)

        # st()
        #调整后（高度，宽度，时间维度，3通道）加上（原图像的高度，宽度）？？？？在干嘛？
        images_1 = images_1.view((-1,config.DATA.NUM_FRAMES,3)+images_1.size()[-2:]) # [2, 8, 3, 224, 224]
        images_2 = images_2.view((-1,config.DATA.NUM_FRAMES,3)+images_2.size()[-2:])
        images_3 = images_3.view((-1,config.DATA.NUM_FRAMES,3)+images_3.size()[-2:])
       
        ###############
        if mixup_fn is not None:
            images_1, images_2,images_3, label_id, label_id_id ,label_id_uuu,label_id_vvv, label_id_www,label_id_xxx,label_id_yyy = mixup_fn(images_1, images_2, images_3,label_id,label_id_id,label_id_uuu,label_id_vvv, label_id_www,label_id_xxx,label_id_yyy)

        # label_id = one_hot(label_id, num_classes=20, on_value=1, off_value=0, device=label_id.device)
        # label_id_id = one_hot(label_id_id, num_classes=22, on_value=1, off_value=0, device=label_id_id.device)
        
        if texts.shape[0] == 1:
            texts = texts.view(1, -1)

        #images = torch.randn(1,8,3,224,224)
        #images= resize_video_tensor(images, resize_transform)
        
        #with autocast():

        output_ac, output_id ,output_uuu,output_vvv,output_www,output_xxx,output_yyy = model(images_1,images_2,images_3,texts)

        # total_loss = criterion(output_ac, label_id)
        total_loss = criterion(output_ac, label_id) + 0.2*criterion(output_id, label_id_id)+0.05*criterion(output_uuu, label_id_uuu)+0.05*criterion(output_vvv, label_id_vvv)+0.05*criterion(output_www, label_id_www)+0.05*criterion(output_xxx, label_id_xxx)+0.05*criterion(output_yyy, label_id_yyy)
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

    acc1_meter_uuu, acc5_meter_uuu = AverageMeter(), AverageMeter()
    acc1_meter_vvv, acc5_meter_vvv = AverageMeter(), AverageMeter()
    acc1_meter_www, acc5_meter_www = AverageMeter(), AverageMeter()
    acc1_meter_xxx, acc5_meter_xxx = AverageMeter(), AverageMeter()
    acc1_meter_yyy, acc5_meter_yyy = AverageMeter(), AverageMeter()
    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):
            ### 合并三个 dict
            # merged_batch = {
                # 'imgs': torch.stack([batch_data[0]['imgs'], batch_data[1]['imgs']], dim=1),
                # 'label': batch_data[0]['label'],
                # 'subject': batch_data[0]['subject']
            # }
            merged_batch = {
                        'imgs_1': batch_data[0]['imgs'].cuda(non_blocking=True),
                        'imgs_2':batch_data[1]['imgs'].cuda(non_blocking=True),
                        'imgs_3':batch_data[2]['imgs'].cuda(non_blocking=True),
                        'label': batch_data[0]['label'].cuda(non_blocking=True),
                        'subject': batch_data[0]['subject'].cuda(non_blocking=True),
                        'uuu': batch_data[0]['uuu'].cuda(non_blocking=True),
                        'vvv': batch_data[0]['vvv'].cuda(non_blocking=True),
                        'www': batch_data[0]['www'].cuda(non_blocking=True),
                        'xxx': batch_data[0]['xxx'].cuda(non_blocking=True),
                        'yyy': batch_data[0]['yyy'].cuda(non_blocking=True)   
                    }

                        
            _image_1 = merged_batch["imgs_1"]
            _image_2 = merged_batch["imgs_2"]
            _image_3 = merged_batch["imgs_3"]
            
            label_id = merged_batch["label"]
            label_id = label_id.reshape(-1)
            label_id_id = merged_batch["subject"]
            label_id_id = label_id_id.reshape(-1)

            label_id_uuu = merged_batch["uuu"]
            label_id_uuu = label_id_uuu.reshape(-1)
            label_id_vvv = merged_batch["vvv"]
            label_id_vvv = label_id_vvv.reshape(-1)
            label_id_www = merged_batch["www"]
            label_id_www = label_id_www.reshape(-1)
            label_id_xxx = merged_batch["xxx"]
            label_id_xxx = label_id_xxx.reshape(-1)
            label_id_yyy = merged_batch["yyy"]
            label_id_yyy = label_id_yyy.reshape(-1)


            b, tn, c, h, w = _image_1.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t

            _image_1 = _image_1.view(b, n, t, c, h, w)
            _image_2 = _image_2.view(b, n, t, c, h, w)
            _image_3 = _image_3.view(b, n, t, c, h, w)

            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            tot_similarity_id = torch.zeros((b,22)).cuda()

            tot_similarity_uuu = torch.zeros((b,2)).cuda()
            tot_similarity_vvv = torch.zeros((b,5)).cuda()
            tot_similarity_www = torch.zeros((b,2)).cuda()
            tot_similarity_xxx = torch.zeros((b,9)).cuda()
            tot_similarity_yyy = torch.zeros((b,12)).cuda()
            for i in range(n):   
                image_1 = _image_1[:, i, :, :, :, :] # [b,t,c,h,w]
                image_2 = _image_2[:, i, :, :, :, :] # [b,t,c,h,w]
                image_3 = _image_3[:, i, :, :, :, :] # [b,t,c,h,w]


                label_id = label_id
                label_id_id = label_id_id
                image_input_1 = image_1
                image_input_2 = image_2
                image_input_3 = image_3


                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()

                #image_input= resize_video_tensor(image_input, resize_transform)    
                #image_input = torch.randn(1,8,3,224,224)
                output_ac, output_id,output_uuu,output_vvv,output_www,output_xxx,output_yyy = model(image_input_1,image_input_2,image_input_3,text_inputs)

                similarity = output_ac.view(b, -1).softmax(dim=-1)
                similarity_id = output_id.view(b, -1).softmax(dim=-1)

                similarity_uuu = output_uuu.view(b, -1).softmax(dim=-1)
                similarity_vvv = output_vvv.view(b, -1).softmax(dim=-1)
                similarity_www = output_www.view(b, -1).softmax(dim=-1)
                similarity_xxx = output_xxx.view(b, -1).softmax(dim=-1)
                similarity_yyy = output_yyy.view(b, -1).softmax(dim=-1)

                #tot=total
                tot_similarity += similarity
                tot_similarity_id += similarity_id
                tot_similarity_uuu += similarity_uuu
                tot_similarity_vvv += similarity_vvv
                tot_similarity_www += similarity_www
                tot_similarity_xxx += similarity_xxx
                tot_similarity_yyy += similarity_yyy
             

            values_1, indices_1 = tot_similarity.topk(1, dim=-1)
            values_5, indices_5 = tot_similarity.topk(5, dim=-1)

            values_1_id, indices_1_id = tot_similarity_id.topk(1, dim=-1)
            values_5_id, indices_5_id = tot_similarity_id.topk(5, dim=-1)

            values_1_uuu, indices_1_uuu = tot_similarity_uuu.topk(1, dim=-1)
            values_5_uuu, indices_5_uuu = tot_similarity_uuu.topk(2, dim=-1)

            values_1_vvv, indices_1_vvv = tot_similarity_vvv.topk(1, dim=-1)
            values_5_vvv, indices_5_vvv = tot_similarity_vvv.topk(5, dim=-1)

            values_1_www, indices_1_www = tot_similarity_www.topk(1, dim=-1)
            values_5_www, indices_5_www = tot_similarity_www.topk(2, dim=-1)

            values_1_xxx, indices_1_xxx = tot_similarity_xxx.topk(1, dim=-1)
            values_5_xxx, indices_5_xxx = tot_similarity_xxx.topk(5, dim=-1)

            values_1_yyy, indices_1_yyy = tot_similarity_yyy.topk(1, dim=-1)
            values_5_yyy, indices_5_yyy = tot_similarity_yyy.topk(5, dim=-1)

            acc1, acc5 = 0, 0
            acc1_id, acc5_id = 0, 0
            acc1_uuu, acc5_uuu = 0, 0
            acc1_vvv, acc5_vvv = 0, 0
            acc1_www, acc5_www = 0, 0
            acc1_xxx, acc5_xxx = 0, 0
            acc1_yyy, acc5_yyy = 0, 0

            for i in range(b):
                if indices_1[i] == label_id[i]:
                    acc1 += 1
                if label_id[i] in indices_5[i]:
                    acc5 += 1

            for i in range(b):
                if indices_1_id[i] == label_id_id[i]:
                    acc1_id += 1
                if label_id_id[i] in indices_5_id[i]:
                    acc5_id += 1

            for i in range(b):
                if indices_1_uuu[i] == label_id_uuu[i]:
                    acc1_uuu += 1
                if label_id_uuu[i] in indices_5_uuu[i]:
                    acc5_uuu += 1

            for i in range(b):
                if indices_1_vvv[i] == label_id_vvv[i]:
                    acc1_vvv += 1
                if label_id_vvv[i] in indices_5_vvv[i]:
                    acc5_vvv += 1

            for i in range(b):
                if indices_1_www[i] == label_id_www[i]:
                    acc1_www += 1
                if label_id_www[i] in indices_5_www[i]:
                    acc5_www += 1

            for i in range(b):
                if indices_1_xxx[i] == label_id_xxx[i]:
                    acc1_xxx += 1
                if label_id_xxx[i] in indices_5_xxx[i]:
                    acc5_xxx += 1

            for i in range(b):
                if indices_1_yyy[i] == label_id_yyy[i]:
                    acc1_yyy += 1
                if label_id_yyy[i] in indices_5_yyy[i]:
                    acc5_yyy += 1


                        
            
            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)
            acc1_meter_id.update(float(acc1_id) / b * 100, b)
            acc5_meter_id.update(float(acc5_id) / b * 100, b)

            acc1_meter_uuu.update(float(acc1_uuu) / b * 100, b)
            acc5_meter_uuu.update(float(acc5_uuu) / b * 100, b)
            acc1_meter_vvv.update(float(acc1_vvv) / b * 100, b)
            acc5_meter_vvv.update(float(acc5_vvv) / b * 100, b)
            acc1_meter_www.update(float(acc1_www) / b * 100, b)
            acc5_meter_www.update(float(acc5_www) / b * 100, b)
            acc1_meter_xxx.update(float(acc1_xxx) / b * 100, b)
            acc5_meter_xxx.update(float(acc5_xxx) / b * 100, b)
            acc1_meter_yyy.update(float(acc1_yyy) / b * 100, b)
            acc5_meter_yyy.update(float(acc5_yyy) / b * 100, b)

            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc1_meter.avg:.3f}\t'
                    f'Acc_id@1: {acc1_meter_id.avg:.3f}\t'
                    f'Acc_uuu@1: {acc1_meter_uuu.avg:.3f}\t'
                    f'Acc_vvv@1: {acc1_meter_vvv.avg:.3f}\t'
                    f'Acc_www@1: {acc1_meter_www.avg:.3f}\t'
                    f'Acc_xxx@1: {acc1_meter_xxx.avg:.3f}\t'
                    f'Acc_yyy@1: {acc1_meter_yyy.avg:.3f}\t'                                       
                )
    acc1_meter.sync()
    acc5_meter.sync()
    acc1_meter_id.sync()
    acc5_meter_id.sync()

    acc1_meter_uuu.sync()
    acc5_meter_uuu.sync()
    acc1_meter_vvv.sync()
    acc5_meter_vvv.sync()
    acc1_meter_www.sync()
    acc5_meter_www.sync()
    acc1_meter_xxx.sync()
    acc5_meter_xxx.sync()
    acc1_meter_yyy.sync()
    acc5_meter_yyy.sync()


    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f} Acc_id@1 {acc1_meter_id.avg:.3f} Acc_id@5 {acc5_meter_id.avg:.3f} Acc_uuu@1: {acc1_meter_uuu.avg:.3f} Acc_vvv@1: {acc1_meter_vvv.avg:.3f} Acc_www@1: {acc1_meter_www.avg:.3f}  Acc_xxx@1: {acc1_meter_xxx.avg:.3f}  Acc_yyy@1: {acc1_meter_yyy.avg:.3f}'  )
    return acc1_meter.avg, acc1_meter_id.avg, acc1_meter_uuu.avg, acc1_meter_vvv.avg, acc1_meter_www.avg, acc1_meter_xxx.avg, acc1_meter_yyy.avg


@torch.no_grad() 
def test(val_loader, text_labels, model, config):
    model.eval()
    
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    acc1_meter_id, acc5_meter_id = AverageMeter(), AverageMeter()

    acc1_meter_uuu, acc5_meter_uuu = AverageMeter(), AverageMeter()
    acc1_meter_vvv, acc5_meter_vvv = AverageMeter(), AverageMeter()
    acc1_meter_www, acc5_meter_www = AverageMeter(), AverageMeter()
    acc1_meter_xxx, acc5_meter_xxx = AverageMeter(), AverageMeter()
    acc1_meter_yyy, acc5_meter_yyy = AverageMeter(), AverageMeter()

    # Initialize accuracy dictionaries for each task
    accuracy_dict_action = {}
    accuracy_dict_person = {}
    accuracy_dict_uuu = {}
    accuracy_dict_vvv = {}
    accuracy_dict_www = {}
    accuracy_dict_xxx = {}
    accuracy_dict_yyy = {}

    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):

            merged_batch = {
                        'imgs_1': batch_data[0]['imgs'].cuda(non_blocking=True),
                        'imgs_2': batch_data[1]['imgs'].cuda(non_blocking=True),
                        'imgs_3':batch_data[2]['imgs'].cuda(non_blocking=True),
                        'label': batch_data[0]['label'].cuda(non_blocking=True),
                        'subject': batch_data[0]['subject'].cuda(non_blocking=True),
                        'uuu': batch_data[0]['uuu'].cuda(non_blocking=True),
                        'vvv': batch_data[0]['vvv'].cuda(non_blocking=True),
                        'www': batch_data[0]['www'].cuda(non_blocking=True),
                        'xxx': batch_data[0]['xxx'].cuda(non_blocking=True),        
                        'yyy': batch_data[0]['yyy'].cuda(non_blocking=True),               
                        'camera': batch_data[0]['camera'].cuda(non_blocking=True),          
                    }

            _image_1 = merged_batch["imgs_1"]
            _image_2 = merged_batch["imgs_2"]
            _image_3 = merged_batch["imgs_3"]

            label_id = merged_batch["label"]
            label_id = label_id.reshape(-1)
            label_id_id = merged_batch["subject"]
            label_id_id = label_id_id.reshape(-1)

            label_id_uuu = merged_batch["uuu"]
            label_id_uuu = label_id_uuu.reshape(-1)

            label_id_vvv = merged_batch["vvv"]
            label_id_vvv = label_id_vvv.reshape(-1)

            label_id_www = merged_batch["www"]
            label_id_www = label_id_www.reshape(-1)

            label_id_xxx = merged_batch["xxx"]
            label_id_xxx = label_id_xxx.reshape(-1)

            label_id_yyy = merged_batch["yyy"]
            label_id_yyy = label_id_yyy.reshape(-1)

            label_id_camera = merged_batch["camera"]
            label_id_camera = label_id_camera.reshape(-1)

            b, tn, c, h, w = _image_1.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t

            _image_1 = _image_1.view(b, n, t, c, h, w)
            _image_2 = _image_2.view(b, n, t, c, h, w)
            _image_3 = _image_3.view(b, n, t, c, h, w)
           
            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            tot_similarity_id = torch.zeros((b, 22)).cuda()
            tot_similarity_uuu = torch.zeros((b,2)).cuda()
            tot_similarity_vvv = torch.zeros((b,5)).cuda()
            tot_similarity_www = torch.zeros((b,2)).cuda()
            tot_similarity_xxx = torch.zeros((b,9)).cuda()
            tot_similarity_yyy = torch.zeros((b,12)).cuda()

            for i in range(n):   
                image_1 = _image_1[:, i, :, :, :, :] # [b,t,c,h,w]
                image_2 = _image_2[:, i, :, :, :, :] # [b,t,c,h,w]
                image_3 = _image_3[:, i, :, :, :, :] # [b,t,c,h,w]

                label_id = label_id.cuda(non_blocking=True)
                label_id_id = label_id_id.cuda(non_blocking=True)
                
                image_input_1 = image_1.cuda(non_blocking=True)
                image_input_2 = image_2.cuda(non_blocking=True)
                image_input_3 = image_3.cuda(non_blocking=True)

                label_id_uuu = label_id_uuu.cuda(non_blocking=True)
                label_id_vvv = label_id_vvv.cuda(non_blocking=True)                
                label_id_www = label_id_www.cuda(non_blocking=True)
                label_id_xxx = label_id_xxx.cuda(non_blocking=True)
                label_id_yyy = label_id_yyy.cuda(non_blocking=True)                

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input_1 = image_input_1.half()
                
                output_ac, output_id, output_uuu, output_vvv, output_www, output_xxx, output_yyy = model(image_input_1, image_input_2, image_input_3, text_inputs)

                # Compute softmax probabilities
                similarity = output_ac.view(b, -1).softmax(dim=-1)
                similarity_id = output_id.view(b, -1).softmax(dim=-1)
                similarity_uuu = output_uuu.view(b, -1).softmax(dim=-1)
                similarity_vvv = output_vvv.view(b, -1).softmax(dim=-1)
                similarity_www = output_www.view(b, -1).softmax(dim=-1)
                similarity_xxx = output_xxx.view(b, -1).softmax(dim=-1)
                similarity_yyy = output_yyy.view(b, -1).softmax(dim=-1)

                # Accumulate similarities
                tot_similarity += similarity
                tot_similarity_id += similarity_id
                tot_similarity_uuu += similarity_uuu
                tot_similarity_vvv += similarity_vvv
                tot_similarity_www += similarity_www
                tot_similarity_xxx += similarity_xxx
                tot_similarity_yyy += similarity_yyy

            if(idx == 0):
                score_list = tot_similarity.data.cpu()
                score_list_id = tot_similarity_id.data.cpu()
                score_list_uuu = tot_similarity_uuu.data.cpu()
                score_list_vvv = tot_similarity_vvv.data.cpu()
                score_list_www = tot_similarity_www.data.cpu()
                score_list_xxx = tot_similarity_xxx.data.cpu()
                score_list_yyy = tot_similarity_yyy.data.cpu()
            else:
                score_list = torch.concat((score_list, tot_similarity.data.cpu()), dim = 0)
                score_list_id = torch.concat((score_list_id, tot_similarity_id.data.cpu()), dim = 0)
                score_list_uuu = torch.concat((score_list_uuu, tot_similarity_uuu.data.cpu()), dim = 0)
                score_list_vvv = torch.concat((score_list_vvv, tot_similarity_vvv.data.cpu()), dim = 0)
                score_list_www = torch.concat((score_list_www, tot_similarity_www.data.cpu()), dim = 0)
                score_list_xxx = torch.concat((score_list_xxx, tot_similarity_xxx.data.cpu()), dim = 0)
                score_list_yyy = torch.concat((score_list_yyy, tot_similarity_yyy.data.cpu()), dim = 0)
            
            values_1, indices_1 = tot_similarity.topk(1, dim=-1)
            values_5, indices_5 = tot_similarity.topk(5, dim=-1)

            values_1_id, indices_1_id = tot_similarity_id.topk(1, dim=-1)
            values_5_id, indices_5_id = tot_similarity_id.topk(5, dim=-1)

            values_1_uuu, indices_1_uuu = tot_similarity_uuu.topk(1, dim=-1)
            values_5_uuu, indices_5_uuu = tot_similarity_uuu.topk(2, dim=-1)

            values_1_vvv, indices_1_vvv = tot_similarity_vvv.topk(1, dim=-1)
            values_5_vvv, indices_5_vvv = tot_similarity_vvv.topk(5, dim=-1)

            values_1_www, indices_1_www = tot_similarity_www.topk(1, dim=-1)
            values_5_www, indices_5_www = tot_similarity_www.topk(2, dim=-1)

            values_1_xxx, indices_1_xxx = tot_similarity_xxx.topk(1, dim=-1)
            values_5_xxx, indices_5_xxx = tot_similarity_xxx.topk(5, dim=-1)

            values_1_yyy, indices_1_yyy = tot_similarity_yyy.topk(1, dim=-1)
            values_5_yyy, indices_5_yyy = tot_similarity_yyy.topk(5, dim=-1)

            acc1, acc5 = 0, 0
            acc1_id, acc5_id = 0, 0
            acc1_uuu, acc5_uuu = 0, 0
            acc1_vvv, acc5_vvv = 0, 0
            acc1_www, acc5_www = 0, 0
            acc1_xxx, acc5_xxx = 0, 0
            acc1_yyy, acc5_yyy = 0, 0
            

            for i in range(b):
                if indices_1[i] == label_id[i]:
                    acc1 += 1
                if label_id[i] in indices_5[i]:
                    acc5 += 1

                if indices_1_id[i] == label_id_id[i]:
                    acc1_id += 1
                if label_id_id[i] in indices_5_id[i]:
                    acc5_id += 1

                if indices_1_uuu[i] == label_id_uuu[i]:
                    acc1_uuu += 1
                if label_id_uuu[i] in indices_5_uuu[i]:
                    acc5_uuu += 1

                if indices_1_vvv[i] == label_id_vvv[i]:
                    acc1_vvv += 1
                if label_id_vvv[i] in indices_5_vvv[i]:
                    acc5_vvv += 1

                if indices_1_www[i] == label_id_www[i]:
                    acc1_www += 1
                if label_id_www[i] in indices_5_www[i]:
                    acc5_www += 1

                if indices_1_xxx[i] == label_id_xxx[i]:
                    acc1_xxx += 1
                if label_id_xxx[i] in indices_5_xxx[i]:
                    acc5_xxx += 1

                if indices_1_yyy[i] == label_id_yyy[i]:
                    acc1_yyy += 1
                if label_id_yyy[i] in indices_5_yyy[i]:
                    acc5_yyy += 1

                # Update accuracy_dicts
                action_label = label_id[i].item()
                person_id = label_id_id[i].item()
                uuu_label = label_id_uuu[i].item()
                vvv_label = label_id_vvv[i].item()
                www_label = label_id_www[i].item()
                xxx_label = label_id_xxx[i].item()
                yyy_label = label_id_yyy[i].item()
                camera_id = label_id_camera[i].item()

                correct_action = (indices_1[i] == label_id[i]).item()
                correct_person = (indices_1_id[i] == label_id_id[i]).item()
                correct_uuu = (indices_1_uuu[i] == label_id_uuu[i]).item()
                correct_vvv = (indices_1_vvv[i] == label_id_vvv[i]).item()
                correct_www = (indices_1_www[i] == label_id_www[i]).item()
                correct_xxx = (indices_1_xxx[i] == label_id_xxx[i]).item()
                correct_yyy = (indices_1_yyy[i] == label_id_yyy[i]).item()

                key_action = (action_label, camera_id)
                key_person = (person_id, camera_id)
                key_uuu = (uuu_label, camera_id)
                key_vvv = (vvv_label, camera_id)
                key_www = (www_label, camera_id)
                key_xxx = (xxx_label, camera_id)
                key_yyy = (yyy_label, camera_id)

                # Update accuracy_dict_action
                if key_action not in accuracy_dict_action:
                    accuracy_dict_action[key_action] = {'correct': 0, 'total': 0}
                if correct_action:
                    accuracy_dict_action[key_action]['correct'] += 1
                accuracy_dict_action[key_action]['total'] += 1

                # Similarly for other tasks
                if key_person not in accuracy_dict_person:
                    accuracy_dict_person[key_person] = {'correct': 0, 'total': 0}
                if correct_person:
                    accuracy_dict_person[key_person]['correct'] += 1
                accuracy_dict_person[key_person]['total'] += 1

                if key_uuu not in accuracy_dict_uuu:
                    accuracy_dict_uuu[key_uuu] = {'correct': 0, 'total': 0}
                if correct_uuu:
                    accuracy_dict_uuu[key_uuu]['correct'] += 1
                accuracy_dict_uuu[key_uuu]['total'] += 1

                if key_vvv not in accuracy_dict_vvv:
                    accuracy_dict_vvv[key_vvv] = {'correct': 0, 'total': 0}
                if correct_vvv:
                    accuracy_dict_vvv[key_vvv]['correct'] += 1
                accuracy_dict_vvv[key_vvv]['total'] += 1

                if key_www not in accuracy_dict_www:
                    accuracy_dict_www[key_www] = {'correct': 0, 'total': 0}
                if correct_www:
                    accuracy_dict_www[key_www]['correct'] += 1
                accuracy_dict_www[key_www]['total'] += 1

                if key_xxx not in accuracy_dict_xxx:
                    accuracy_dict_xxx[key_xxx] = {'correct': 0, 'total': 0}
                if correct_xxx:
                    accuracy_dict_xxx[key_xxx]['correct'] += 1
                accuracy_dict_xxx[key_xxx]['total'] += 1

                if key_yyy not in accuracy_dict_yyy:
                    accuracy_dict_yyy[key_yyy] = {'correct': 0, 'total': 0}
                if correct_yyy:
                    accuracy_dict_yyy[key_yyy]['correct'] += 1
                accuracy_dict_yyy[key_yyy]['total'] += 1

            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)
            acc1_meter_id.update(float(acc1_id) / b * 100, b)
            acc5_meter_id.update(float(acc5_id) / b * 100, b)

            acc1_meter_uuu.update(float(acc1_uuu) / b * 100, b)
            acc5_meter_uuu.update(float(acc5_uuu) / b * 100, b)
            acc1_meter_vvv.update(float(acc1_vvv) / b * 100, b)
            acc5_meter_vvv.update(float(acc5_vvv) / b * 100, b)
            acc1_meter_www.update(float(acc1_www) / b * 100, b)
            acc5_meter_www.update(float(acc5_www) / b * 100, b)
            acc1_meter_xxx.update(float(acc1_xxx) / b * 100, b)
            acc5_meter_xxx.update(float(acc5_xxx) / b * 100, b)
            acc1_meter_yyy.update(float(acc1_yyy) / b * 100, b)
            acc5_meter_yyy.update(float(acc5_yyy) / b * 100, b)


            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc1_meter.avg:.3f}\t'
                    f'Acc_id@1: {acc1_meter_id.avg:.3f}\t'
                    f'Acc_uuu@1: {acc1_meter_uuu.avg:.3f}\t'
                    f'Acc_vvv@1: {acc1_meter_vvv.avg:.3f}\t'
                    f'Acc_www@1: {acc1_meter_www.avg:.3f}\t'
                    f'Acc_xxx@1: {acc1_meter_xxx.avg:.3f}\t'
                    f'Acc_yyy@1: {acc1_meter_yyy.avg:.3f}\t'                                       
                )
    acc1_meter.sync()
    acc5_meter.sync()
    acc1_meter_id.sync()
    acc5_meter_id.sync()

    acc1_meter_uuu.sync()
    acc5_meter_uuu.sync()
    acc1_meter_vvv.sync()
    acc5_meter_vvv.sync()
    acc1_meter_www.sync()
    acc5_meter_www.sync()
    acc1_meter_xxx.sync()
    acc5_meter_xxx.sync()
    acc1_meter_yyy.sync()
    acc5_meter_yyy.sync()
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f} Acc_id@1 {acc1_meter_id.avg:.3f} Acc_id@5 {acc5_meter_id.avg:.3f} Acc_uuu@1: {acc1_meter_uuu.avg:.3f} Acc_vvv@1: {acc1_meter_vvv.avg:.3f} Acc_www@1: {acc1_meter_www.avg:.3f}  Acc_xxx@1: {acc1_meter_xxx.avg:.3f}  Acc_yyy@1: {acc1_meter_yyy.avg:.3f}'  )

    # Now process and save the accuracy tables for each task
    # Define a function to process an accuracy_dict and save to file
    def process_accuracy_dict(accuracy_dict, label_name):
        # Extract labels and camera_ids
        labels = set()
        camera_ids = set()
        for key in accuracy_dict.keys():
            label, camera_id = key
            labels.add(label)
            camera_ids.add(camera_id)
        labels = sorted(labels)
        camera_ids = sorted(camera_ids)

        # Create a table with rows as camera_ids, columns as labels
        import numpy as np
        table = np.zeros((len(camera_ids), len(labels)))

        # Build mappings from labels and camera_ids to indices
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        camera_to_idx = {camera_id: idx for idx, camera_id in enumerate(camera_ids)}

        # Fill the table
        for key, value in accuracy_dict.items():
            label, camera_id = key
            idx_label = label_to_idx[label]
            idx_camera = camera_to_idx[camera_id]
            accuracy = value['correct'] / value['total'] if value['total'] > 0 else 0
            table[idx_camera, idx_label] = accuracy

        # Save the table to a txt file
        file_path = os.path.join(config.OUTPUT, f'accuracy_table_{label_name}.txt')
        with open(file_path, 'w') as f:
            # Write header
            header = f'Camera_ID\\{label_name}\t' + '\t'.join(map(str, labels)) + '\n'
            f.write(header)
            # Write each row
            for i, camera_id in enumerate(camera_ids):
                row = [str(camera_id)]
                for j in range(len(labels)):
                    acc = table[i, j]
                    row.append(f'{acc:.4f}')
                f.write('\t'.join(row) + '\n')

    # Process and save each accuracy_dict
    process_accuracy_dict(accuracy_dict_action, 'action_label')
    process_accuracy_dict(accuracy_dict_person, 'person_id')
    process_accuracy_dict(accuracy_dict_uuu, 'uuu')
    process_accuracy_dict(accuracy_dict_vvv, 'vvv')
    process_accuracy_dict(accuracy_dict_www, 'www')
    process_accuracy_dict(accuracy_dict_xxx, 'xxx')
    process_accuracy_dict(accuracy_dict_yyy, 'yyy')

    return acc1_meter.avg, acc1_meter_id.avg, acc1_meter_uuu.avg, acc1_meter_vvv.avg, acc1_meter_www.avg, acc1_meter_xxx.avg, acc1_meter_yyy.avg



if __name__ == '__main__':
    # prepare config
    args, config = parse_option()
    # init_distributed
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

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    
    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")
    
    if (dist.get_rank() == 0) and (not os.path.exists(args.output)):
        os.makedirs(args.output)
    
    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config)


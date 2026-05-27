import os
import time
import yaml
import random
import logging
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import torchvision
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter
#from torch.cuda.amp import GradScaler, autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import torch.distributed
import matplotlib.pyplot as plt
import torch
import argparse
from thop import profile
import math
import pandas as pd
from models.submodules.layers import Conv1x1, Conv3x3, Linear, SpikingMatmul,SpikingMul
import models.spikingresformer
from utils.augment import DVSAugment
from utils.scheduler import BaseSchedulerPerEpoch, BaseSchedulerPerIter
from utils.utils import RecordDict, GlobalTimer, Timer
# from utils.utils import count_convNd, count_linear, count_matmul,count_spikingmul
from utils.utils import DatasetSplitter, DatasetWarpper, CriterionWarpper, OptimalTETLoss,DVStransform, SOPMonitor
from utils.utils import is_main_process, save_on_master, tb_record, accuracy, safe_makedirs
from spikingjelly.activation_based import functional, layer, base
from timm.data import FastCollateMixup, create_loader
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.models import create_model
from tqdm import tqdm
from lava.magma.core.run_configs import Loihi1SimCfg  # CPU仿真配置
from lava.magma.core.run_configs import Loihi2HwCfg    # 真实硬件配置（如有）
def parse_args():
    config_parser = argparse.ArgumentParser(description="Training Config", add_help=False)

    config_parser.add_argument(
        "-c",
        "--config",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser(description='Training')

    # training options
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--T', default=4, type=int, help='simulation steps')
    parser.add_argument('--model', default='spikingresformer_cifar', help='model type')
    parser.add_argument('--dataset', default='CIFAR100', help='dataset type')
    parser.add_argument('--augment', type=str, help='data augmentation')
    parser.add_argument('--mixup', type=bool, default=False, help='Mixup')
    parser.add_argument('--cutout', type=bool, default=False, help='Cutout')
    parser.add_argument('--label-smoothing', type=float, default=0, help='Label smoothing')
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers')
    parser.add_argument('--lr', default=1e-2, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--weight-decay', default=0, type=float, help='weight decay')

    parser.add_argument('--print-freq', default=0, type=int,
                        help='Number of times a debug message is printed in one epoch')
    parser.add_argument('--data-path', default='./datasets')
    parser.add_argument('--output-dir', default='./logs/temp/6_25')
    parser.add_argument('--resume', type=str, help='resume from checkpoint')
    parser.add_argument('--transfer', type=str, help='transfer from pretrained checkpoint')
    parser.add_argument('--input-size', type=int, nargs='+', default=[])
    parser.add_argument('--distributed-init-mode', type=str, default='env://')

    # argument of TET
    parser.add_argument('--TET', action='store_true', help='Use TET training')
    parser.add_argument('--TET-phi', type=float, default=1.0)
    parser.add_argument('--TET-lambda', type=float, default=0.0)

    parser.add_argument('--save-latest', action='store_true')
    parser.add_argument("--test-only", action="store_true", help="Only test the model")
    parser.add_argument('--amp', type=bool, default=True, help='Use AMP training')
    parser.add_argument('--sync-bn', action='store_true', help='Use SyncBN training')

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)

    return args


def setup_logger(output_dir):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]%(message)s',
                                  datefmt=r'%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    return logger


def init_distributed(logger: logging.Logger, distributed_init_mode):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        logger.info('Not using distributed mode')
        return False, 0, 1, 0
    assert local_rank < torch.cuda.device_count(), f"Invalid local_rank {local_rank} for available devices: {torch.cuda.device_count()}"

    torch.cuda.set_device(torch.device("cuda", local_rank))

    backend = 'nccl'
    logger.info('Distributed init rank {}'.format(rank))
    torch.distributed.init_process_group(backend=backend, init_method=distributed_init_mode,
                                         world_size=world_size, rank=rank)
    # only master process logs
    if rank != 0:
        logger.setLevel(logging.WARNING)
    return True, rank, world_size, local_rank


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def load_data(
    dataset_dir: str,
    batch_size: int,
    workers: int,
    num_classes: int,
    dataset_type: str,
    input_size: Tuple[int],
    distributed: bool,
    augment: str,
    mixup: bool,
    cutout: bool,
    label_smoothing: float,
    T: int,
):

    if dataset_type == 'CIFAR10':
        dataset_train = torchvision.datasets.CIFAR10(root=os.path.join(dataset_dir), train=True,
                                                     download=True)
        dataset_test = torchvision.datasets.CIFAR10(root=os.path.join(dataset_dir), train=False,
                                                    download=True)
        augment_args = dict(
            scale=[1.0, 1.0],
            ratio=[1.0, 1.0],
            hflip=0.5,
            vflip=0.0,
        )
        if augment:
            augment_args.update(dict(
                color_jitter=0.0,
                auto_augment=augment,
            ))
        if cutout:
            augment_args.update(dict(
                re_prob=0.25,
                re_mode='const',
                re_count=1,
                re_split=False,
            ))
        if mixup:
            augment_args.update(
                dict(collate_fn=FastCollateMixup(mixup_alpha=0.5, cutmix_alpha=0.0,
                                                 cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                                                 mode='batch', label_smoothing=label_smoothing,
                                                 num_classes=num_classes)))
        data_loader_train = create_loader(
            dataset_train,
            input_size=input_size,
            batch_size=batch_size,
            is_training=True,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            num_workers=workers,
            distributed=distributed,
            pin_memory=True,
            **augment_args,
        )
        data_loader_test = create_loader(
            dataset_test,
            input_size=input_size,
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            num_workers=workers,
            distributed=distributed,
            crop_pct=1.0,
            pin_memory=True,
        )
    elif dataset_type == 'CIFAR100':
        dataset_train = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=True,
                                                      download=True)
        dataset_test = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=False,
                                                     download=True)
        augment_args = dict(
            scale=[1.0, 1.0],
            ratio=[1.0, 1.0],
            hflip=0.5,
            vflip=0.0,
        )
        if augment:
            augment_args.update(dict(
                color_jitter=0.0,
                auto_augment=augment,
            ))
        if cutout:
            augment_args.update(dict(
                re_prob=0.25,
                re_mode='const',
                re_count=1,
                re_split=False,
            ))
        if mixup:
            augment_args.update(
                dict(collate_fn=FastCollateMixup(mixup_alpha=0.5, cutmix_alpha=0.0,
                                                 cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                                                 mode='batch', label_smoothing=label_smoothing,
                                                 num_classes=num_classes)))
        data_loader_train = create_loader(
            dataset_train,
            input_size=input_size,
            batch_size=batch_size,
            is_training=True,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=[n / 255. for n in [129.3, 124.1, 112.4]],
            std=[n / 255. for n in [68.2, 65.4, 70.4]],
            num_workers=workers,
            distributed=distributed,
            pin_memory=True,
            **augment_args,
        )
        data_loader_test = create_loader(
            dataset_test,
            input_size=input_size,
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=[n / 255. for n in [129.3, 124.1, 112.4]],
            std=[n / 255. for n in [68.2, 65.4, 70.4]],
            num_workers=workers,
            distributed=distributed,
            crop_pct=1.0,
            pin_memory=True,
        )
    elif dataset_type == 'CIFAR10DVS':
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        if augment:
            transform_train = DVStransform(transform=transforms.Compose([
                transforms.Resize(size=input_size[-2:], antialias=True),
                DVSAugment()]))
        else:
            transform_train = DVStransform(transform=transforms.Compose([
                transforms.Resize(size=input_size[-2:], antialias=True)]))
        transform_test = DVStransform(
            transform=transforms.Resize(size=input_size[-2:], antialias=True))

        dataset = CIFAR10DVS(dataset_dir, data_type='frame', frames_number=T, split_by='number')
        dataset_train, dataset_test = DatasetSplitter(dataset, 0.9,
                                                      True), DatasetSplitter(dataset, 0.1, False)
        dataset_train = DatasetWarpper(dataset_train, transform_train)
        dataset_test = DatasetWarpper(dataset_test, transform_test)
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(  # type:ignore
                dataset_train)
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_test)  # type:ignore
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset_train)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                        sampler=train_sampler, num_workers=workers,
                                                        pin_memory=True, drop_last=True)

        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                       sampler=test_sampler, num_workers=workers,
                                                       pin_memory=True, drop_last=False)
    elif dataset_type == 'DVS128Gesture':
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        if augment:
            transform_train = DVStransform(transform=transforms.Compose([
                transforms.Resize(size=input_size[-2:], antialias=True),
                DVSAugment()]))
        else:
            transform_train = DVStransform(transform=transforms.Compose([
                transforms.Resize(size=input_size[-2:], antialias=True)]))
        transform_test = DVStransform(
            transform=transforms.Resize(size=input_size[-2:], antialias=True))

        dataset_train = DVS128Gesture(dataset_dir, train=True, data_type='frame', frames_number=T,
                                      split_by='number')
        dataset_test = DVS128Gesture(dataset_dir, train=False, data_type='frame', frames_number=T,
                                     split_by='number')
        dataset_train = DatasetWarpper(dataset_train, transform_train)
        dataset_test = DatasetWarpper(dataset_test, transform_test)
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(  # type:ignore
                dataset_train)
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_test)  # type:ignore
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset_train)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                        sampler=train_sampler, num_workers=workers,
                                                        pin_memory=True, drop_last=True)

        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                       sampler=test_sampler, num_workers=workers,
                                                       pin_memory=True, drop_last=False)
    elif dataset_type == 'ImageNet' or dataset_type == 'ImageNet100':
        traindir = os.path.join(dataset_dir, 'train')
        valdir = os.path.join(dataset_dir, 'val')
        dataset_train = torchvision.datasets.ImageFolder(traindir)
        dataset_test = torchvision.datasets.ImageFolder(valdir)
        augment_args = dict(
            scale=[0.08, 1.0],
            ratio=[3. / 4., 4. / 3.],
            hflip=0.5,
            vflip=0.0,
        )
        if augment:
            augment_args.update(dict(
                color_jitter=0.4,
                auto_augment=augment,
            ))
        if cutout:
            augment_args.update(dict(
                re_prob=0.25,
                re_mode='const',
                re_count=1,
                re_split=False,
            ))
        if mixup:
            augment_args.update(
                dict(collate_fn=FastCollateMixup(mixup_alpha=0.2, cutmix_alpha=1.0,
                                                 cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                                                 mode='batch', label_smoothing=label_smoothing,
                                                 num_classes=num_classes)))
        data_loader_train = create_loader(
            dataset_train,
            input_size=input_size,
            batch_size=batch_size,
            is_training=True,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_workers=workers,
            distributed=distributed,
            pin_memory=True,
            **augment_args,
        )
        data_loader_test = create_loader(
            dataset_test,
            input_size=input_size,
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            interpolation='bicubic',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_workers=workers,
            distributed=distributed,
            crop_pct=0.95,
            pin_memory=True,
        )
    else:
        raise ValueError(dataset_type)

    return dataset_train, dataset_test, data_loader_train, data_loader_test


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader_train: torch.utils.data.DataLoader,
    logger: logging.Logger,
    args: argparse.Namespace,
    print_freq: int,
    factor: int,
    scheduler_per_iter: Optional[BaseSchedulerPerIter] = None,
    scaler: Optional[GradScaler] = None,
    one_hot: Optional[int] = None,
    epoch: int = 0,):
    model.train()
    metric_dict = RecordDict({'loss': None, 'acc@1': None, 'acc@5': None})
    timer_container = [0.0]

    model.zero_grad()
    with tqdm(total=len(data_loader_train)) as t:
        for idx, (image, target) in enumerate(data_loader_train):
            with GlobalTimer('iter', timer_container):
                image, target = image.cuda(), target.cuda()
                if one_hot:
                    target = F.one_hot(target, one_hot).float()
                if scaler is not None:
                    with autocast(device_type='cuda'):
                        output = model(image)
                        loss = criterion(output, target)
                else:
                    output = model(image)
                    loss = criterion(output, target)
                metric_dict['loss'].update(loss.item())

                if scaler is not None:
                    scaler.scale(loss).backward()  # type:ignore
                    scaler.step(optimizer)
                    scaler.update()
                    model.zero_grad()
                else:
                    loss.backward()
                    if idx % 100 == 0 and epoch == args.epochs:
                        # 假设 grad = model.prologue[0].weight.grad, shape [64, 3, 3, 3]
                        grad = model.prologue[0].weight.grad 

                        # 1️展开所有输入通道，不做平均 -> [64*3, 3, 3]
                        out_channels, in_channels, kH, kW = grad.shape
                        grad_kernels = grad.detach().cpu().abs().reshape(out_channels * in_channels, kH, kW)

                        # 2️计算网格大小
                        n = grad_kernels.shape[0]  # 64*3 = 192
                        grid_size = int(math.ceil(math.sqrt(n)))  # 每边铺多少个 kernel

                        # 3️创建画布
                        canvas_height = grid_size * kH
                        canvas_width = grid_size * kW
                        canvas = torch.zeros(canvas_height, canvas_width)

                        # 4️填充每个 kernel 到画布
                        for i in range(n):
                            row = i // grid_size
                            col = i % grid_size
                            canvas[row*kH:(row+1)*kH, col*kW:(col+1)*kW] = grad_kernels[i]

                        # 5️确保保存目录存在
                        save_dir = args.output_dir
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"last_conv1_grad_all_channels_{idx}.png")

                        # 6️ 绘图并标注尺寸
                        plt.figure(figsize=(12, 12))
                        plt.imshow(canvas.numpy(), cmap='hot', vmin=0.0, vmax=0.1)
                        plt.colorbar()
                        plt.title("Conv1 Gradient (Absolute Values, All Channels)")

                        # 标注 x/y 轴信息
                        plt.xlabel(f"Width: {canvas_width} pixels ({grid_size} kernels per row)")
                        plt.ylabel(f"Height: {canvas_height} pixels ({grid_size} kernels per column)")

                        plt.savefig(save_path, bbox_inches='tight', dpi=300)
                        plt.close()

                        print(f"Gradient image with all channels displayed saved to {save_path}")
                    optimizer.step()
                    model.zero_grad()

                if scheduler_per_iter is not None:
                    scheduler_per_iter.step()

                functional.reset_net(model)

                if target.dim() > 1:
                    target = target.argmax(-1)
                acc1, acc5 = accuracy(output.mean(0), target, topk=(1, 5))
                acc1_s = acc1.item()
                acc5_s = acc5.item()

                batch_size = image.shape[0]
                metric_dict['acc@1'].update(acc1_s, batch_size)
                metric_dict['acc@5'].update(acc5_s, batch_size)

            if print_freq != 0 and ((idx + 1) % int(len(data_loader_train) / (print_freq))) == 0:
                #torch.distributed.barrier()
                metric_dict.sync()
                logger.debug(' [{}/{}] it/s: {:.5f}, loss: {:.5f}, acc@1: {:.5f}, acc@5: {:.5f}'.format(
                    idx + 1, len(data_loader_train),
                    (idx + 1) * batch_size * factor / timer_container[0], metric_dict['loss'].ave,
                    metric_dict['acc@1'].ave, metric_dict['acc@5'].ave))
            t.set_postfix(loss='{:05.3f}'.format(loss.item()))
            t.update()

    #torch.distributed.barrier()
    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['acc@1'].ave, metric_dict['acc@5'].ave


def evaluate(model, criterion, data_loader, print_freq, logger, one_hot=None):
    model.eval()
    metric_dict = RecordDict({'loss': None, 'acc@1': None, 'acc@5': None})
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader):
            image, target = image.cuda(), target.cuda()
            if one_hot:
                target = F.one_hot(target, one_hot).float()
            output = model(image)
            loss = criterion(output, target)
            metric_dict['loss'].update(loss.item())
            functional.reset_net(model)

            if target.dim() > 1:
                target = target.argmax(-1)
            acc1, acc5 = accuracy(output.mean(0), target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1.item(), batch_size)
            metric_dict['acc@5'].update(acc5.item(), batch_size)

            if print_freq != 0 and ((idx + 1) % int(len(data_loader) / print_freq)) == 0:
                #torch.distributed.barrier()
                metric_dict.sync()
                logger.debug(' [{}/{}] loss: {:.5f}, acc@1: {:.5f}, acc@5: {:.5f}'.format(
                    idx + 1, len(data_loader), metric_dict['loss'].ave, metric_dict['acc@1'].ave,
                    metric_dict['acc@5'].ave))

    #torch.distributed.barrier()
    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['acc@1'].ave, metric_dict['acc@5'].ave

from collections import OrderedDict, defaultdict
import re

def get_module_key(name):
    if name == 'prologue.0':
        return 'prologue.0'

    if name == 'classifier':
        return 'classifier'

    # layers.i.j.xxx
    m = re.match(r'layers\.(\d+)\.(\d+)\.(.+)', name)
    if m:
        stage = int(m.group(1))
        block = int(m.group(2))
        tail  = m.group(3)

        # Downsample
        if block == 0 and tail == 'conv':
            return f'stage{stage}_downsample'
        if stage == 0 and block == 0:
            return f'stage{stage}_attn{block}'
        if stage == 0 and block == 1:
            return f'stage{stage}_mlp{block}'
        if block%2 == 1 :
            return f'stage{stage}_attn{block}'
        if block%2 == 0 :
            return f'stage{stage}_mlp{block}'

        # if block%2 == 0:
        #     return f'stage{stage}_attn{block}'
        # if block%2 == 1:
        #     return f'stage{stage}_mlp{block}'

    return 'others'

def test(
    model: nn.Module,
    data_loader_test: torch.utils.data.DataLoader,
    input_size: Tuple[int],
    args: argparse.Namespace,
    logger: logging.Logger,
):

    logger.info('[Test]')
    mon = SOPMonitor(model)
    model.eval()
    model = model.cuda()
    mon.enable()
    logger.debug('Test start')
    metric_dict = RecordDict({'acc@1': None, 'acc@5': None}, test=True)
    with torch.no_grad():
        t = time.time()
        for idx, (image, target) in enumerate(data_loader_test):
            image, target = image.cuda(), target.cuda()
            output = model(image).mean(0)
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1.item(), batch_size)
            metric_dict['acc@5'].update(acc5.item(), batch_size)
            if args.print_freq != 0 and ((idx + 1) %
                                         int(len(data_loader_test) / args.print_freq)) == 0:
                logger.debug('Test: [{}/{}]'.format(idx + 1, len(data_loader_test)))
        logger.info('Throughput: {:.5f} it/s'.format(
            len(data_loader_test) * args.batch_size / (time.time() - t)))

    metric_dict.sync()
    logger.info('Acc@1: {:.5f}, Acc@5: {:.5f}'.format(metric_dict['acc@1'].ave,
                                                      metric_dict['acc@5'].ave))

    step_mode = 's'
    for m in model.modules():
        if isinstance(m, base.StepModule):
            if m.step_mode == 'm':
                step_mode = 'm'
            else:
                step_mode = 's'
            break
    # parameter count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = params / 1e6
    
    logger.info('params: {:.2f} M.'.format(params))

    sops = 0
    ops = 0
    module_energy = OrderedDict()
    for name in mon.monitored_layers:
        sublist = mon[name]
        if not sublist:
            continue
        print(name)
        val = torch.cat(sublist).float().mean().item()
        module_key = get_module_key(name)

        if module_key not in module_energy:
            module_energy[module_key] = 0.0

        # 浮点 or 脉冲
        # if name in ['prologue.0','classifier']:
        #     ops += val
        # else:
        sops += val
        module_energy[module_key] += val
    sops = sops / (1000**3)
    ops = ops / (1000**3)
    for name in module_energy:
        # if name in ['prologue.0','classifier']:
        #     module_energy[name] = (module_energy[name]*4.6/ (1000**3))/ args.batch_size
        # else:
        module_energy[name] = (module_energy[name]*0.9 / (1000**3))/ args.batch_size
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    excel_path = os.path.join(save_dir, "module_energy.xlsx")

    df = pd.DataFrame(
        list(module_energy.items()),
        columns=["Module", "Energy_per_image(J)"]
    )
    df.to_excel(excel_path, index=False)

    print(f"Module energy saved to {excel_path}")
    # input is [N, C, H, W] or [T*N, C, H, W]
    sops = sops / args.batch_size
    ops = ops / args.batch_size
    logger.info('Avg SOPs: {:.5f} G, Avg OPs: {:.5f} G,Energy: {:.5f} mJ.'.format(sops,ops, (0.9 * sops)+(4.6*ops)))
    functional.reset_net(model)



def main():

    ##################################################
    #                       setup
    ##################################################

    args = parse_args()
    output_dir = args.output_dir

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    #作用：强制 cuDNN 选择确定性算法（deterministic algorithms）目的：为了保证每次运行结果一致，确保可复现性 代价：可能会降低速度，因为某些最快的算法是非确定性的（比如 conv）
    torch.backends.cudnn.benchmark = False  # type: ignore
    #作用：禁止 cuDNN 自动寻找最快的卷积实现方式 原因：benchmark=True 会根据输入数据动态选择最快算法，但可能带来不同运行结果（非确定性） 设置为 False：可以避免这一“随机优化”带来的不可重复性
    safe_makedirs(args.output_dir)
    logger = setup_logger(args.output_dir)
    print('初始化...')
    distributed, rank, world_size, local_rank = init_distributed(logger, args.distributed_init_mode)

    logger.info(str(args))

    # load data

    dataset_type = args.dataset
    one_hot = None
    if dataset_type == 'CIFAR10':
        num_classes = 10
        input_size = (3, 32, 32)
    elif dataset_type == 'CIFAR10DVS':
        one_hot = 10
        num_classes = 10
        input_size = (3, 64, 64)
    elif dataset_type == 'DVS128Gesture':
        one_hot = 11
        num_classes = 11
        input_size = (3, 64, 64)
    elif dataset_type == 'CIFAR100':
        num_classes = 100
        input_size = (3, 32, 32)
    elif dataset_type == 'ImageNet':
        num_classes = 1000
        input_size = (3, 224, 224)
    elif dataset_type == 'ImageNet100':
        num_classes = 100
        input_size = (3, 224, 224)
    else:
        raise ValueError(dataset_type)
    if len(args.input_size) != 0:
        input_size = args.input_size
    print('加载数据...')
    dataset_train, dataset_test, data_loader_train, data_loader_test = load_data(
        args.data_path, args.batch_size, args.workers, num_classes, dataset_type, input_size,
        distributed, args.augment, args.mixup, args.cutout, args.label_smoothing, args.T)
    logger.info('dataset_train: {}, dataset_test: {}'.format(len(dataset_train), len(dataset_test)))

    # model

    model = create_model(
        args.model,
        T=args.T,
        num_classes=num_classes,
        img_size=input_size[-1],
    ).cuda()

    # transfer
    if args.transfer:
        checkpoint = torch.load(args.transfer, map_location='cpu')
        model.transfer(checkpoint['model'])

    # optimzer

    optimizer = create_optimizer_v2(
        model,
        opt=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # loss_fn

    if args.mixup:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    # criterion = CriterionWarpper(criterion, args.TET, args.TET_phi, args.TET_lambda)

    criterion = OptimalTETLoss(
    criterion,
    args.TET,
    lambda_final=args.TET_lambda,
    phi_temporal=args.TET_phi
    )
    criterion_eval = nn.CrossEntropyLoss()
    criterion_eval = OptimalTETLoss(criterion_eval)
    # amp speed up

    if args.amp:
        scaler = GradScaler()
        #在使用 float16 或 bfloat16 训练时，数值精度降低，可能会造成梯度非常小、为 0，导致训练不收敛。GradScaler 通过把 loss 放大，避免梯度过小的问题，并在反向传播之后再自动缩小。
    else:
        scaler = None

    # lr scheduler

    lr_scheduler, _ = create_scheduler_v2(
        optimizer,
        sched='cosine',
        num_epochs=args.epochs,
        cooldown_epochs=10,
        min_lr=1e-5,
        warmup_lr=1e-5,
        warmup_epochs=3,
    )

    # Sync BN
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DDP

    model_without_ddp = model
    if distributed and not args.test_only:
        print(f"[Rank {local_rank}] before DDP", flush=True)

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False
        )

        print(f"[Rank {local_rank}] after DDP", flush=True)

        model_without_ddp = model.module

    # custom scheduler

    scheduler_per_iter = None
    scheduler_per_epoch = None

    # resume

    if args.resume:
        #断点续训
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        max_acc1 = checkpoint['max_acc1']
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logger.info('Resume from epoch {}'.format(start_epoch))
        start_epoch += 1
        # custom scheduler
    else:
        start_epoch = 0
        max_acc1 = 0

    # logger.debug(str(model))

    ##################################################
    #                   test only
    ##################################################

    if args.test_only:
        if distributed:
            logger.error('Using distribute mode in test, abort')
            return
        test(model_without_ddp, data_loader_test, input_size, args, logger)
        return

    ##################################################
    #                   Train
    ##################################################

    tb_writer = None
    if is_main_process():
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'),
                                  purge_step=start_epoch)

    logger.info("[Train]")
    for epoch in range(start_epoch, args.epochs):
        if distributed and hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)
        logger.info('Epoch [{}] Start, lr {:.6f}'.format(epoch, optimizer.param_groups[0]["lr"]))
        with Timer(' Train', logger):
            train_loss, train_acc1, train_acc5 = train_one_epoch(model, criterion, optimizer,
                                                                 data_loader_train, logger, args,
                                                                 args.print_freq, world_size,
                                                                 scheduler_per_iter, scaler,
                                                                 one_hot,epoch+1)
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)
            if scheduler_per_epoch is not None:
                scheduler_per_epoch.step()

        with Timer(' Test', logger):
            test_loss, test_acc1, test_acc5 = evaluate(model, criterion_eval, data_loader_test,
                                                       args.print_freq, logger, one_hot)
        if is_main_process() and tb_writer is not None:
            tb_record(tb_writer, train_loss, train_acc1, train_acc5, test_loss, test_acc1,
                      test_acc5, epoch)

        logger.info(' Test loss: {:.5f}, Acc@1: {:.5f}, Acc@5: {:.5f}'.format(
            test_loss, test_acc1, test_acc5))

        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_acc1': max_acc1, }
        if lr_scheduler is not None:
            checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
        # custom scheduler

        if args.save_latest:
            save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_latest.pth'))

        if max_acc1 < test_acc1:
            max_acc1 = test_acc1
            save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_max_acc1.pth'))

    logger.info('Training completed.')

    ##################################################
    #                   test
    ##################################################

    ##### reset utils #####

    # reset model

    del model, model_without_ddp

    model = create_model(
        args.model,
        T=args.T,
        num_classes=num_classes,
        img_size=input_size[-1],
    )

    try:
        checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoint_max_acc1.pth'),
                                map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    except:
        logger.warning('Cannot load max acc1 model, skip test.')
        logger.warning('Exit.')
        return

    # reload data

    del dataset_train, dataset_test, data_loader_train, data_loader_test
    _, _, _, data_loader_test = load_data(args.data_path, args.batch_size, args.workers,
                                          num_classes, dataset_type, input_size, False,
                                          args.augment, args.mixup, args.cutout,
                                          args.label_smoothing, args.T)

    ##### test #####

    if is_main_process():
        test(model, data_loader_test, input_size, args, logger)
    logger.info('All Done.')

if __name__ == "__main__":
    main()


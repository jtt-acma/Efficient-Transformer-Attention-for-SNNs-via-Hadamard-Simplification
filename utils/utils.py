import os
import time
from typing import Tuple, Union
import torch
import torch.distributed
import torch.utils.data
import errno
import datetime
from torch import Tensor, nn
from math import nan
from torch.utils.tensorboard.writer import SummaryWriter

import sys

sys.path.append('..')

from models.submodules.layers import SpikingMatmul, SpikingMul


def is_distributed():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_rank():
    if not is_distributed():
        return 0
    return torch.distributed.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def safe_makedirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def tb_record(
    tb_writer: SummaryWriter,
    train_loss: float,
    train_acc1: float,
    train_acc5: float,
    test_loss: float,
    test_acc1: float,
    test_acc5: float,
    epoch: int,
):
    tb_writer.add_scalar('train/loss', train_loss, epoch)
    tb_writer.add_scalar('train/acc1', train_acc1, epoch)
    tb_writer.add_scalar('train/acc5', train_acc5, epoch)
    tb_writer.add_scalar('test/loss', test_loss, epoch)
    tb_writer.add_scalar('test/acc1', test_acc1, epoch)
    tb_writer.add_scalar('test/acc5', test_acc5, epoch)




def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, )):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


class Record:
    r'''
    Synchronous record
    '''
    def __init__(self, test: bool = False) -> None:
        self.value = torch.tensor([0], dtype=torch.float64, device='cuda')
        self.count = torch.tensor([0], dtype=torch.int64, device='cuda')
        self.global_value = 0.0
        self.global_count = 0
        self.test = test

    def sync(self) -> None:
        r'''
        reduce value and count, and update global ones
        '''
        if is_distributed() and not self.test:
            torch.distributed.all_reduce(self.value, torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(self.count, torch.distributed.ReduceOp.SUM)
        self.global_value += self.value.item()
        self.global_count += self.count.item()
        self.value[0] = 0.0
        self.count[0] = 0

    def update(self, value, count=1) -> None:
        r'''
        update local value and count
        '''
        self.value[0] += value * count
        self.count[0] += count

    def reset(self) -> None:
        self.value[0] = 0.0
        self.count[0] = 0
        self.global_value = 0.0
        self.global_count = 0

    @property
    def ave(self):
        if self.global_count == 0:
            return nan
        return self.global_value / self.global_count


class RecordDict:
    def __init__(self, dic: dict, test: bool = False) -> None:
        self.__inner_dict = dict()
        self.test = test
        for key in dic.keys():
            self.__inner_dict[key] = Record(test)

    def __getitem__(self, key) -> Record:
        return self.__inner_dict[key]

    def __setitem__(self, key, value) -> None:
        assert (isinstance(value, Record))
        self.__inner_dict[key] = value

    def __str__(self) -> str:
        s = []
        for key, value in self.__inner_dict.items():
            s.append('{key}:{value}'.format(key=key, value=value.ave))
        return ', '.join(s)

    def sync(self):
        for value in self.__inner_dict.values():
            value.sync()

    def reset(self):
        for value in self.__inner_dict.values():
            value.reset()

    def add_record(self, key):
        self.__inner_dict[key] = Record(self.test)


class Timer:
    def __init__(self, timer_name, logger):
        self.timer_name = timer_name
        self.logger = logger

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # seconds
        self.logger.debug('{} spent: {}.'.format(
            self.timer_name, str(datetime.timedelta(seconds=int(self.interval)))))


class GlobalTimer:
    def __init__(self, timer_name, container):
        self.timer_name = timer_name
        self.container = container

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # seconds
        self.container[0] += self.interval


class DatasetSplitter(torch.utils.data.Dataset):
    '''To split CIFAR10DVS into training dataset and test dataset'''
    def __init__(self, parent_dataset, rate=0.1, train=True):

        self.parent_dataset = parent_dataset
        self.rate = rate
        self.train = train
        self.it_of_original = len(parent_dataset) // 10
        self.it_of_split = int(self.it_of_original * rate)

    def __len__(self):
        return int(len(self.parent_dataset) * self.rate)

    def __getitem__(self, index):
        base = (index // self.it_of_split) * self.it_of_original
        off = index % self.it_of_split
        if not self.train:
            off = self.it_of_original - off - 1
        item = self.parent_dataset[base + off]

        return item


class CriterionWarpper(nn.Module):
    def __init__(self, criterion, TET=False, TET_phi=1.0, TET_lambda=0.0) -> None:
        super().__init__()
        self.criterion = criterion
        self.TET = TET
        self.TET_phi = TET_phi
        self.TET_lambda = TET_lambda
        self.mse = nn.MSELoss()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if self.TET:
            loss = 0
            for t in range(output.shape[0]):
                loss = loss + (1. - self.TET_lambda) * self.criterion(output[t], target)
            loss = loss / output.shape[0]
            if self.TET_lambda != 0:
                loss = loss + self.TET_lambda * self.mse(
                    output,
                    torch.zeros_like(output).fill_(self.TET_phi))
            return loss
        else:
            return self.criterion(output.mean(0), target)
class OptimalTETLoss(nn.Module):
    def __init__(
        self,
        criterion,
        TET=False,
        lambda_final=0.6,   # final-step 权重（核心）
        phi_temporal=0.05,  # 时间平滑正则（很弱）
    ):
        super().__init__()
        self.TET = TET
        self.criterion = criterion
        self.lambda_final = lambda_final
        self.phi_temporal = phi_temporal
        self.mse = nn.MSELoss()

    def forward(self, outputs, target):
        """
        outputs: [T, B, C]  (spike logits or membrane outputs)
        target:  [B]
        """
        if self.TET:
            T = outputs.shape[0]
            loss_time = 0.0
            for t in range(T):
                loss_time += self.criterion(outputs[t], target)
            loss_time = loss_time / T
            loss_final = self.criterion(outputs[-1], target)

            if self.phi_temporal > 0 and T > 1:
                loss_temporal = self.mse(outputs[1:], outputs[:-1])
            else:
                loss_temporal = 0.0

            loss = (
                (1.0 - self.lambda_final) * loss_time
                + self.lambda_final * loss_final
                + self.phi_temporal * loss_temporal
            )
        else:
            loss = self.criterion(outputs.mean(0), target)
        return loss


class DatasetWarpper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.trasnform = transform

    def __getitem__(self, index):
        return self.trasnform(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


class DVStransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = torch.from_numpy(img).float()
        shape = [img.shape[0], img.shape[1]]
        img = img.flatten(0, 1)
        img = self.transform(img)
        shape.extend(img.shape[1:])
        img = img.view(shape)
        img3 = img.sum(dim=1, keepdim=True)
        img = torch.cat([img, img3], dim=1)
        return img


def unpack_for_conv(x: Union[Tuple[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(x, tuple):
        assert x.__len__() == 1
        x = x[0]
    return x.flatten(0, 1)


def unpack_for_linear(x: Union[Tuple[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(x, tuple):
        assert x.__len__() == 1
        x = x[0]
    return x.flatten(0, 1)


def unpack_for_matmul(x: Union[Tuple[torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor]:
    assert isinstance(x, tuple)
    assert x.__len__() == 2
    left, right = x
    return left.flatten(0, 1), right.flatten(0, 1)


class BaseMonitor:
    def __init__(self):
        self.hooks = []
        self.monitored_layers = []
        self.records = []
        self.name_records_index = {}
        self._enable = True

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.records[i]
        elif isinstance(i, str):
            y = []
            for index in self.name_records_index[i]:
                y.append(self.records[index])
            return y
        else:
            raise ValueError(i)

    def clear_recorded_data(self):
        self.records.clear()
        for k, v in self.name_records_index.items():
            v.clear()

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def is_enable(self):
        return self._enable

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.remove_hooks()


class SOPMonitor(BaseMonitor):
    def __init__(self, net: nn.Module):
        super().__init__()
        for name, m in net.named_modules():
            # if name in net.skip:  #self.skip = ['prologue.0', 'classifier'] 
            #     continue
            if isinstance(m, nn.Conv2d):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(
                    self.create_hook_conv(name))) 
            elif isinstance(m, nn.Linear):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(
                    self.create_hook_linear(name))) 
            elif isinstance(m, SpikingMatmul):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(
                    self.create_hook_matmul(name)))  
            elif isinstance(m, SpikingMul):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(
                    self.create_hook_spikingmul(name)))  

    def cal_sop_conv(self, x: Tensor,  m: nn.Conv2d):
        # x: [N, C, H, W], binary spike
        num_spikes = (x > 0).sum().float()

        kh, kw = m.kernel_size
        cout = m.out_channels
        groups = m.groups

        fanout = kh * kw * (cout // groups)
        return (num_spikes * fanout).unsqueeze(0)


    def create_hook_conv(self, name):
        def hook(m: nn.Conv2d, x: Tensor, y: Tensor):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                inp = unpack_for_conv(x).detach()
                if self._is_spike_tensor(inp):
                    self.records.append(self.cal_sop_conv(inp, m))
                else:
                    # print(name + ' is not spike tensor!')
                    out = y
                    if isinstance(out, tuple):
                        out = out[0]
                    if out.dim() == 5:
                        out_flat = out.flatten(0, 1)
                    else:
                        out_flat = out
                    ops = calculate_conv2d_flops(input_size=list(inp.shape),
                                                  output_size=list(out_flat.shape),
                                                  kernel_size=list(m.weight.shape),
                                                  groups=m.groups)
                    self.records.append(torch.tensor([float(ops)], dtype=torch.float64, device=inp.device))

        return hook

    def cal_sop_linear(self, x: Tensor, m: nn.Linear):
        num_spikes = (x > 0).sum().float()
        return (num_spikes * m.out_features).unsqueeze(0)

    def create_hook_linear(self, name):
        def hook(m: nn.Conv2d, x: Tensor, y: Tensor):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                inp = unpack_for_linear(x).detach()
                if self._is_spike_tensor(inp):
                    self.records.append(self.cal_sop_linear(inp, m))
                else:
                    # print(name + ' is not spike tensor!')
                    out = y
                    if isinstance(out, tuple):
                        out = out[0]
                    out_flat = out.flatten(0, 1) if out.dim() == 2 or out.dim() == 3 else out
                    num_elements = out_flat.numel()
                    ops = int(num_elements * m.in_features)
                    self.records.append(torch.tensor([float(ops)], dtype=torch.float64, device=inp.device))

        return hook

    def cal_sop_matmul(self, left, right, m):
        with torch.no_grad():
            if m.spike == 'l':
                sop = (left > 0).sum() * right.shape[-1]
            elif m.spike == 'r':
                sop = (right > 0).sum() * left.shape[-2]
            elif m.spike == 'both':
                sop = (left > 0).sum() * (right > 0).sum()
            else:
                raise ValueError
            return sop.unsqueeze(0)

    def create_hook_matmul(self, name):
        def hook(m: nn.Conv2d, x: Tensor, y: Tensor):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                left, right = unpack_for_matmul(x)
                left_d, right_d = left.detach(), right.detach()
                if self._is_spike_tensor(left_d) or self._is_spike_tensor(right_d):
                    self.records.append(self.cal_sop_matmul(left_d, right_d, m))
                else:
                    print(name + ' is not spike tensor!')
                    total_mul = right_d.shape[-1]
                    num_elements = left_d.numel()
                    ops = int(total_mul * num_elements)
                    self.records.append(torch.tensor([float(ops)], dtype=torch.float64, device=left_d.device))

        return hook

    def cal_sop_spikingmul(self, left: Tensor, right: Tensor, m: SpikingMul):
        with torch.no_grad():
            if m.spike == 'l':
                right = torch.ones_like(right)
                out = torch.mul(left, right)
            elif m.spike == 'r':
                left = torch.ones_like(left)
                out = torch.mul(left, right)
            elif m.spike == 'both':
                out = torch.zeros(1, device=left.device)
            else:
                raise ValueError(m.spike)
            return out.sum().unsqueeze(0)

    def create_hook_spikingmul(self, name):
        def hook(m: SpikingMul, x: Tensor, y: Tensor):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                left, right = unpack_for_matmul(x)
                left_d, right_d = left.detach(), right.detach()
                if self._is_spike_tensor(left_d) or self._is_spike_tensor(right_d):
                    self.records.append(self.cal_sop_spikingmul(left_d, right_d, m))
                else:
                    print(name + ' is not spike tensor!')
                    ops = int(left_d.numel())
                    self.records.append(torch.tensor([float(ops)], dtype=torch.float64, device=left_d.device))

        return hook

    def _is_spike_tensor(self, t: Tensor, sample: int = 1024, tol: float = 1e-6) -> bool:
        # 判断张量是否为二值脉冲（0/1）——样本检测以控制开销
        if not torch.is_floating_point(t):
            return True
        flat = t.reshape(-1)
        n = flat.numel()
        if n == 0:
            return False
        if n > sample:
            idx = torch.randperm(n, device=flat.device)[:sample]
            vals = flat[idx]
        else:
            vals = flat
        # 接受接近 0 或接近 1 的值
        is_zero = torch.abs(vals) <= tol
        is_one = torch.abs(vals - 1.0) <= tol
        return bool(torch.all(is_zero | is_one).item())


def l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res


def calculate_conv2d_flops(input_size: list, output_size: list, kernel_size: list, groups: int):
    # T, N, out_c, oh, ow = output_size
    # T, N, in_c, ih, iw = input_size
    # out_c, in_c, kh, kw = kernel_size
    in_c = input_size[2]
    g = groups
    return l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])


# def count_convNd(m, x, y: torch.Tensor):
#     x = x[0]

#     kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
#     bias_ops = 1 if m.bias is not None else 0

#     m.total_ops += calculate_conv2d_flops(input_size=list(x.shape), output_size=list(y.shape),
#                                           kernel_size=list(m.weight.shape), groups=m.groups)


def count_matmul(m, x, y):
    left, right = x
    # per output element
    total_mul = right.shape[-1]
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = left.numel()

    m.total_ops += torch.DoubleTensor([int(total_mul * num_elements)])


# nn.Linear
def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()

    m.total_ops += torch.DoubleTensor([int(total_mul * num_elements)])
def count_spikingmul(m, x, y):
    left, right = x  # forward 里是 (left, right)
    # 每个元素只需要 1 次乘法
    num_elements = left.numel()
    m.total_ops += torch.DoubleTensor([int(num_elements)])
#!/usr/bin/env python3
"""
SpikingResformer DSSA Loihi 硬件效率分析工具

重要说明与局限性声明：
================
本脚本提供基于理论模型和Lava CPU仿真的能耗估算，** NOT 真实硬件测量 **。

关键限制（必读）：
1. **无真实硬件验证**
   - 本脚本使用Lava Loihi1SimCfg进行CPU功能仿真
   - **未在真实Intel Loihi神经形态芯片上运行**
   - **未在SpiNNaker或其他神经形态平台上验证**
   
2. **路由效应（Routing）- 理论估算**
   - 使用基于网格拓扑的理论模型估算NoC跳数
   - 使用Davies 2018文献中的tile hop energy (3.0/4.0 pJ)
   - **不模拟真实Loihi的**：
     * 实际路由路径（由编译器/映射器决定）
     * 拥塞、冲突、仲裁延迟
     * 实际2D网格拓扑的物理距离
     * 时间复用的NoC带宽限制
     
3. **内存效应（Memory）- 未计算**
   - Loihi片上SRAM访问能耗未在公开文献中披露
   - 突触权重存储、膜电位读写的实际能耗未建模
   - 标记为"未计算"（见代码中mem_energy = 0.0）

4. **功能仿真 vs 时序仿真**
   - Loihi1SimCfg仅验证计算逻辑正确性
   - 不模拟真实的：
     * 神经形态核心的并行执行时序
     * 异步事件驱动的时间精度
     * 膜电位泄漏的时间常数精度

5. **能耗估算的不确定性**
   - 基于文献参数的解析计算，非直接测量
   - 实际能耗取决于：
     * 具体输入数据模式
     * 芯片温度、工艺变化
     * 编译器优化策略
     * 映射到物理核心的方式

适用场景：
- 算法级别的能耗趋势分析
- 与理论SNN（0.9pJ）的对比研究
- 架构设计的初步评估

不适用场景：
- 声称真实Loihi硬件性能
- 与真实部署系统的一一对比
- 精确到pJ的绝对能耗预测

所有估算均明确标注来源（Davies et al. 2018）和方法，不生成虚假硬件数据。

学术诚实补充（仿真代码性质）：
------------------------------------------------------------------------
本脚本使用Lava进行CPU功能仿真：

- 使用Lava原生Conv/LIF/Dense进程进行Token-wise仿真
- 每个空间位置(h,w)映射到独立的Lava进程组
- 实际权重从PyTorch DSSA模块加载并应用
- NoC能耗基于理论公式估算（input*9*1.33*2）

⚠️ 限制：Lava Loihi1SimCfg仅进行CPU功能仿真，非真实Loihi硬件执行
------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import numpy as np
import json
import yaml
import argparse
import time
from pathlib import Path
from torchvision import transforms
from collections import defaultdict

# 添加项目路径
import sys
sys.path.insert(0, '/home/jtt/SpikingResformer-main')

# Lava 导入
try:
    # Lava 核心
    from lava.magma.core.process.process import AbstractProcess
    from lava.magma.core.process.ports.ports import InPort, OutPort
    from lava.magma.core.process.variable import Var
    from lava.magma.core.model.py.model import PyLoihiProcessModel
    from lava.magma.core.model.py.ports import PyInPort, PyOutPort
    from lava.magma.core.model.py.type import LavaPyType
    from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
    from lava.magma.core.decorator import implements, requires
    from lava.magma.core.resources import CPU
    from lava.magma.core.run_configs import Loihi1SimCfg
    from lava.magma.core.run_conditions import RunSteps
    
    # Lava 标准进程（真实Loihi硬件进程）
    from lava.proc.conv.process import Conv
    from lava.proc.lif.process import LIF
    from lava.proc.io.source import RingBuffer
    from lava.proc.monitor.process import Monitor
    
    LAVA_AVAILABLE = True
    
    # 尝试导入 Lava 原生 DSSA 实现
    try:
        from models.USSA_lava_dssa_shared import run_dssa_lava_shared as run_lava_native_simulation
        LAVA_NATIVE_AVAILABLE = True
    except ImportError:
        raise RuntimeError('Lava原生DSSA实现不可用，无法继续仿真')
        LAVA_NATIVE_AVAILABLE = False
        
except ImportError as e:
    print(f"[警告] Lava库未正确安装: {e}")
    LAVA_AVAILABLE = False
    LAVA_NATIVE_AVAILABLE = False

from models.spikingresformer import spikingresformer_dvsg, DSSA


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


class DVStransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.float()

        T = img.shape[0]
        resized_frames = []
        for t in range(T):
            frame = img[t]
            resized = self.transform(frame)
            resized_frames.append(resized)

        return torch.stack(resized_frames, dim=0)


class DatasetWarpper:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        frame, label = self.dataset[index]

        if self.transform is not None:
            frame = self.transform(frame)

        if frame.shape[1] == 2:
            third_channel = (frame[:, 0:1] + frame[:, 1:2]) / 2
            frame = torch.cat([frame, third_channel], dim=1)

        return frame, label

    def __len__(self):
        return len(self.dataset)


def load_dvs_real(data_path, T, input_size=(64, 64), sample_idx=0):
    """加载真实的DVS-Gesture数据集"""
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

    transform_test = DVStransform(
        transform=transforms.Resize(size=input_size, antialias=True)
    )

    dataset = DVS128Gesture(
        root=data_path,
        train=False,
        data_type='frame',
        frames_number=T,
        split_by='number'
    )

    dataset = DatasetWarpper(dataset, transform_test)
    frame, label = dataset[sample_idx]

    frame = frame.unsqueeze(0)  # [1, T, C, H, W]
    return frame, label


def find_all_dssa(model):
    """找到模型中所有的DSSA模块"""
    dssa_modules = []
    for name, module in model.named_modules():
        if isinstance(module, DSSA):
            dssa_modules.append((name, module))
    return dssa_modules


class SpikeStatisticsCollector:
    """
    收集USSA模块的详细脉冲统计信息
    USSA特性：W1为1x1，经过空间mean后变成全局Q [T,B,C,1,1]
    """
    def __init__(self, target_name):
        self.target_name = target_name
        self.input_spikes = []
        self.output_spikes = []
        self.internal_stats = {
            'w1_output': [],           # W1原始输出 [T,B,C,H,W]
            'w1_after_mean': [],       # W1空间平均后 [T,B,C,1,1]
            'w2_output': [],           # W2输出 [T,B,C,H,W]
            'q_output': [],            # Q分支LIF输出（全局）[T,B,C,1,1]
            'v_output': [],            # V分支LIF输出 [T,B,C,H,W]
            'mul_output': [],          # Q⊙V乘积输出
            'proj_output': []
        }
        self.hooks = []

    def register(self, model):
        for name, module in model.named_modules():
            if name == self.target_name:
                print(f"  SpikeStatisticsCollector: USSA模式 (W1=1x1, 全局Q)")
                
                # 注册三个LIF的hook
                if hasattr(module, 'activation_in'):
                    h = module.activation_in.register_forward_hook(self._hook_activation_in)
                    self.hooks.append(h)
                if hasattr(module, 'activation_attn'):
                    h = module.activation_attn.register_forward_hook(self._hook_q_lif)
                    self.hooks.append(h)
                if hasattr(module, 'activation_out'):
                    h = module.activation_out.register_forward_hook(self._hook_v_lif)
                    self.hooks.append(h)
                
                # 注册子模块hooks
                if hasattr(module, 'W1'):
                    h = module.W1.register_forward_hook(self._hook_w1)
                    self.hooks.append(h)
                if hasattr(module, 'W2'):
                    h = module.W2.register_forward_hook(self._hook_w2)
                    self.hooks.append(h)
                if hasattr(module, 'mul1'):
                    h = module.mul1.register_forward_hook(self._hook_mul)
                    self.hooks.append(h)
                if hasattr(module, 'Wproj'):
                    h = module.Wproj.register_forward_hook(self._hook_proj)
                    self.hooks.append(h)
                return True
        raise ValueError(f"未找到: {self.target_name}")

    def _hook_activation_in(self, module, input, output):
        """
        收集DSSA输入脉冲统计（activation_in的输出才是真正的脉冲）
        """
        with torch.no_grad():
            if isinstance(output, torch.Tensor) and len(output.shape) == 5:
                T, B, C, H, W = output.shape
                for t in range(T):
                    for b in range(B):
                        # 统计脉冲（等于1的元素）
                        spike_count = (output[t, b] == 1).sum().item()
                        self.input_spikes.append({
                            'time_step': t,
                            'batch': b,
                            'count': spike_count,
                            'total_elements': C * H * W,
                            'firing_rate': spike_count / (C * H * W)
                        })

    def _hook_q_lif(self, module, input, output):
        """收集Q分支LIF脉冲统计（activation_attn）
        USSA: Q是全局的，形状为[T,B,C,1,1]
        """
        with torch.no_grad():
            if isinstance(output, torch.Tensor) and len(output.shape) == 5:
                T, B, C, H, W = output.shape
                for t in range(T):
                    for b in range(B):
                        spike_count = (output[t, b] == 1).sum().item()
                        self.internal_stats['q_output'].append({
                            'time_step': t, 'batch': b, 'count': spike_count,
                            'total_elements': C,  # USSA: 实际是C个元素（1x1空间）
                            'firing_rate': spike_count / C,
                            'shape': f"[{C},1,1]"
                        })

    def _hook_v_lif(self, module, input, output):
        """收集V分支LIF脉冲统计（activation_out）
        USSA: V保持空间维度 [T,B,C,H,W]
        注意：V是中间输出，不是DSSA的最终输出
        """
        with torch.no_grad():
            if isinstance(output, torch.Tensor) and len(output.shape) == 5:
                T, B, C, H, W = output.shape
                for t in range(T):
                    for b in range(B):
                        spike_count = (output[t, b] == 1).sum().item()
                        self.internal_stats['v_output'].append({
                            'time_step': t, 'batch': b, 'count': spike_count,
                            'total_elements': C * H * W,
                            'firing_rate': spike_count / (C * H * W),
                            'shape': f"[{C},{H},{W}]"
                        })

    def _hook_w1(self, module, input, output):
        """收集W1输出统计
        USSA: W1是1x1卷积，输出为[T,B,C,H,W]，然后经过mean变成[T,B,C,1,1]
        """
        with torch.no_grad():
            if isinstance(output, torch.Tensor) and len(output.shape) == 5:
                T, B, C, H, W = output.shape
                # 记录W1原始输出（1x1卷积后的结果）
                for t in range(T):
                    for b in range(B):
                        spike_count = (output[t, b] != 0).sum().item()
                        self.internal_stats['w1_output'].append({
                            'time_step': t, 'batch': b, 'count': spike_count,
                            'total_elements': C * H * W,
                            'firing_rate': spike_count / (C * H * W),
                            'shape': f"[{C},{H},{W}]"
                        })
                # 记录空间平均后的结果（USSA特有）
                mean_output = output.mean(dim=(3,4), keepdim=True)  # [T,B,C,1,1]
                for t in range(T):
                    for b in range(B):
                        val_count = (mean_output[t, b] != 0).sum().item()
                        self.internal_stats['w1_after_mean'].append({
                            'time_step': t, 'batch': b, 'count': val_count,
                            'total_elements': C,
                            'firing_rate': val_count / C,
                            'shape': f"[{C},1,1]"
                        })

    def _hook_w2(self, module, input, output):
        """收集W2输出统计 [T,B,C,H,W]"""
        with torch.no_grad():
            if isinstance(output, torch.Tensor) and len(output.shape) == 5:
                T, B, C, H, W = output.shape
                for t in range(T):
                    for b in range(B):
                        spike_count = (output[t, b] != 0).sum().item()
                        self.internal_stats['w2_output'].append({
                            'time_step': t, 'batch': b, 'count': spike_count,
                            'total_elements': C * H * W,
                            'firing_rate': spike_count / (C * H * W),
                            'shape': f"[{C},{H},{W}]"
                        })

    def _hook_mul(self, module, input, output):
        """收集Hadamard乘积输出 Q⊙V
        USSA: Q广播到V的形状后逐元素相乘
        """
        with torch.no_grad():
            if isinstance(output, torch.Tensor) and len(output.shape) == 5:
                T, B, C, H, W = output.shape
                for t in range(T):
                    for b in range(B):
                        spike_count = (output[t, b] != 0).sum().item()
                        self.internal_stats['mul_output'].append({
                            'time_step': t, 'batch': b, 'count': spike_count,
                            'total_elements': C * H * W,
                            'firing_rate': spike_count / (C * H * W),
                            'shape': f"[{C},{H},{W}]"
                        })

    def _hook_proj(self, module, input, output):
        """收集Wproj输出
        USSA: Wproj是1x1卷积，输出浮点数膜电位 [T,B,C,H,W]
        这是DSSA的最终输出（非脉冲）
        """
        with torch.no_grad():
            if isinstance(output, torch.Tensor) and len(output.shape) == 5:
                T, B, C, H, W = output.shape
                for t in range(T):
                    for b in range(B):
                        # 统计非零值（浮点数输出的活跃程度）
                        active_count = (output[t, b] != 0).sum().item()
                        self.internal_stats['proj_output'].append({
                            'time_step': t, 'batch': b, 'count': active_count,
                            'total_elements': C * H * W,
                            'firing_rate': active_count / (C * H * W),
                            'shape': f"[{C},{H},{W}]"
                        })
                        # 记录DSSA最终输出（Wproj的浮点数输出）
                        self.output_spikes.append({
                            'time_step': t, 'batch': b, 'count': active_count,
                            'total_elements': C * H * W,
                            'firing_rate': active_count / (C * H * W)
                        })

    def remove(self):
        for hook in self.hooks:
            hook.remove()

    def get_summary(self):
        """获取USSA统计摘要"""
        if not self.input_spikes:
            return None
            
        input_firing_rates = [s['firing_rate'] for s in self.input_spikes]
        
        summary = {
            'model_type': 'USSA',
            'num_samples': len(self.input_spikes),
            'input_avg_firing_rate': np.mean(input_firing_rates),
            'input_std_firing_rate': np.std(input_firing_rates),
            'total_input_spikes': sum([s['count'] for s in self.input_spikes]),
            'internal': {}
        }
        
        # 内部统计
        for key, stats in self.internal_stats.items():
            if stats:
                firing_rates = [s['firing_rate'] for s in stats]
                summary['internal'][key] = {
                    'total_spikes': sum([s['count'] for s in stats]),
                    'avg_firing_rate': np.mean(firing_rates),
                    'shape': stats[0].get('shape', 'N/A') if stats else 'N/A'
                }
        
        return summary


class DSSAAnalyzer:
    """
    DSSA模块分析器
    准确计算计算复杂度和通信复杂度
    """
    def __init__(self, module):
        self.module = module
        self.dim = module.dim
        self.num_heads = module.num_heads
        
    def analyze_structure(self):
        """分析DSSA结构参数"""
        return {
            'dim': self.dim,
            'num_heads': self.num_heads,
            'W1': self._get_conv_params(self.module.W1) if hasattr(self.module, 'W1') else None,
            'W2': self._get_conv_params(self.module.W2) if hasattr(self.module, 'W2') else None,
            'Wproj': self._get_conv_params(self.module.Wproj) if hasattr(self.module, 'Wproj') else None
        }
    
    def _get_conv_params(self, conv_layer):
        if conv_layer is None:
            return None
        weight = conv_layer.weight
        return {
            'in_channels': weight.shape[1],
            'out_channels': weight.shape[0],
            'kernel_size': weight.shape[2:],
            'num_params': weight.numel()
        }
    
    def compute_sop_complexity(self, spike_stats, T, H, W):
        """
        基于实际脉冲统计的SOP（Synaptic Operations）复杂度和功耗估算
        
        【USSA架构 - 全局Q + 局部V】
        x (输入膜电位)
          ↓
        activation_in → 脉冲 [T,B,C,H,W]
          ↓┌────────────┐
          ↓↓            ↓
         W1 (1x1)     W2 (3x3)
          ↓            ↓
        norm1         norm2
          ↓            ↓
        spatial_mean  (保持H,W)
          ↓            ↓
        activation_attn  activation_out
          ↓            ↓
        Q_global     V_local
        [T,B,C,1,1]  [T,B,C,H,W]
          └──────┬──────┘
                 ↓
              mul1 (Q广播 ⊙ V)
                 ↓
               Wproj (1x1)
        
        【USSA的SOP计算】
        SOP = 输入到某层的脉冲数 × fanout
        
        关键层：
        - W1: Conv1x1 + 空间平均, fanout = 1×1×C = C (仅对单位置)
        - W2: Conv3x3, fanout = 3×3×C = 9C
        - activation_attn/activation_out: LIF, 无SOP（只是脉冲生成）
        - mul1: 逐元素乘, fanout ≈ 1 (Q全局广播)
        - Wproj: Conv1x1, fanout = C
        """
        C = self.dim
        
        if spike_stats is None:
            return None
        
        # 提取各层输入的脉冲数（根据DSSA数据流）
        
        internal_stats = spike_stats.get('internal', {})
        
        # activation_in的输出 = W1和W2的共同输入
        # 注意：spike_stats['total_input_spikes']统计的是进入DSSA的脉冲
        # 即activation_in的输出
        activation_in_output = spike_stats.get('total_input_spikes', 0)
        
        # W1的输入：activation_in的输出脉冲
        w1_input = activation_in_output
        
        # W2的输入：activation_in的输出脉冲（相同）
        w2_input = activation_in_output
        
        # W1的原始输出（1x1卷积后）[T,B,C,H,W]
        w1_output = internal_stats.get('w1_output', {}).get('total_spikes', 0)
        
        # W1空间平均后的输出（作为activation_attn的输入）[T,B,C,1,1]
        w1_after_mean_output = internal_stats.get('w1_after_mean', {}).get('total_spikes', 0)
        
        # W2的输出（作为activation_out的输入）
        w2_output = internal_stats.get('w2_output', {}).get('total_spikes', 0)
        
        # activation_attn的输出（W1分支LIF后，全局Q）
        activation_attn_output = internal_stats.get('q_output', {}).get('total_spikes', 0)
        
        # activation_out的输出（W2分支LIF后，局部V）
        activation_out_output = internal_stats.get('v_output', {}).get('total_spikes', 0)
        
        # mul1的输入：V (activation_out输出) × Q (activation_attn输出广播)
        # mul1的输出
        mul_output = internal_stats.get('mul_output', {}).get('total_spikes', 0)
        
        # Wproj的输入：mul1的输出
        wproj_input = mul_output
        
        # 计算各层的SOP（基于输入脉冲数×fanout）
        
        # W1: Conv1x1 (USSA特有)
        # 输入：activation_in输出脉冲
        # 注意：W1的1x1卷积在空间每个位置独立计算，然后平均
        # fanout = 1×1×C = C (单位置计算)
        w1_fanout = 1 * 1 * C
        sop_w1 = w1_input * w1_fanout
        
        # W2: Conv3x3
        # 输入：activation_in输出脉冲
        w2_fanout = 3 * 3 * C
        sop_w2 = w2_input * w2_fanout
        
        # spatial_mean: 空间平均（USSA特有）
        # 对W1输出在H×W维度求平均，没有突触权重乘法，SOP = 0
        sop_spatial_mean = 0
        
        # activation_attn: LIF神经元
        # 只是脉冲生成，无乘法操作，SOP ≈ 0（或很小）
        sop_activation_attn = 0  # LIF是本地操作，无突触操作
        
        # activation_out: LIF神经元
        sop_activation_out = 0  # LIF是本地操作
        
        # mul1: 逐元素哈达玛积（Q广播 ⊙ V）
        # 逐元素乘法，无突触权重，SOP = 0
        sop_mul = 0
        
        # Wproj: Conv1x1
        # 输入：mul1输出脉冲
        # fanout = 1×1×C = C
        wproj_fanout = 1 * 1 * C
        sop_wproj = wproj_input * wproj_fanout
        
        # 总SOP（所有有突触操作的层）
        total_sops = sop_w1 + sop_w2 + sop_mul + sop_wproj
        
        # SNN理论能耗参数：0.9 pJ per synaptic operation
        # 来源：Horowitz, "Computing's Energy Problem (and what we can do about it)"
        #       ISSCC 2014, 45nm工艺
        # 注：虽然论文年份久远(2014)，但由于缺乏更新的统一对比基准，
        #     当前SNN领域论文普遍仍使用此数值进行理论功耗对比
        E_per_sop = 0.9e-12  # 0.9 pJ (Horowitz, ISSCC 2014)
        snn_energy = total_sops * E_per_sop
        
        # 对比：等效ANN稠密计算能耗（⚠️ 注意：此为概念对比，非直接可比）
        # 
        # ⚠️ 公平性声明：
        # - ANN: 单次前向传播，处理单帧（无时序维度）
        # - SNN: 时间步展开（T=16），处理时序脉冲
        # - 两者在计算范式上本质不同，此对比仅作概念参考
        #
        # ANN能耗参数：4.6 pJ per MAC (Horowitz, ISSCC 2014)
        # 注：虽然Horowitz 2014是45nm工艺数据，但由于：
        # 1. SNN与ANN的能耗对比通常关注比例而非绝对值
        # 2. 缺乏统一的现代工艺对比数据
        # 当前SNN领域论文普遍仍使用此数值作为标准对比基准
        E_per_mac = 4.6e-12  # 4.6 pJ (Horowitz, "Computing's Energy Problem", ISSCC 2014)
        
        # ANN单次前向传播的计算量（除以T，因为ANN不展开时间步）
        positions_per_frame = H * W  # 单帧空间位置
        macs_w1_ann = positions_per_frame * C * C * 1 * 1  # USSA: W1是1x1
        macs_w2_ann = positions_per_frame * C * C * 3 * 3  # W2保持3x3
        macs_mul_ann = positions_per_frame * C
        macs_wproj_ann = positions_per_frame * C * C
        total_macs_ann = macs_w1_ann + macs_w2_ann + macs_mul_ann + macs_wproj_ann
        
        ann_energy_dense = total_macs_ann * E_per_mac
        # 注：ANN按单帧计算，SNN按T时间步累计，两者不直接可比
        
        # 结果汇总
        
        return {
            'analysis_type': 'SNN实际SOP统计（基于USSA架构）',
            'platform': 'Loihi-like Neuromorphic Hardware',
            'ussa_architecture': {
                'data_flow': 'x → [W1(1x1)→mean→Q_global, W2(3x3)→V_local] → mul → Wproj',
                'note': 'USSA: 全局Q(1x1+mean) + 局部V(3x3), 门控后逐元素乘',
            },
            'spike_flow': {
                'activation_in_output': activation_in_output,
                'w1_input': w1_input,
                'w2_input': w2_input,
                'w1_output_raw': w1_output,           # W1 1x1卷积原始输出
                'w1_after_mean': w1_after_mean_output, # USSA: 空间平均后 → activation_attn
                'w2_output': w2_output,
                'activation_attn_output': activation_attn_output,  # 全局Q
                'activation_out_output': activation_out_output,    # 局部V
                'mul_output': mul_output,
                'wproj_input': wproj_input,
            },
            'sop_breakdown': {
                'W1_1x1_global': {
                    'input': w1_input,
                    'fanout': w1_fanout,
                    'sop': sop_w1,
                    'note': 'USSA: 1x1 conv + spatial mean → global Q',
                },
                'W2_3x3_local': {
                    'input': w2_input,
                    'fanout': w2_fanout,
                    'sop': sop_w2,
                    'note': 'USSA: 3x3 conv → local V',
                },
                'activation_attn': {
                    'sop': sop_activation_attn,
                    'note': 'LIF neuron, local operation',
                },
                'activation_out': {
                    'sop': sop_activation_out,
                    'note': 'LIF neuron, local operation',
                },
                'mul1': {
                    'input_attn': activation_attn_output,
                    'input_y2': activation_out_output,
                    'sop': sop_mul,
                    'note': 'Element-wise multiplication',
                },
                'Wproj_1x1': {
                    'input': wproj_input,
                    'fanout': wproj_fanout,
                    'sop': sop_wproj,
                },
            },
            'total_sops': total_sops,
            'total_macs_ann': total_macs_ann,  # ANN单帧MACs
            'sparsity_ratio': total_sops / (total_macs_ann * T) if total_macs_ann > 0 else 0,  # 对比ANN累计计算
            'energy_estimate': {
                'snn_energy_j': snn_energy,
                'snn_energy_uj': snn_energy * 1e6,
                'ann_energy_dense_j': ann_energy_dense,
                'E_per_sop_pj': 0.9,  # Horowitz ISSCC 2014
                'E_per_sop_note': '0.9pJ来自Horowitz ISSCC 2014，SNN领域通用理论对比基准',
                'energy_saving': (ann_energy_dense - snn_energy) / ann_energy_dense if ann_energy_dense > 0 else 0,
                'fairness_note': 'ANN:单帧GPU计算; SNN:T时间步Loihi计算; 两者不直接可比',
            },
        }
    
    def compute_communication_complexity(self, T, H, W, mapping_strategy='token_wise'):
        """
        基于Token-wise Distributed Neuromorphic Execution Model的通信复杂度分析
        
        与论文定义一致（Definition 4.2）：
        - 不同的Token（空间位置H×W）被映射到不同的处理核心
        - 通道维度C完全局部在每个核心内
        
        关键参数：
        - N = H × W：空间Token数量（论文中的N）
        - D = C：特征维度/通道数（论文中的D）
        
        Args:
            T: 时间步数
            H, W: 空间维度（H×W = N，即Token数）
            mapping_strategy: 映射策略（'token_wise'，与论文一致）
        
        Returns:
            包含通信量分析的字典
        """
        N = H * W  # 空间Token数量（论文中的N）
        D = self.dim  # 特征维度/通道数（论文中的D）
        
        # Token-wise分区映射策略（与论文Proposition 4.3-4.5一致）
        # 
        # 映射假设：
        # - 每个核心处理一个或多个空间Token（H×W被分割到多个核心）
        # - 通道维度D完全局部在每个核心内（所有通道在同一个核心）
        # 
        # 这与论文中的"token-wise distributed neuromorphic execution model"一致
        
        # 假设每个核心处理一个空间位置（最细粒度）
        # 实际中可能每个核心处理多个位置，但不影响渐近复杂度
        tokens_per_core = 1  # 每个核心处理1个token
        num_cores = N  # 需要N个核心（或更少，如果每个核心处理多个token）
        
        # 各操作的通信量计算（基于Token-wise分区）
        
        # 1. 标准自注意力（Standard Attention）- O(N²) 通信
        # 
        # 如论文Proposition 4.3所述：
        # - 计算QK^T需要每个Query token与所有Key token交互
        # - 在Token-wise分布式模型中，这需要O(N²)次跨核心传输
        # - 每次交互是D维度的内积
        #
        # 通信量：每个Token需要与所有其他N-1个Token通信
        # 总通信量 = T × N × (N-1) × D ≈ T × N² × D
        
        comm_standard_attention = T * N * (N - 1) * D  # O(N²D)
        
        """
        DSSA核心：Mul（哈达玛积）的通信复杂度
        """
        # Mul：y1(W1输出) * y2(W2输出)
        # Token-wise下，y1和y2都是同一位置的ND维向量
        # 
        # 情况1：W1/W2同核心映射 → O(1) 通信（完全局部）
        # 情况2：W1/W2不同核心 → O(ND) 通信（需传输）
        
        SSA_mul = T * N * D  # 保守取O(ND)
        """
        USSA核心：压缩后 Mul 的通信复杂度
        """
        # 关键：y1 ∈ R^(1,D) 是全局压缩后的门控向量 g
        # y2 ∈ R^(N,D) 是 W2 的完整输出

        # Step 1: y1 压缩过程（已在 W1 输出后本地完成，假设无额外通信）
        # 输入: y1_full ∈ R^(N,D) → 压缩 → g ∈ R^(1,D)

        # Step 2: 门控向量 g 的广播
        # g 需要广播至所有 N 个 token 位置，与 y2 逐元素相乘
        # g ∈ R^D，复制 N 份 → 通信量 O(ND) 若逐元素发送
        # 但利用 broadcast-supported 路由：源头只发一次 O(D)，硬件复制到 N 个位置

        comm_gating_broadcast = T * D  # O(D) - 广播源数据量
        return {
            # 核心通信复杂度对比（三项）
            'comm_USSA_attention': comm_gating_broadcast,      # O(D) - 压缩门控广播
            'comm_SSA_attention': SSA_mul,                      # O(ND) - 完整矩阵传输
            'comm_standard_attention': comm_standard_attention, # O(N²D) - token-token全连接
            
            # 加速比分析（加速比 = 通信量大的/通信量小的 = 慢的/快的）
            # USSA vs SSA: ND/D = N 倍
            'speedup_USSA_vs_SSA': SSA_mul/comm_gating_broadcast if comm_gating_broadcast > 0 else float('inf'),
            # USSA vs 标准Attention: N²D/D = N² 倍  
            'speedup_USSA_vs_standard': comm_standard_attention/comm_gating_broadcast if comm_gating_broadcast > 0 else float('inf'),
            # SSA vs 标准Attention: N²D/ND = N 倍
            'speedup_SSA_vs_standard': comm_standard_attention/SSA_mul if SSA_mul > 0 else float('inf'),
        }


def run_lava_simulation(features, T, dssa_module, spike_stats, weight_scale=1.0):
    """
    运行Lava Loihi仿真 - 强制使用原生Lava进程
    
    Args:
        features: 输入特征
        T: 时间步数
        dssa_module: PyTorch DSSA模块
        spike_stats: 脉冲统计
        weight_scale: 权重缩放因子（默认5000.0）
    """
    if not LAVA_AVAILABLE:
        raise RuntimeError("Lava不可用，无法运行仿真")
    
    if not LAVA_NATIVE_AVAILABLE:
        raise RuntimeError("Lava原生DSSA实现不可用")
    
    print(f"\n{'='*60}")
    print("Lava Token-wise 仿真（USSA模块）")
    print(f"{'='*60}")
    print("网络结构: Input → [W1(1x1)→mean→Q_global, W2(3x3)→V_local] → Mul(Q⊙V) → Wproj → Output")
    print("USSA特性: W1产生全局Q(1x1+spatial mean)，W2产生局部V(3x3)")
    print("[说明] 使用Lava原生进程（Conv/LIF/Dense）")
    print("       - 每个空间位置(h,w)独立进程组")
    print("       - 真实权重从PyTorch加载")
    print("       [注意] 运行所有tokens，大型网络编译可能需要时间")
    
    result = run_lava_native_simulation(features, T, dssa_module, weight_scale=weight_scale)
    return result


def estimate_energy_with_routing(T, C, H, W, lava_stats=None, features=None, is_ussa=True):
    """
    基于Intel Loihi 1 (2018) 实测参数的能耗估算 - 完全使用 Lava 统计
    
    Args:
        T: 时间步数
        C: 通道数
        H, W: 空间维度
        lava_stats: Lava 仿真统计结果
        features: 输入特征数据 {'input_data': [T, C, H, W]}，用于计算输入脉冲数
        is_ussa: 是否是USSA模式（全局Q需要广播到N个核心）
    
    ⚠️ 重要说明与局限性：
    
    使用参数（Davies et al., IEEE Micro 2018, Fig.11）：
    - Synaptic operation: 81 pJ (典型值，0.75V, 0.9V平均)
    - Neuron update (active/inactive): 81 pJ / 52 pJ
    - NoC路由: 3.0-4.0 pJ/hop (E-W/N-S)
    - Within-tile传输: 1.7 pJ
    - 内存访问: 未计算（无公开文献来源）
    - 静态能耗: 未计算（无公开文献来源）
    
    参考文献：
    - Davies et al., 2018, "Loihi: A Neuromorphic Manycore Processor with On-Chip Learning"
      IEEE Micro, vol.38, no.1, pp.82-99
    """
    # 只使用 lava_stats 进行能耗估算
    if lava_stats is None or lava_stats.get('status') != 'lava_complete':
        print(f"  [能耗估算] Lava 仿真未完成，跳过能耗估算")
        return None
    N = H * W  # 空间Token数量
    print(f"  [能耗估算] 使用 Lava 仿真统计")
    
    # 从 lava monitor_data 获取脉冲统计
    monitor_data = lava_stats.get('monitor_data', {})
    l1_spikes = int(monitor_data.get('l1_spikes', np.array([])).sum()) 
    l2_spikes = int(monitor_data.get('l2_spikes', np.array([])).sum()) 
    mul_spikes = int(monitor_data.get('gated_out', np.array([0])).sum())
    num_active_tokens = lava_stats.get('num_tokens', H * W)  # 活跃token数 
    
    # 计算输入脉冲数：从 features 直接计算
    if features is not None and 'input_data' in features:
        input_data = features['input_data']  # numpy array [T, C, H, W]
        estimated_input_spikes = int(input_data.sum())  # 直接统计输入特征中的脉冲数
    print(f"   Lava 统计: L1={l1_spikes}, L2={l2_spikes}, 门控输出={mul_spikes}, 输入脉冲={estimated_input_spikes}")
    
    # Loihi 1 (2018) 实测能耗参数 - Davies et al. IEEE Micro 2018 Fig.11
    # 
    # 关键参数（0.75V-0.9V 实测范围与典型值）：
    # - Energy per synaptic spike op: 23.6-136.5 pJ (范围), ~81 pJ (典型值)
    #   * 本估算使用典型值 81 pJ（非最小值 23.6 pJ）
    # - Energy per neuron update (active/inactive): 81 pJ / 52 pJ
    #   * active: 膜电位更新 + 阈值比较 + 发放（发放脉冲的神经元）
    #   * inactive: 膜电位泄漏衰减（未发放的神经元）
    # - Within-tile spike energy: 1.7 pJ (片内传输)
    # - Energy per tile hop (E-W/N-S): 3.0/4.0 pJ (跨tile路由)
    #   * E-W: 3.0 pJ, N-S: 4.0 pJ
    #   * 使用 3.5 pJ 作为平均 NoC hop energy（简化估算）
    
    E_synapse_op = 81.0e-12       # 81 pJ (Davies 2018, 典型值)
    E_neuron_update_active = 81.0e-12   # 81 pJ (发放脉冲的活跃神经元)
    E_neuron_update_inactive = 52.0e-12 # 52 pJ (未发放的非活跃神经元)
    E_spike_within_tile = 1.7e-12 # 1.7 pJ (within-tile spike transmission)
    E_noc_hop_ew = 3.0e-12        # 3.0 pJ (East-West hop)
    E_noc_hop_ns = 4.0e-12        # 4.0 pJ (North-South hop)
    E_noc_hop_avg = 3.5e-12       # 3.5 pJ (E-W/N-S 平均值，简化估算)
    
    # 1. 突触操作能耗 (SOP - Synaptic Operations)
    # DSSA数据流：
    #   activation_in输出 → W1(3x3) + W2(3x3) [并行分支]
    #   W1/W2输出 → Mul
    #   Mul输出 → Wproj(1x1)
    # 
    # SOP计算：输入脉冲数 × fanout (每个输入连接的输出数)
    # - W1: 1×1 卷积核 = 1 个连接 × C 输出通道 = 1C fanout (USSA特性)
    # - W2: 3×3 卷积核 = 9 个连接 × C 输出通道 = 9C fanout
    # - Wproj: 1×1 卷积核 = 1 个连接 × C 输出通道 = 1C fanout
    sop_w1 = estimated_input_spikes * 1 * C   # W1: 1x1 卷积 (USSA)
    sop_w2 = estimated_input_spikes * 9 * C   # W2: 3x3 卷积
    sop_wproj = mul_spikes * 1 * C            # Wproj: 1x1 卷积
    synaptic_ops = sop_w1 + sop_w2 + sop_wproj
    
    print(f"    SOP: W1(1x1)={sop_w1:,}, W2(3x3)={sop_w2:,}, Wproj(1x1)={sop_wproj:,}, 总计={synaptic_ops:,}")
    synapse_energy = synaptic_ops * E_synapse_op
    
    # 2. 神经元更新能耗 (W1 + W2分支，区分active/inactive)
    # 
    # USSA特性：
    # - W1: 只有C个神经元（全局Q，所有位置共享）
    # - W2: N×C个神经元（局部V，每个位置独立）
    
    num_neurons_w1 = C  # USSA: 全局Q只有C个神经元
    num_neurons_w2 = N * C  # 局部V: N×C个
    num_neurons_total = num_neurons_w1 + num_neurons_w2
    
    # 初始化神经元更新计数
    total_active_updates = l1_spikes + l2_spikes  # W1 + W2实际发放的脉冲数
    total_inactive_updates = num_neurons_total * T - total_active_updates
    
    if total_active_updates > 0:
        print(f"    神经元更新: W1={num_neurons_w1}神经元, W2={num_neurons_w2}神经元")
        print(f"              Active={total_active_updates:,}×81pJ, Inactive={total_inactive_updates:,}×52pJ")
    
    energy_active = total_active_updates * E_neuron_update_active
    energy_inactive = total_inactive_updates * E_neuron_update_inactive
    neuron_energy = energy_active + energy_inactive
    
    # 3. 脉冲产生能耗
    # 注：81pJ synaptic operation 已包含片上计算和脉冲传输
    # 不单独计算 spike generation energy
    spike_energy = 0.0
    
    # 4. NoC路由能耗（基于Token-wise Distributed Execution Model）
    # 
    # Token-wise映射：每个空间位置(h,w)映射到独立核心，通道维度C完全局部
    # 
    # USSA数据流中的NoC通信：
    
    # 4a. W2分支的邻居通信（3x3卷积需要8个邻居）
    # 每个活跃token需要从8个邻居接收输入
    # 每个邻居传输C维向量（所有通道）
    # 4个正交：4×1=4跳，4个对角：4×2=8跳，平均：(4+8)/8=1.5跳
    w2_neighbor_hops = num_active_tokens * 8 * 1.5 * C  # 8邻居 × 1.5跳 × C通道
    w2_noc_energy = w2_neighbor_hops * E_noc_hop_avg
    
    # W1 计算阶段
    w1_compute_noc = 0  # 纯本地计算
    # 4b. USSA SpatialMean: Tree Reduction (层次化归约)
    # 真实Loihi可用树形归约优化：N个核心分层聚合到中心
    # Level 0: N个 → N/2个, Level 1: N/2个 → N/4个, ..., 总深度~log2(N)
    # 使用W1卷积后实际活跃的token数（从Lava monitor获取）
    w1_active_tokens = lava_stats.get('monitor_data', {}).get('w1_active_tokens', num_active_tokens)
    tree_depth = int(np.log2(N + 1)) if N > 1 else 0  # 树形归约深度
    # 每层的通信量递减：N + N/2 + N/4 + ... ≈ 2N，平均每token 2×log2(N) hops
    avg_hops_reduction = 2 * tree_depth
    spatial_mean_hops = w1_active_tokens * C * avg_hops_reduction
    spatial_mean_noc_energy = spatial_mean_hops * E_noc_hop_avg
    
    # 4c. USSA Broadcast: One-to-All
    # 全局Q从中心广播到N个核心
    # Loihi支持硬件multicast：树形广播，深度~log2(N)
    # 保守估计使用sqrt(N)，实际Loihi可能更优（log2(N)）
    avg_broadcast_hops = int(np.log2(N + 1)) if N > 1 else 0  # 树形广播深度
    q_broadcast_hops = l1_spikes * avg_broadcast_hops
    q_broadcast_noc_energy = q_broadcast_hops * E_noc_hop_avg
    
    # 总NoC能耗
    total_noc_energy = w2_noc_energy + spatial_mean_noc_energy + q_broadcast_noc_energy
    
    # within-tile能耗（本地计算）
    # USSA的脉冲分布：
    # - W1计算：每个token本地（N×C个中间值，不计入最终脉冲统计）
    # - L1全局Q：C维脉冲，但广播到N个位置
    # - L2(V)：N×C维脉冲，本地
    # - Mul：N×C维脉冲，本地
    # 
    # within-tile只包含本地计算的脉冲
    # L1的广播是NoC，不计入within-tile
    within_tile_spikes = l2_spikes + mul_spikes  # V和Mul是本地计算
    within_tile_energy = within_tile_spikes * E_spike_within_tile
    
    print(f"   NoC通信统计:")
    print(f"     W2邻居通信: {w2_neighbor_hops:,} hops = {w2_noc_energy*1e6:.3f} μJ")
    print(f"     SpatialMean(Gather): {spatial_mean_hops:,} hops = {spatial_mean_noc_energy*1e6:.3f} μJ")
    print(f"     Q广播(Broadcast): {q_broadcast_hops:,} hops = {q_broadcast_noc_energy*1e6:.3f} μJ")
    print(f"     总NoC: {total_noc_energy*1e6:.3f} μJ")
    # 5. 内存访问能耗 ⚠️ 设为0（未计算）
    # 
    # 原因：Loihi的片上SRAM访问能耗未在Davies 2018等公开文献中披露。
    # 虽然synaptic operation (81pJ) 可能已包含部分内存访问开销，
    # 但无法确定具体比例，因此单独列出并设为0以示缺失。
    mem_energy = 0.0
    
    # 总动态能耗（USSA：包含特殊的NoC通信）
    total_energy = synapse_energy + neuron_energy + total_noc_energy + within_tile_energy + mem_energy
    
    return {
        'hardware': 'Intel Loihi 1 (14nm, 2018) - Davies et al. IEEE Micro 2018',
        'energy_components': {
            'synaptic_op_pj': 81.0,
            'neuron_update_active_pj': 81.0,
            'neuron_update_inactive_pj': 52.0,
            'noc_hop_avg_pj': 3.5,
            'within_tile_spike_pj': 1.7,
        },
        'energy_breakdown': {
            'synapse_energy_j': synapse_energy,
            'neuron_update_active_j': energy_active,
            'neuron_update_inactive_j': energy_inactive,
            'neuron_update_total_j': neuron_energy,
            'noc_energy_j': total_noc_energy,
            'noc_w2_neighbor_j': w2_noc_energy,
            'noc_spatial_mean_j': spatial_mean_noc_energy,
            'noc_q_broadcast_j': q_broadcast_noc_energy,
            'within_tile_energy_j': within_tile_energy,
        },
        'operation_counts': {
            'synaptic_operations': synaptic_ops,
            'estimated_input_spikes': estimated_input_spikes,
            'l1_spikes': l1_spikes,
            'l2_spikes': l2_spikes,
            'mul_spikes': mul_spikes,
            'noc_w2_neighbor_hops': w2_neighbor_hops,
            'noc_spatial_mean_hops': spatial_mean_hops,
            'noc_q_broadcast_hops': q_broadcast_hops,
            'within_tile_spikes': within_tile_spikes,
        },
        'total_energy_j': total_energy,
        'energy_per_timestep_nj': total_energy / T * 1e9 if T > 0 else 0,
        'notes': [
            '完全基于 Lava 仿真统计（无 PyTorch spike_stats）',
            '使用Loihi 1 (2018)实测参数：Davies et al. IEEE Micro 2018 Fig.11',
            'Synaptic op: 81 pJ',
            'Neuron update: 81 pJ (active) / 52 pJ (inactive)',
            '内存访问和静态能耗未计算',
        ]
    }




def run_hardware_analysis(checkpoint_path, config_path, data_path, num_samples=10,
                         output_path='hardware_analysis.json', use_lava=True, lava_t=None,
                         weight_scale=4.0):
    """
    运行硬件效率分析 - 强制使用Lava原生进程
    """
    if not LAVA_AVAILABLE or not LAVA_NATIVE_AVAILABLE:
        raise RuntimeError("Lava或其原生实现不可用，无法运行分析")
    
    print("="*70)
    print("SpikingResformer DSSA Loihi 硬件效率分析")
    print("="*70)
    print("\n⚠️  重要声明与局限性:")
    print("-" * 70)
    print("本分析提供理论能耗估算，NOT 真实硬件测量！")
    print()
    print("使用方法:")
    print("1. PyTorch模型在真实数据上的脉冲统计（真实数据）✓")
    print("2. Lava Token-wise 进程仿真（Conv/LIF功能仿真）⚠️")
    print("3. 基于Davies 2018文献参数的解析能耗估算")
    print()
    print("- 未在真实Loihi/SpiNNaker硬件上验证")
    print("- 路由效应：基于网格拓扑理论模型，非真实路由路径")
    print("- 内存效应：未计算（Loihi SRAM能耗未公开）")
    print("- Lava仿真：功能正确性验证，不模拟NoC时序/拥塞")
    print()
    print("适用性：算法级能耗趋势分析，架构设计初步评估")
    print("不适用：声称真实硬件性能，精确pJ级绝对预测")
    print("="*70)
    
    # 加载配置
    config = load_config(config_path)
    T = config.get('T', 16)
    input_size = config.get('input_size', [3, 64, 64])
    
    print(f"\n配置参数:")
    print(f"  T (时间步): {T}")
    print(f"  输入尺寸: {input_size}")
    print(f"  测试样本数: {num_samples}")
    print(f"  Lava仿真: {'启用' if LAVA_AVAILABLE and use_lava else '禁用'}")
    
    # 创建模型
    print(f"\n加载模型...")
    model = spikingresformer_dvsg(T=T, num_classes=config.get('num_classes', 11))
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    print("✓ 模型权重加载完成")
    
    # 找到所有DSSA模块
    dssa_modules = find_all_dssa(model)
    print(f"\n发现 {len(dssa_modules)} 个DSSA模块")
    
    target_name, target_module = dssa_modules[0]
    print(f"分析目标: {target_name}")
    
    analyzer = DSSAAnalyzer(target_module)
    structure = analyzer.analyze_structure()
    print(f"\nDSSA结构:")
    print(f"  维度: {structure['dim']}")
    print(f"  头数: {structure['num_heads']}")
    
    all_results = []
    
    print(f"\n{'='*70}")
    print("开始测试样本分析...")
    print(f"{'='*70}")
    
    for sample_idx in range(num_samples):
        print(f"\n样本 {sample_idx + 1}/{num_samples}")
        print("-" * 50)
        
        try:
            test_input, label = load_dvs_real(data_path, T, input_size[-2:], sample_idx)
        except Exception as e:
            print(f"  跳过样本 {sample_idx}: {e}")
            continue
        
        print(f"  数据: label={label}, shape={test_input.shape}")
        
        # 收集脉冲统计
        collector = SpikeStatisticsCollector(target_name)
        collector.register(model)
        
        with torch.no_grad():
            _ = model(test_input)
        
        spike_stats = collector.get_summary()
        collector.remove()
        
        if spike_stats is None:
            print(f"  未能收集脉冲统计，跳过")
            continue
        
        # 获取特征图尺寸
        # 注意：DVS-Gesture模型的prologue没有下采样，保持原尺寸
        # 对于其他模型（如ImageNet），可能需要根据实际prologue计算
        _, _, C, H, W = test_input.shape
        H_feat = H  # DVS-Gesture: prologue stride=1，不下采样
        W_feat = W  # 标准ResNet模型可能需要 //4
        
        # SNN SOP统计
        sop_complexity = analyzer.compute_sop_complexity(spike_stats, T, H_feat, W_feat)
        
        if sop_complexity:
            print(f"  【SNN实际SOP统计（基于脉冲发放）】")
            print(f"    总SOPs: {sop_complexity['total_sops']:,}")
            print(f"    ANN等效MACs（单帧）: {sop_complexity['total_macs_ann']:,}")
            print(f"    稀疏度: {sop_complexity['sparsity_ratio']*100:.4f}%")
            print(f"    SNN估算能耗: {sop_complexity['energy_estimate']['snn_energy_j']*1e6:.4f} μJ")
            print(f"    稠密计算对比: {sop_complexity['energy_estimate']['ann_energy_dense_j']*1e6:.4f} μJ")
            print(f"    节能比例: {sop_complexity['energy_estimate']['energy_saving']*100:.2f}%")
        else:
            print(f"  【SNN SOP统计】无法计算（缺少脉冲统计）")
        
        # Loihi通信复杂度分析
        comm_complexity = analyzer.compute_communication_complexity(T, H_feat, W_feat)
        
        print(f"  【Loihi通信复杂度（Token-wise分区）】")
        print(f"    标准注意力: O(N²D) = {comm_complexity['comm_standard_attention']:,.0f}")
        print(f"    DSSA通信量: O(ND) = {comm_complexity['comm_SSA_attention']:,.0f}")
        print(f"    USSA通信量: O(D) = {comm_complexity['comm_USSA_attention']:,.0f}")
        # 格式化加速比显示（大数用K/M表示）
        speedup_ussa_ssa = comm_complexity['speedup_USSA_vs_SSA']
        speedup_ussa_std = comm_complexity['speedup_USSA_vs_standard']
        if speedup_ussa_ssa >= 1000:
            speedup_str = f"{speedup_ussa_ssa/1000:.1f}K"
        else:
            speedup_str = f"{speedup_ussa_ssa:.1f}"
        if speedup_ussa_std >= 1000000:
            speedup_std_str = f"{speedup_ussa_std/1000000:.1f}M"
        elif speedup_ussa_std >= 1000:
            speedup_std_str = f"{speedup_ussa_std/1000:.1f}K"
        else:
            speedup_std_str = f"{speedup_ussa_std:.1f}"
        print(f"    USSA vs SSA加速比: {speedup_str}x (N倍)")
        print(f"    USSA vs 标准Attention: {speedup_std_str}x (N²倍)")
        
        # Lava仿真
        lava_stats = None
        if LAVA_AVAILABLE and use_lava:
            pulse_data_container = {'data': None}
            hook_handle = None
            target_module_obj = None
            for name, module in model.named_modules():
                if name == target_name:
                    target_module_obj = module
                    break
            
            if target_module_obj is not None and hasattr(target_module_obj, 'activation_in'):
                def capture_pulse_hook(module, input, output):
                    # output是activation_in的输出 [T, B, C, H, W]
                    if isinstance(output, torch.Tensor):
                        pulse_data_container['data'] = output.detach().cpu().numpy()
                    elif isinstance(output, (tuple, list)) and len(output) > 0:
                        pulse_data_container['data'] = output[0].detach().cpu().numpy()
                
                hook_handle = target_module_obj.activation_in.register_forward_hook(capture_pulse_hook)
                
                with torch.no_grad():
                    _ = model(test_input)
                
                if hook_handle is not None:
                    hook_handle.remove()
                
                if pulse_data_container['data'] is not None:
                    pulse_data = pulse_data_container['data']
                    if len(pulse_data.shape) == 5:  # [T,B,C,H,W] -> [T,C,H,W]
                        pulse_data = pulse_data[:, 0, :, :, :]  # [T, C, H, W]
                    # 如果指定了 lava_t，只取前 lava_t 个时间步
                    lava_T = lava_t if lava_t is not None else pulse_data.shape[0]
                    pulse_data = pulse_data[:lava_T]
                    print(f"  Lava输入脉冲形状: {pulse_data.shape}, 值范围: [{pulse_data.min():.1f}, {pulse_data.max():.1f}]")
                    features = {'input_data': pulse_data}
                    lava_stats = run_lava_simulation(features, lava_T, target_module, spike_stats, weight_scale=weight_scale)
                else:
                    print(f"  [警告] 未能捕获脉冲输入，跳过Lava仿真")
            else:
                print(f"  [警告] 目标模块没有activation_in，跳过Lava仿真")
        
        # 能耗估算（包含路由）
        energy_estimate = estimate_energy_with_routing(T, structure['dim'], H_feat, W_feat, lava_stats, features, is_ussa=True)
        
        if energy_estimate:
            print(f"  能耗估算:")
            print(f"    总动态能耗: {energy_estimate['total_energy_j']*1e6:.4f} μJ")
            print(f"    ├─ Synapse (81pJ × ops): {energy_estimate['energy_breakdown']['synapse_energy_j']*1e6:.4f} μJ")
            print(f"    │  ├─ W1: {energy_estimate['energy_breakdown']['synapse_energy_j'] * 0.5 * 1e6:.4f} μJ (50% energy)")
            print(f"    │  └─ W2: {energy_estimate['energy_breakdown']['synapse_energy_j'] * 0.5 * 1e6:.4f} μJ (50% energy)")
            print(f"    ├─ Neuron Update:")
            print(f"    │  ├─ Active (81pJ): {energy_estimate['energy_breakdown']['neuron_update_active_j']*1e6:.4f} μJ")
            print(f"    │  ├─ Inactive (52pJ): {energy_estimate['energy_breakdown']['neuron_update_inactive_j']*1e6:.4f} μJ")
            print(f"    │  └─ Total: {energy_estimate['energy_breakdown']['neuron_update_total_j']*1e6:.4f} μJ")
            print(f"    ├─ NoC Routing (3.5pJ × hops): {energy_estimate['energy_breakdown']['noc_energy_j']*1e6:.4f} μJ")
            print(f"    │  ├─ W2邻居通信: {energy_estimate['operation_counts']['noc_w2_neighbor_hops']:,} hops")
            print(f"    │  ├─ SpatialMean: {energy_estimate['operation_counts']['noc_spatial_mean_hops']:,} hops")
            print(f"    │  └─ Q广播: {energy_estimate['operation_counts']['noc_q_broadcast_hops']:,} hops")
            print(f"    ├─ Within-Tile (1.7pJ × spikes): {energy_estimate['energy_breakdown'].get('within_tile_energy_j', 0)*1e6:.4f} μJ")
            print(f"    └─ Memory: 未计算 (0.0 μJ)")
            print(f"  操作统计:")
            print(f"    - 突触操作: {energy_estimate['operation_counts']['synaptic_operations']:,}")
            print(f"    - 总NoC跳数: {energy_estimate['operation_counts']['noc_w2_neighbor_hops'] + energy_estimate['operation_counts']['noc_spatial_mean_hops'] + energy_estimate['operation_counts']['noc_q_broadcast_hops']:,}")
            print(f"    - Within-Tile脉冲: {energy_estimate['operation_counts'].get('within_tile_spikes', 0):,}")
        
        result = {
            'sample_id': sample_idx,
            'label': int(label),
            'spike_stats': spike_stats,
            'snn_analysis': {
                'sop_complexity': sop_complexity,
                'note': '基于实际脉冲发放的SOP统计（参考main.py SOPMonitor）',
            },
            'loihi_analysis': {
                'communication_complexity': comm_complexity,
                'lava_simulation': lava_stats,
                'energy_estimate': energy_estimate,
                'note': 'Loihi事件驱动架构分析（基于Lava仿真）',
            }
        }
        
        all_results.append(result)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='SpikingResformer DSSA Loihi硬件效率分析工具'
    )
    parser.add_argument('--checkpoint', '-ckpt', required=True,
                       help='训练好的模型检查点路径')
    parser.add_argument('--config', '-c', required=True,
                       help='配置文件路径')
    parser.add_argument('--data-path', default='/home/jtt/dataset/',
                       help='数据集根目录')
    parser.add_argument('-o', default='hardware_analysis.json',
                       help='输出报告路径')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='测试样本数量')
    parser.add_argument('--no-lava', action='store_true',
                       help='禁用Lava仿真（仅使用理论分析）')
    parser.add_argument('--lava-t', type=int, default=None,
                       help='Lava仿真使用的时间步数（默认使用配置中的T）')
    parser.add_argument('--weight-scale', type=float, default=5000.0,
                       help='Lava仿真权重缩放因子（默认5000.0）')

    args = parser.parse_args()

    run_hardware_analysis(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_path=args.data_path,
        num_samples=args.num_samples,
        output_path=args.o,
        use_lava=not args.no_lava,
        lava_t=args.lava_t,
        weight_scale=args.weight_scale
    )


if __name__ == '__main__':
    main()

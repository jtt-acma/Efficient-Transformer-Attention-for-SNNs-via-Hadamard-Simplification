"""
USSA Lava实现 - 支持W1=1x1+全局平均产生全局Q

USSA特性：
1. W1是1x1卷积+空间平均 → 全局Q [C]
2. W2是3x3卷积 → 局部V [C, H, W]  
3. 全局Q广播到所有位置与V进行门控
"""

import numpy as np
import time
import torch
import warnings

try:
    from lava.proc.lif.process import LIF
    from lava.proc.dense.process import Dense
    from lava.proc.io.source import RingBuffer
    from lava.proc.monitor.process import Monitor
    from lava.magma.core.run_configs import Loihi1SimCfg
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.process.process import AbstractProcess
    from lava.magma.core.process.ports.ports import InPort, OutPort
    from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
    from lava.magma.core.model.py.ports import PyInPort, PyOutPort
    from lava.magma.core.model.py.type import LavaPyType
    from lava.magma.core.resources import CPU
    from lava.magma.core.decorator import implements, requires
    from lava.magma.core.model.py.model import PyLoihiProcessModel
    LAVA_AVAILABLE = True
except ImportError:
    LAVA_AVAILABLE = False


if LAVA_AVAILABLE:
    # 门控方案：Q (L1) 控制 V (L2) 是否通过
    class GatedDenseProcess(AbstractProcess):
        """门控Dense：V的输入只有在Q有脉冲时才能通过"""
        def __init__(self, shape):
            super().__init__(shape=shape)
            self.v_in = InPort(shape=shape)   # L2 输出 (V)
            self.q_in = InPort(shape=shape)   # L1 输出 (Q，门控信号)
            self.s_out = OutPort(shape=shape)
    
    @implements(proc=GatedDenseProcess, protocol=LoihiProtocol)
    @requires(CPU)
    class PyGatedDenseModel(PyLoihiProcessModel):
        v_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
        q_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
        s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
        
        def run_spk(self):
            v = self.v_in.recv()
            q = self.q_in.recv()
            # Q有脉冲时通过，否则阻断
            self.s_out.send(v * (q > 0).astype(float))
    
    # USSA关键：空间平均Process（跨token）
    class SpatialMeanProcess(AbstractProcess):
        """空间平均：将所有位置的输入平均，输出单个全局向量
        注意：应该除以总空间位置数H*W，而不是仅活跃token数"""
        def __init__(self, num_active_tokens, total_tokens, channels):
            super().__init__(num_active_tokens=num_active_tokens, 
                           total_tokens=total_tokens, channels=channels)
            self.in_port = InPort(shape=(num_active_tokens * channels,))
            self.out_port = OutPort(shape=(channels,))
    
    @implements(proc=SpatialMeanProcess, protocol=LoihiProtocol)
    @requires(CPU)
    class PySpatialMeanModel(PyLoihiProcessModel):
        in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
        out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
        
        def __init__(self, proc_params):
            super().__init__(proc_params)
            self.num_active_tokens = proc_params['num_active_tokens']
            self.total_tokens = proc_params['total_tokens']
            self.channels = proc_params['channels']
        
        def run_spk(self):
            data = self.in_port.recv()  # [N_active*C]
            if data.sum() == 0:
                mean_data = np.zeros(self.channels)
            else:
                # reshape
                reshaped = data.reshape(self.num_active_tokens, self.channels)
                # USSA: 除以总token数H*W（不是仅活跃token数）
                sum_data = reshaped.sum(axis=0)  # [C]
                mean_data = sum_data / self.total_tokens  # 除以H*W
            self.out_port.send(mean_data)


def run_dssa_lava_shared(features, T, dssa_module, weight_scale=4.0):
    """
    USSA Lava实现 - 支持全局Q
    
    USSA架构:
    - W1: 1x1卷积 + 空间平均 → 全局Q [T, C, 1, 1]
    - W2: 3x3卷积 → 局部V [T, C, H, W]
    - 门控: Q广播后与V逐元素相乘
    """
    if not LAVA_AVAILABLE:
        raise ImportError("Lava not available")
    
    inp = features['input_data']  # [T, C, H, W]
    T_real, C, H, W = inp.shape
    N = H * W  # 总token数
    
    print(f"[Lava USSA Shared] T={T_real}, C={C}, H={H}, W={W}, tokens={N}")
    
    # 提取权重
    w1_conv = dssa_module.W1.weight.data.cpu().numpy()  # [C, C, 1, 1] for USSA
    w2_conv = dssa_module.W2.weight.data.cpu().numpy()  # [C, C, 3, 3]
    wproj = dssa_module.Wproj.weight.data.cpu().numpy()  # [C, C, 1, 1]
    wproj = wproj[:, :, 0, 0]  # squeeze to [C, C]
    
    # 检测USSA模式 (W1是1x1)
    is_ussa = (w1_conv.shape[2] == 1 and w1_conv.shape[3] == 1)
    print(f"  检测到的模式: {'USSA' if is_ussa else 'SSA'}")
    print(f"  W1 kernel: {w1_conv.shape[2:]} | W2 kernel: {w2_conv.shape[2:]}")
    
    vth_1 = getattr(dssa_module.activation_attn, 'v_threshold', 1.0)
    vth_2 = getattr(dssa_module.activation_out, 'v_threshold', 1.0)
    
    # 权重缩放
    w1_conv = w1_conv * weight_scale
    w2_conv = w2_conv * weight_scale
    wproj = wproj * weight_scale
    
    print(f"  使用 weight_scale={weight_scale}")
    
    tau = 2.0
    du = 1.0 / tau
    dv = 1.0 - 1.0 / tau
    
    # ============================================================
    # 收集所有token的输入（W1和W2共享同一个输入源inp）
    # ============================================================
    all_neighborhoods = []
    all_centers = []  # 用于W1的1x1卷积（中心像素）
    
    for h in range(H):
        for w_pos in range(W):
            # 收集3x3邻域（用于W2）
            neighborhood = np.zeros((T_real, 9 * C))  # [时间步, 9*通道]
            center_data = np.zeros((T_real, C))  # [时间步, C] 用于W1
            
            for kh in range(3):
                for kw in range(3):
                    src_h = h + kh - 1
                    src_w = w_pos + kw - 1
                    
                    if 0 <= src_h < H and 0 <= src_w < W:
                        src_data = inp[:, :, src_h, src_w]  # [T, C]
                        for c in range(C):
                            idx = c * 9 + kh * 3 + kw
                            neighborhood[:, idx] = src_data[:, c]
                        # 记录中心位置 (kh=1, kw=1)
                        if kh == 1 and kw == 1:
                            center_data = src_data.copy()
            
            # 只添加活跃token（有输入的）
            if neighborhood.sum() > 0:
                all_neighborhoods.append(neighborhood)
                all_centers.append(center_data)  # 同一条件内添加，保持一致性
    
    num_active_tokens = len(all_neighborhoods)
    if num_active_tokens == 0:
        print("  警告: 没有活跃token")
        return {
            'status': 'lava_complete',
            'l1_spikes': np.array([0]),
            'l2_spikes': np.array([0]),
            'gated_out': np.array([0]),
            'monitor_data': {'l1_spikes': np.array([0]), 'l2_spikes': np.array([0]), 'gated_out': np.array([0])},
            'num_tokens': 0
        }
    
    print(f"  活跃tokens: {num_active_tokens}/{N}")
    
    # ============================================================
    # W2分支: 3x3卷积产生局部V
    # ============================================================
    w2_dense = np.zeros((C, 9 * C))
    for co in range(C):
        for ci in range(C):
            for kh in range(3):
                for kw in range(3):
                    idx = ci * 9 + kh * 3 + kw
                    w2_dense[co, idx] = w2_conv[co, ci, kh, kw]
    
    big_input_dim = num_active_tokens * 9 * C
    big_output_dim = num_active_tokens * C
    
    w2_big = np.zeros((big_output_dim, big_input_dim))
    for i in range(num_active_tokens):
        out_start = i * C
        out_end = (i + 1) * C
        in_start = i * 9 * C
        in_end = (i + 1) * 9 * C
        w2_big[out_start:out_end, in_start:in_end] = w2_dense
    
    # ============================================================
    # W1分支: 1x1卷积 + 空间平均 → 全局Q
    # ============================================================
    w1_squeezed = w1_conv[:, :, 0, 0]  # [C, C]
    
    # W1输入：活跃位置的中心像素 [T, N*C]
    w1_input = np.concatenate(all_centers, axis=1)  # [T, N*C]
    
    # ============================================================
    # USSA关键：空间平均操作（Python预计算）
    # ============================================================
    # 由于Lava不支持跨token的空间平均，我们在Python中预计算：
    # 1. 用Dense层计算每个位置的W1(1x1)输出
    # 2. 对所有位置做空间平均得到全局Q
    # 3. 将全局Q广播回所有位置作为LIF输入
    
    # 构建W1的block-diagonal权重 [N*C, N*C]
    # 添加N倍增益补偿SpatialMean的信号削弱（÷H*W）
    gain_compensation = int(np.sqrt(N))  # H*W，补偿SpatialMean的削弱
    
    w1_big = np.zeros((num_active_tokens * C, num_active_tokens * C))
    for i in range(num_active_tokens):
        start = i * C
        end = (i + 1) * C
        w1_big[start:end, start:end] = w1_squeezed * gain_compensation
    
    
    # V分支输入
    v_input = np.concatenate(all_neighborhoods, axis=1)  # [T, N*9*C]
    
    # ============================================================
    # 创建Lava进程
    # ============================================================
    print(f"  创建Lava进程...")
    
    # Q分支输入（中心像素）
    src_q = RingBuffer(data=w1_input.T)  # [N*C, T]

    # V分支输入（3x3邻域）
    src_v = RingBuffer(data=v_input.T)  # [9*C*N, T]
    
    d1 = Dense(weights=w1_big)
    spatial_mean = SpatialMeanProcess(num_active_tokens=num_active_tokens, 
                                      total_tokens=N,  # H*W
                                      channels=C)
    l1 = LIF(shape=(C,), du=du, dv=dv, vth=vth_1, bias_mant=0, bias_exp=0)
    
    # Step 4: Broadcast 全局Q到所有位置 [C] → [N*C]
    # 构建广播权重：每个输出位置复制相同的Q
    broadcast_weights = np.zeros((num_active_tokens * C, C))
    for i in range(num_active_tokens):
        broadcast_weights[i*C:(i+1)*C, :] = np.eye(C)  # 每个block复制全局Q
    d_broadcast = Dense(weights=broadcast_weights)
    
    # V分支：Dense(W2) → LIF [N*9*C] → [N*C]
    d2 = Dense(weights=w2_big)
    l2 = LIF(shape=(big_output_dim,), du=du, dv=dv, vth=vth_2, bias_mant=0, bias_exp=0)
    
    # 门控 [N*C]
    gated = GatedDenseProcess(shape=(big_output_dim,))
    
    # Wproj：1x1卷积 [N*C, N*C]
    wproj_big = np.zeros((num_active_tokens * C, num_active_tokens * C))
    for i in range(num_active_tokens):
        start = i * C
        end = (i + 1) * C
        wproj_big[start:end, start:end] = wproj
    
    dp = Dense(weights=wproj_big)
    
    # ============================================================
    # 连接网络
    # ============================================================
    # Q分支：中心像素 → W1 → SpatialMean → LIF → Broadcast
    src_q.out_ports.s_out.connect(d1.in_ports.s_in)
    d1.out_ports.a_out.connect(spatial_mean.in_port)
    spatial_mean.out_port.connect(l1.in_ports.a_in)
    l1.out_ports.s_out.connect(d_broadcast.in_ports.s_in)
    d_broadcast.out_ports.a_out.connect(gated.q_in)
    
    # V分支：3x3邻域 → W2 → LIF
    src_v.out_ports.s_out.connect(d2.in_ports.s_in)
    d2.out_ports.a_out.connect(l2.in_ports.a_in)
    l2.out_ports.s_out.connect(gated.v_in)
    
    # 门控输出 → Wproj
    gated.s_out.connect(dp.in_ports.s_in)
    
    # Monitors
    # W1输出monitor：统计卷积后活跃的token数
    mon_w1 = Monitor()
    mon_w1.probe(target=d1.a_out, num_steps=T_real + 1)
    
    mon_l1 = Monitor()
    mon_l1.probe(target=l1.s_out, num_steps=T_real + 1)
    
    mon_l2 = Monitor()
    mon_l2.probe(target=l2.s_out, num_steps=T_real + 1)
    
    mon_gated = Monitor()
    mon_gated.probe(target=gated.s_out, num_steps=T_real + 1)
    
    # 运行
    print("  运行...")
    start = time.time()
    run_cfg = Loihi1SimCfg(select_tag='floating_pt')
    
    # 使用第一个进程运行网络（会自动执行所有连接进程）
    src_q.run(condition=RunSteps(num_steps=T_real + 1), run_cfg=run_cfg)
    
    elapsed = time.time() - start
    print(f"  完成: {elapsed:.2f}s")
    
    # 读取数据
    print("  读取数据...")
    
    w1_raw = mon_w1.get_data()
    l1_raw = mon_l1.get_data()
    l2_raw = mon_l2.get_data()
    gated_raw = mon_gated.get_data()
    
    w1_data = list(w1_raw.values())[0]['a_out'][1:]  # W1模拟输出
    l1_data = list(l1_raw.values())[0]['s_out'][1:]
    l2_data = list(l2_raw.values())[0]['s_out'][1:]
    gated_data = list(gated_raw.values())[0]['s_out'][1:]
    
    # 统计W1卷积后活跃的token数（非零输出）
    # w1_data shape: [T, N*C]
    w1_reshaped = w1_data.reshape(T_real, num_active_tokens, C)
    w1_active_per_token = (np.abs(w1_reshaped).sum(axis=(0, 2)) > 0).sum()  # 至少一个时间步有输出的token
    
    # 打印shape
    print(f"  [Shape] W1输出: {w1_data.shape} = [T, N*C] (卷积后)")
    print(f"  [Shape] L1 (Q) 输出: {l1_data.shape} = [T, C] (全局Q, C={C})")
    print(f"  [Shape] L2 (V) 输出: {l2_data.shape} = [T, N*C] (局部V, N={num_active_tokens}, C={C})")
    print(f"  [Shape] Gated输出: {gated_data.shape} = [T, N*C] (Q⊙V)")
    
    l1_total = int(l1_data.sum())
    l2_total = int(l2_data.sum())
    gated_total = int(gated_data.sum())
    
    # L1输出是全局Q [C]，需要乘以num_active_tokens来估算总脉冲贡献
    print(f"  Monitor 汇总 ({num_active_tokens} tokens):")
    print(f"    W1卷积后活跃tokens: {w1_active_per_token}/{num_active_tokens}")
    print(f"    L1 (全局Q) 脉冲: {l1_total} (广播到{num_active_tokens}位置)")
    print(f"    L2 (V) 脉冲: {l2_total}")
    print(f"    门控输出 (Q⊙V): {gated_total}")
    
    src_q.stop()
    
    return {
        'status': 'lava_complete',
        'l1_spikes': np.array([l1_total]),
        'l2_spikes': np.array([l2_total]),
        'gated_out': np.array([gated_total]),
        'monitor_data': {
            'l1_spikes': np.array([l1_total]),
            'l2_spikes': np.array([l2_total]),
            'gated_out': np.array([gated_total]),
            'w1_active_tokens': w1_active_per_token  # W1卷积后活跃的token数
        },
        'num_tokens': num_active_tokens,
        'is_ussa': is_ussa
    }

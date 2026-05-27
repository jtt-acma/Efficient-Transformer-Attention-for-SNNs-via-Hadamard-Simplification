"""
最优DSSA Lava实现 - 共享权重 + Token并行

核心优化：
1. 权重共享：使用block-diagonal矩阵实现逻辑共享
2. Token并行：所有token一次性处理
3. LIF独立：每个token保持独立LIF（可接受近似）
4. 门控改进：Hadamard乘积 (a * b)
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


def run_dssa_lava_shared(features, T, dssa_module, weight_scale=4.0):
    """
    优化的DSSA Lava实现 - 共享权重 + Token并行
    """
    if not LAVA_AVAILABLE:
        raise ImportError("Lava not available")
    
    inp = features['input_data']  # [T, C, H, W]
    T_real, C, H, W = inp.shape
    N = H * W  # 总token数
    
    print(f"[Lava DSSA Shared] T={T_real}, C={C}, H={H}, W={W}, tokens={N}")
    
    # 提取权重
    w1_conv = dssa_module.W1.weight.data.cpu().numpy()  # [C, C, 3, 3]
    w2_conv = dssa_module.W2.weight.data.cpu().numpy()
    wproj = dssa_module.Wproj.weight.data.cpu().numpy()  # [C, C, 1, 1]
    wproj = wproj[:, :, 0, 0]  # squeeze to [C, C]
    
    vth_1 = getattr(dssa_module.activation_attn, 'v_threshold', 1.0)
    vth_2 = getattr(dssa_module.activation_out, 'v_threshold', 1.0)
    
    # ============================================================
    # 为什么需要 weight_scale？
    # ============================================================
    # 
    # 1. 原始权重太小：
    #    模型训练得到的权重 std≈0.02，单次脉冲产生的电流≈0.02
    #    远小于 LIF 阈值 vth=1.0，无法激活神经元
    #
    # 2. Lava LIF 的双层衰减：
    #    Lava:  u[t] = u[t-1]*(1-du) + input  (电流衰减)
    #          v[t] = v[t-1]*(1-dv) + u[t]    (电压衰减)
    #    PyTorch: v[t] = v[t-1]*(1-1/tau) + input  (单层)
    #    
    #    Lava 有两次衰减，同样的输入在 Lava 中有效增益更低
    # ============================================================
    w1_conv = w1_conv * weight_scale
    w2_conv = w2_conv * weight_scale
    wproj = wproj * weight_scale
    
    print(f"  使用 weight_scale={weight_scale} (补偿Lava双层衰减)")
    print(f"  w1 range: [{w1_conv.min():.2f}, {w1_conv.max():.2f}]")
    
    tau = 2.0  # 与 PyTorch LIF 一致
    du = 1.0 / tau
    dv = 1.0 - 1.0 / tau
    
    # 收集所有token的输入
    print(f"  收集所有token输入...")
    
    all_neighborhoods = []
    
    # === 收集每个空间位置的3x3邻域输入 ===
    # 对于卷积操作，每个输出位置(h,w)需要其3x3邻域的输入
    # 
    # 示例：位置(1,1)的邻域包含位置(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)
    #       共9个位置，每个位置有C个通道
    #       展开后为一维向量，长度 = 9 * C
    
    for h in range(H):           # 遍历所有行
        for w_pos in range(W):   # 遍历所有列
            neighborhood = np.zeros((T_real, 9 * C))  # [时间步, 9*通道]
            
            # 收集3x3邻域（考虑padding）
            for kh in range(3):   # 卷积核行: 0,1,2
                for kw in range(3):  # 卷积核列: 0,1,2
                    # 计算源位置（中心对齐）
                    # kh-1, kw-1 产生偏移: -1,0,+1
                    # 例如：当前位置1 + (1-1) = 2，即下方邻居
                    src_h = h + kh - 1
                    src_w = w_pos + kw - 1
                    
                    # 边界检查（超出边界的位置用0填充，相当于padding=1）
                    if 0 <= src_h < H and 0 <= src_w < W:
                        # 获取该邻域位置的输入 [T, C]
                        src_data = inp[:, :, src_h, src_w]
                        
                        # 将数据放入展开数组的对应位置
                        # 展开顺序：先通道，再卷积核位置
                        # idx = c*9 + kh*3 + kw
                        # 例如：c=0,kh=0,kw=0 -> idx=0 (第0通道,左上角)
                        #      c=0,kh=1,kw=1 -> idx=4 (第0通道,中心)
                        #      c=1,kh=0,kw=0 -> idx=9 (第1通道,左上角)
                        for c in range(C):
                            idx = c * 9 + kh * 3 + kw
                            neighborhood[:, idx] = src_data[:, c]
            
            # 优化：跳过完全没有输入的token（节省计算）
            # 如果3x3邻域内没有任何脉冲，卷积输出必为0，无需仿真
            if neighborhood.sum() > 0:
                all_neighborhoods.append(neighborhood)
    
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
    
    # 构建基础dense权重 [C, 9*C]
    w1_dense = np.zeros((C, 9 * C))
    w2_dense = np.zeros((C, 9 * C))
    
    for co in range(C):
        for ci in range(C):
            for kh in range(3):
                for kw in range(3):
                    idx = ci * 9 + kh * 3 + kw
                    w1_dense[co, idx] = w1_conv[co, ci, kh, kw]
                    w2_dense[co, idx] = w2_conv[co, ci, kh, kw]
    
    # 构建block-diagonal矩阵
    big_input_dim = num_active_tokens * 9 * C
    big_output_dim = num_active_tokens * C
    
    w1_big = np.zeros((big_output_dim, big_input_dim))
    w2_big = np.zeros((big_output_dim, big_input_dim))
    
    for i in range(num_active_tokens):
        out_start = i * C
        out_end = (i + 1) * C
        in_start = i * 9 * C
        in_end = (i + 1) * 9 * C
        
        w1_big[out_start:out_end, in_start:in_end] = w1_dense
        w2_big[out_start:out_end, in_start:in_end] = w2_dense
    
    print(f"  Block-diagonal权重: [{big_output_dim}, {big_input_dim}]")
    
    # 拼接所有token输入
    big_input = np.concatenate(all_neighborhoods, axis=1)
    
    # 创建进程
    print(f"  创建共享Dense + 独立LIF...")
    
    src = RingBuffer(data=big_input.T)
    
    # W1分支
    d1 = Dense(weights=w1_big)
    l1 = LIF(shape=(big_output_dim,), du=du, dv=dv, vth=vth_1, bias_mant=0, bias_exp=0)
    
    # W2分支
    d2 = Dense(weights=w2_big)
    l2 = LIF(shape=(big_output_dim,), du=du, dv=dv, vth=vth_2, bias_mant=0, bias_exp=0)
    
    # Hadamard乘积
    gated = GatedDenseProcess(shape=(big_output_dim,))
    
    # Wproj
    wproj_big = np.zeros((num_active_tokens * C, num_active_tokens * C))
    for i in range(num_active_tokens):
        start = i * C
        end = (i + 1) * C
        wproj_big[start:end, start:end] = wproj
    
    dp = Dense(weights=wproj_big)
    
    # 连接
    src.out_ports.s_out.connect(d1.in_ports.s_in)
    d1.out_ports.a_out.connect(l1.in_ports.a_in)
    src.out_ports.s_out.connect(d2.in_ports.s_in)
    d2.out_ports.a_out.connect(l2.in_ports.a_in)
    
    # 门控连接：V是数据，Q是门控
    l2.out_ports.s_out.connect(gated.v_in)
    l1.out_ports.s_out.connect(gated.q_in)
    gated.s_out.connect(dp.in_ports.s_in)
    
    # Monitors
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
    
    src.run(condition=RunSteps(num_steps=T_real + 1), run_cfg=run_cfg)
    
    elapsed = time.time() - start
    print(f"  完成: {elapsed:.2f}s")
    
    # 读取数据
    print("  读取数据...")
    
    l1_raw = mon_l1.get_data()
    l2_raw = mon_l2.get_data()
    gated_raw = mon_gated.get_data()
    
    l1_data = list(l1_raw.values())[0]['s_out'][1:]
    l2_data = list(l2_raw.values())[0]['s_out'][1:]
    gated_data = list(gated_raw.values())[0]['s_out'][1:]
    
    l1_total = int(l1_data.sum())
    l2_total = int(l2_data.sum())
    gated_total = int(gated_data.sum())
    
    print(f"  Monitor 汇总 ({num_active_tokens} tokens):")
    print(f"    L1 (Q) 脉冲: {l1_total}")
    print(f"    L2 (V) 脉冲: {l2_total}")
    print(f"    门控输出 (Q⊙V): {gated_total}")
    
    src.stop()
    
    return {
        'status': 'lava_complete',
        'l1_spikes': np.array([l1_total]),
        'l2_spikes': np.array([l2_total]),
        'gated_out': np.array([gated_total]),
        'monitor_data': {
            'l1_spikes': np.array([l1_total]),
            'l2_spikes': np.array([l2_total]),
            'gated_out': np.array([gated_total])
        },
        'num_tokens': num_active_tokens
    }

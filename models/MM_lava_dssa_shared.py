"""
MM (Multi-Modal) Lava实现 - PyTorch+Lava混合方案

标准MM架构（2个LIF）:
  x ─→ W1/Conv(Lava) ─→ y1 (浮点) ─┐
                                   ├──→ matmul1(PyTorch) ─→ LIF1(PyTorch) ─→ attn
  x ─→ W2/Conv(Lava) ─→ y2 (浮点) ─┘                          ↓
                                                              matmul2(PyTorch) ─→ LIF2(Lava) ─→ Wproj(Lava)

只有2个LIF:
- LIF1 (activation_attn): matmul1之后 - PyTorch (16M神经元，必须用PyTorch)
- LIF2 (activation_out): matmul2之后 - Lava (524K神经元，可用Lava)

W1/W2/Wproj: Lava Dense层
LIF2: Lava LIF (与SSA一致)
"""

import numpy as np
import time
import torch

try:
    from lava.proc.dense.process import Dense
    from lava.proc.lif.process import LIF
    from lava.proc.io.source import RingBuffer
    from lava.proc.monitor.process import Monitor
    from lava.magma.core.run_configs import Loihi1SimCfg
    from lava.magma.core.run_conditions import RunSteps
    LAVA_AVAILABLE = True
except ImportError:
    LAVA_AVAILABLE = False


def pytorch_lif_step(current, v_prev, vth=1.0, tau=2.0):
    """PyTorch LIF神经元"""
    du = 1.0 / tau
    dv = 1.0 - 1.0 / tau
    v_new = dv * v_prev + du * current
    spike = (v_new >= vth).float()
    v_new = v_new * (1.0 - spike)
    return spike, v_new


def run_dssa_lava_shared(features, T, dssa_module, tile_size=64, weight_scale=4.0):
    """
    MM架构: W1/Conv → matmul1 → LIF1 → matmul2 → LIF2 → Wproj
    
    只有2个LIF，都在matmul之后
    
    注意: 使用 weight_scale 保持与 Lava 一致的发放率
    """
    if not LAVA_AVAILABLE:
        raise ImportError("Lava not available")
    
    inp = features['input_data']  # [T, C, H, W]
    T_real, C, H, W = inp.shape
    N = H * W
    num_heads = getattr(dssa_module, 'num_heads', 1)
    D = C // num_heads
    
    print(f"[Lava MM DSSA - 标准2-LIF架构]")
    print(f"  T={T_real}, C={C}, H={H}, W={W}, N={N}")
    print(f"  num_heads={num_heads}, D={D}")
    
    # 权重和阈值
    # W1/W2 使用 weight_scale 保持与 Lava 一致的发放率
    w1_conv = dssa_module.W1.weight.data.cpu().numpy() * weight_scale
    w2_conv = dssa_module.W2.weight.data.cpu().numpy() * weight_scale
    wproj = dssa_module.Wproj.weight.data.cpu().numpy()[:, :, 0, 0] * weight_scale
    
    # 阈值保持原始值
    vth_1 = getattr(dssa_module.activation_attn, 'v_threshold', 1.0)
    vth_2 = getattr(dssa_module.activation_out, 'v_threshold', 1.0)
    
    tau = 2.0
    
    # 准备数据
    print("\n  准备数据...")
    neighborhoods = np.zeros((T_real, N, 9 * C))
    token_idx = 0
    for h in range(H):
        for w_pos in range(W):
            for kh in range(3):
                for kw in range(3):
                    src_h, src_w = h + kh - 1, w_pos + kw - 1
                    if 0 <= src_h < H and 0 <= src_w < W:
                        for c in range(C):
                            neighborhoods[:, token_idx, c * 9 + kh * 3 + kw] = inp[:, c, src_h, src_w]
            token_idx += 1
    
    # x (中心像素) [T, N, C]
    x_center = neighborhoods[:, :, 4*C:5*C]
    
    # W1/W2 Dense权重 [C, 9*C]
    w1_dense = np.zeros((C, 9 * C))
    w2_dense = np.zeros((C, 9 * C))
    for co in range(C):
        for ci in range(C):
            for kh in range(3):
                for kw in range(3):
                    idx = ci * 9 + kh * 3 + kw
                    w1_dense[co, idx] = w1_conv[co, ci, kh, kw]
                    w2_dense[co, idx] = w2_conv[co, ci, kh, kw]
    
    start = time.time()
    
    num_tiles = (N + tile_size - 1) // tile_size
    
    all_y1 = []
    all_y2 = []
    
    for tile_id in range(num_tiles):
        start_idx = tile_id * tile_size
        end_idx = min((tile_id + 1) * tile_size, N)
        S = end_idx - start_idx
        
        tile_input = neighborhoods[:, start_idx:end_idx, :].reshape(T_real, S * 9 * C)
        
        # Block-diagonal权重
        w1_tile = np.zeros((S * C, S * 9 * C))
        w2_tile = np.zeros((S * C, S * 9 * C))
        for i in range(S):
            w1_tile[i*C:(i+1)*C, i*9*C:(i+1)*9*C] = w1_dense
            w2_tile[i*C:(i+1)*C, i*9*C:(i+1)*9*C] = w2_dense
        
        src = RingBuffer(data=tile_input.T)
        d1 = Dense(weights=w1_tile)
        d2 = Dense(weights=w2_tile)
        
        # 直接监控Dense输出（浮点电流，不是脉冲）
        mon_d1 = Monitor()
        mon_d1.probe(target=d1.out_ports.a_out, num_steps=T_real + 1)
        mon_d2 = Monitor()
        mon_d2.probe(target=d2.out_ports.a_out, num_steps=T_real + 1)
        
        src.out_ports.s_out.connect(d1.in_ports.s_in)
        src.out_ports.s_out.connect(d2.in_ports.s_in)
        
        run_cfg = Loihi1SimCfg(select_tag='floating_pt')
        src.run(condition=RunSteps(num_steps=T_real + 1), run_cfg=run_cfg)
        
        # 获取浮点输出
        y1_data = list(mon_d1.get_data().values())[0]['a_out'][1:]  # [T, S*C]
        y2_data = list(mon_d2.get_data().values())[0]['a_out'][1:]
        
        all_y1.append(y1_data)
        all_y2.append(y2_data)
        
        src.stop()
        
        if (tile_id + 1) % 8 == 0 or tile_id == num_tiles - 1:
            print(f"    Tile {tile_id+1}/{num_tiles}")
    
    y1_all = np.concatenate(all_y1, axis=1)  # [T, N*C] - 浮点
    y2_all = np.concatenate(all_y2, axis=1)  # [T, N*C] - 浮点
    
    print(f"    W1/W2 输出: {y1_all.shape}, range=[{y1_all.min():.2f}, {y1_all.max():.2f}]")
    print(f"    Time: {time.time()-start:.2f}s")
    
    # ============================================================
    # Phase 2: matmul1 (PyTorch)
    # ============================================================
    print(f"\n  Phase 2: matmul1 (PyTorch)...")
    start = time.time()
    
    y1_reshape = y1_all.reshape(T_real, N, num_heads, D)
    x_reshape = x_center.reshape(T_real, N, num_heads, D)
    
    y1_torch = torch.from_numpy(y1_reshape).float()
    x_torch = torch.from_numpy(x_reshape).float()
    
    print(f"    y1_torch: {y1_torch.shape}, x_torch: {x_torch.shape}")
    
    # matmul1 缩放因子: 1/√(firing_rate_x * D)
    # 基于实际输入发放率动态调整，类似标准 Transformer 的 scaled dot-product attention
    firing_rate_x = x_torch.abs().mean().item()  # [0, 1] 范围
    scale1 = 1.0 / np.sqrt(max(firing_rate_x * D, 0.001))  # 避免除零
    print(f"    firing_rate_x: {firing_rate_x:.4f}")
    print(f"    matmul1 缩放因子: 1/√(firing_rate_x * D) = 1/√({firing_rate_x:.4f} * {D}) = {scale1:.4f}")
    
    #如果用 Lava	16M LIF 神经元 × T=16 时间步 = 需要 数百 GB内存，完全不可行,一次这里才使用PyTorch进行 matmul1 和 LIF1 的计算
    attn_pre = torch.matmul(y1_torch.permute(0, 2, 1, 3), x_torch.permute(0, 2, 1, 3).transpose(-1, -2)) * scale1
    attn_pre = attn_pre.permute(0, 2, 1, 3)
    
    print(f"    attn_pre: {attn_pre.shape}")
    print(f"    Time: {time.time()-start:.3f}s")
    
    # ============================================================
    # Phase 3: LIF1 (PyTorch) 
    # ============================================================
    print(f"\n  Phase 3: LIF1 (PyTorch) - 标准MM唯一LIF1...")
    start = time.time()
    
    v_attn = torch.zeros_like(attn_pre)
    attn_spikes = []
    
    for t in range(T_real):
        spike, v_attn[t] = pytorch_lif_step(attn_pre[t:t+1], v_attn[t:t+1] if t == 0 else v_attn[t-1:t], vth=vth_1, tau=tau)
        attn_spikes.append(spike.squeeze(0))
    
    attn_spike = torch.stack(attn_spikes)
    attn_total = int(attn_spike.sum().item())
    
    print(f"    LIF1 输出脉冲: {attn_total:,}")
    print(f"    Time: {time.time()-start:.3f}s")
    
    # ============================================================
    # Phase 4: matmul2 (PyTorch)
    # ============================================================
    print(f"\n  Phase 4: matmul2 (PyTorch)...")
    start = time.time()
    
    y2_torch = torch.from_numpy(y2_all.reshape(T_real, N, num_heads, D)).float()
    
    # matmul2 缩放因子: 1/√(firing_rate_attn * N)
    # 基于 LIF1 实际输出发放率动态调整
    firing_rate_attn = attn_spike.abs().mean().item()  # [0, 1] 范围
    scale2 = 1.0 / np.sqrt(max(firing_rate_attn * N, 0.0001))  # 避免除零
    print(f"    firing_rate_attn: {firing_rate_attn:.6f}")
    print(f"    matmul2 缩放因子: 1/√(firing_rate_attn * N) = 1/√({firing_rate_attn:.6f} * {N}) = {scale2:.4f}")
    
    out_pre = torch.matmul(attn_spike.permute(0, 2, 1, 3), y2_torch.permute(0, 2, 1, 3)) * scale2
    out_pre = out_pre.permute(0, 2, 1, 3)
    
    print(f"    out_pre: {out_pre.shape}")
    print(f"    Time: {time.time()-start:.3f}s")
    
    # ============================================================
    # Phase 5: LIF2 (Lava) - 第二个也是最后一个LIF
    # ============================================================
    print(f"\n  Phase 5: LIF2 (Lava) - 标准MM唯一LIF2...")
    print(f"    LIF2 规模: {N * num_heads * D:,} 神经元 (N={N} × heads={num_heads} × D={D})")
    start = time.time()
    
    # 准备输入数据: [T, N, num_heads, D] -> [T, N*num_heads*D]
    out_pre_np = out_pre.detach().cpu().numpy().reshape(T_real, N * num_heads * D)
    
    # 创建 Lava 进程
    src_lif2 = RingBuffer(data=out_pre_np.T)  # [N*num_heads*D, T]
    
    # LIF2 参数 (与 PyTorch 版本一致)
    du_lif2 = 1.0 / tau
    dv_lif2 = 1.0 - 1.0 / tau
    
    # 创建 LIF 进程 - 规模 524K 神经元，完全可行
    lif2 = LIF(shape=(N * num_heads * D,), du=du_lif2, dv=dv_lif2, 
               vth=vth_2, bias_mant=0, bias_exp=0)
    
    # 连接: RingBuffer -> LIF (电流输入)
    src_lif2.out_ports.s_out.connect(lif2.in_ports.a_in)
    
    # Monitor 记录输出
    mon_lif2 = Monitor()
    mon_lif2.probe(target=lif2.s_out, num_steps=T_real + 1)
    
    # 运行仿真
    run_cfg = Loihi1SimCfg(select_tag='floating_pt')
    src_lif2.run(condition=RunSteps(num_steps=T_real + 1), run_cfg=run_cfg)
    
    # 获取数据
    lif2_raw = mon_lif2.get_data()
    lif2_data = list(lif2_raw.values())[0]['s_out'][1:]  # [T, N*num_heads*D]
    out_total = int(lif2_data.sum())
    
    # 转换回 PyTorch 格式用于后续 Wproj
    out_spike = torch.from_numpy(lif2_data).float().reshape(T_real, N, num_heads, D)
    
    src_lif2.stop()
    
    print(f"    LIF2 输出脉冲: {out_total:,}")
    print(f"    Time: {time.time()-start:.3f}s")
    
    # ============================================================
    # Phase 6: Wproj (Lava)
    # ============================================================
    print(f"\n  Phase 6: Wproj (Lava)...")
    start = time.time()
    
    out_pre_np = out_spike.numpy().reshape(T_real, N * C)
    
    for tile_id in range(num_tiles):
        start_idx = tile_id * tile_size
        end_idx = min((tile_id + 1) * tile_size, N)
        S = end_idx - start_idx
        
        tile_out = out_pre_np[:, start_idx*C:end_idx*C]
        src_out = RingBuffer(data=tile_out.T)
        
        wproj_tile = np.zeros((S * C, S * C))
        for i in range(S):
            wproj_tile[i*C:(i+1)*C, i*C:(i+1)*C] = wproj
        
        dp = Dense(weights=wproj_tile)
        src_out.out_ports.s_out.connect(dp.in_ports.s_in)
        
        run_cfg = Loihi1SimCfg(select_tag='floating_pt')
        src_out.run(condition=RunSteps(num_steps=T_real + 1), run_cfg=run_cfg)
        src_out.stop()
        
        if (tile_id + 1) % 8 == 0 or tile_id == num_tiles - 1:
            print(f"    Tile {tile_id+1}/{num_tiles}")
    
    print(f"    Time: {time.time()-start:.2f}s")
    
    # ============================================================
    # 汇总
    # ============================================================
    # 计算输入脉冲数
    input_spikes_total = int(x_center.sum())  # W1/W2输入
    y1_nonzero = int((np.abs(y1_all) > 0).sum())  # matmul1输入 (W1输出非零)
    y2_nonzero = int((np.abs(y2_all) > 0).sum())  # matmul2输入y2
    attn_pre_nonzero = int((attn_pre != 0).sum())  # LIF1输入 (matmul1输出非零)
    
    print(f"\n" + "="*70)
    print(f"标准MM架构仿真完成 (2-LIF)")
    print(f"="*70)
    print(f"数据流: W1/Conv → matmul1 → LIF1 → matmul2 → LIF2 → Wproj")
    print(f"  W1/W2 输入脉冲:       {input_spikes_total:10,} spikes (x_center)")
    print(f"  LIF1 (matmul1后):     {attn_total:10,} spikes")
    print(f"  LIF2 (matmul2后):     {out_total:10,} spikes")
    print(f"\n总计:                   {attn_total + out_total:10,} spikes")
    
    return {
        'status': 'lava_complete',
        'l1_spikes': np.array([attn_total]),  # LIF1
        'l2_spikes': np.array([out_total]),   # LIF2
        'attn_spikes': np.array([attn_total]),
        'matmul2_output_spikes': np.array([out_total]),
        'monitor_data': {
            'l1_spikes': attn_spike.sum(axis=(1,2,3)).numpy(),
            'l2_spikes': out_spike.sum(axis=(1,2,3)).numpy(),
        },
        'num_tokens': N,
        'num_heads': num_heads,
        'num_tiles': num_tiles,
        'tile_size': tile_size,
        'architecture': 'mm_2_lif_standard',
    }

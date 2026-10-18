#!/usr/bin/env python3
"""reduce_scatter（归约散射） —— 单文件 NCCL 微基准：语义验证 + NVLink 实测流量 + 环形模型对账

运行（Pod 内，8 GPU 单节点）:
    torchrun --nproc_per_node=8 reduce_scatter.py

严格定义（W 个 rank，r 为本 rank 下标）:
    out_r = (Σ_{k=0}^{W-1} x_k) 的第 r 段——先求和再切段, 每 rank 只拿自己那 1/W 段。
"""
import os, re, subprocess
import torch
import torch.distributed as dist

MIB = 2 ** 20
ITERS = 5


def nvln():
    """每 GPU NVLink Tx（KiB，驱动加载以来累计值；前后差值即流量）"""
    out = subprocess.run(["nvidia-smi", "nvlink", "-gt", "d"],
                         capture_output=True, text=True).stdout
    g, tx = -1, {}
    for line in out.splitlines():
        m = re.match(r"GPU (\d+):", line)
        if m:
            g = int(m.group(1)); tx[g] = 0; continue
        m = re.search(r"Data Tx: (\d+) KiB", line)
        if m and g >= 0:
            tx[g] += int(m.group(1))
    return [tx[i] for i in range(len(tx))]


rank = int(os.environ["RANK"]); local = int(os.environ["LOCAL_RANK"])
W = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local)
dist.init_process_group("nccl")
dev = f"cuda:{local}"

# ---------- 1) 语义正确性（小张量，结果可人工核对） ----------

x = torch.cat([torch.full((4,), float(j + 1), device=dev) for j in range(W)])  # 人人相同: 段 j 全为 j+1
o = torch.empty(4, device=dev)
dist.reduce_scatter_tensor(o, x)
ok = torch.equal(o, torch.full((4,), float(W * (rank + 1)), device=dev))      # rank r 得 8*(r+1)
flag = torch.tensor([1.0 if ok else 0.0], device=dev)
dist.all_reduce(flag)
chk_msg = ("PASS: rank r 得到自己段的和 8x(r+1)" if flag.item() == W
           else "FAIL: 仅 %d/%d rank 通过" % (int(flag.item()), W))


# ---------- 2) 数据传输量模型 vs 实际流量（大张量 x ITERS 次） ----------

# 单位约定: 尺寸/流量/带宽一律十进制——GB = 1e9 B, GB/s = 1e9 B/s;
# NVLink 计数器原始读数为 KiB(2^10 B), 按 KiB*1024/1e9 换算为 GB。
S = 16 * 10**9                                 # 每 rank 输入(=8 段); 输出 = S/W = 2 GB
t = torch.randn(S // 4, device=dev)
o = torch.empty(S // 4 // W, device=dev)
op = lambda: dist.reduce_scatter_tensor(o, t)
model_gb = (W - 1) * (S / W) / 1e9              # 环形: (W-1) * 段
alg_bytes = S
s_desc = "每 rank 输入 16 GB, 输出 2 GB"



for _ in range(3):
    op()
torch.cuda.synchronize(); dist.barrier(); torch.cuda.synchronize()
b = nvln() if rank == 0 else None
e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
dist.barrier()
e0.record()
for _ in range(ITERS):
    op()
e1.record(); torch.cuda.synchronize()
ms = e0.elapsed_time(e1) / ITERS
dist.barrier(); torch.cuda.synchronize()
a = nvln() if rank == 0 else None

if rank == 0:
    d = [(y - x) * 1024 / 1e9 / ITERS for x, y in zip(b, a)]  # 每 GPU 每次 GB
    measured = sum(d) / W
    ratio = measured / model_gb
    print(f"=== reduce_scatter（归约散射） ===")
    print(f"[语义] {chk_msg}")
    print(f"[规模] {s_desc}; 迭代 {ITERS} 次")
    print(f"[耗时] 每次 {ms:.3f} ms | algbw {alg_bytes / (ms / 1e3) / 1e9:.0f} GB/s"
          f" | busbw(环形口径) {model_gb * 1e9 / (ms / 1e3) / 1e9:.0f} GB/s")
    print(f"[流量] NVLink 实测: 每 GPU 平均 {measured:.4f} GB/次 (min {min(d):.4f} / max {max(d):.4f})")
    print(f"[对账] 环形模型 {model_gb:.4f} GB/次/GPU -> 实测/模型 = {ratio:.3f}")
dist.barrier()
dist.destroy_process_group()

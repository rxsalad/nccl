#!/usr/bin/env python3
"""reduce（归约到一卡） —— 单文件 NCCL 微基准：语义验证 + NVLink 实测流量 + 环形模型对账

运行（Pod 内，8 GPU 单节点）:
    torchrun --nproc_per_node=8 reduce.py

严格定义（W 个 rank，r 为本 rank 下标）:
    out_dst[i] = Σ_{r=0}^{W-1} x_r[i]  (仅 dst rank 持有结果；其余 rank 输出无定义)
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

x = torch.arange(1, 5, device=dev) * (rank + 1)      # rank r 输入 = (r+1)*[1,2,3,4]
dist.reduce(x, dst=0)
ok = False                                           # 非 dst 无定义, 不检查
if rank == 0:
    exp = torch.arange(1, 5, device=dev) * (W * (W + 1) // 2)
    ok = torch.equal(x, exp)
flag = torch.tensor([1.0 if ok else 0.0], device=dev)
dist.all_reduce(flag)
chk_msg = ("PASS: dst=0 得到 Σ_r 输入_r = [36,72,108,144]" if flag.item() == 1
           else "FAIL: dst 校验未通过")


# ---------- 2) 数据传输量模型 vs 实际流量（大张量 x ITERS 次） ----------

# 单位约定: 尺寸/流量/带宽一律十进制——GB = 1e9 B, GB/s = 1e9 B/s;
# NVLink 计数器原始读数为 KiB(2^10 B), 按 KiB*1024/1e9 换算为 GB。
S = 16 * 10**9                                 # 每 rank 输入（16 GB, 十进制）
t = torch.randn(S // 4, device=dev)
op = lambda: dist.reduce(t, dst=0)
model_gb = 7 / 8 * S / 1e9                     # 环形: (W-1)/W * S
alg_bytes = S
s_desc = "每 rank 输入 16 GB, 归约到 dst=0"



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
    print(f"=== reduce（归约到一卡） ===")
    print(f"[语义] {chk_msg}")
    print(f"[规模] {s_desc}; 迭代 {ITERS} 次")
    print(f"[耗时] 每次 {ms:.3f} ms | algbw {alg_bytes / (ms / 1e3) / 1e9:.0f} GB/s"
          f" | busbw(环形口径) {model_gb * 1e9 / (ms / 1e3) / 1e9:.0f} GB/s")
    print(f"[流量] NVLink 实测: 每 GPU 平均 {measured:.4f} GB/次 (min {min(d):.4f} / max {max(d):.4f})")
    print(f"[对账] 环形模型 {model_gb:.4f} GB/次/GPU -> 实测/模型 = {ratio:.3f}")
dist.barrier()
dist.destroy_process_group()

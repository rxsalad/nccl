#!/usr/bin/env python3
"""broadcast（广播） —— 单文件 NCCL 微基准：语义验证 + NVLink 实测流量 + 环形模型对账

运行（Pod 内，8 GPU 单节点）:
    torchrun --nproc_per_node=8 broadcast.py

严格定义（W 个 rank，r 为本 rank 下标）:
    out_r = x_src  (∀r)——src rank 的张量逐元素复制到每一个 rank。
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

v = torch.tensor([1., 2., 3., 4.], device=dev)
x = v.clone() if rank == 0 else torch.zeros(4, device=dev)
dist.broadcast(x, src=0)
ok = torch.equal(x, v)
flag = torch.tensor([1.0 if ok else 0.0], device=dev)
dist.all_reduce(flag)
chk_msg = ("PASS: 8 个 rank 全部得到 src=0 的 [1,2,3,4]" if flag.item() == W
           else "FAIL: 仅 %d/%d rank 通过" % (int(flag.item()), W))


# ---------- 2) 数据传输量模型 vs 实际流量（大张量 x ITERS 次） ----------
# 单位约定: 尺寸/流量/带宽一律十进制——GB = 1e9 B, GB/s = 1e9 B/s;
# NVLink 计数器原始读数为 KiB(2^10 B), 按 KiB*1024/1e9 换算为 GB。

S = 16 * 10**9                                 # src 持有的消息（16 GB, 十进制）
t = torch.randn(S // 4, device=dev) if rank == 0 else torch.empty(S // 4, device=dev)
op = lambda: dist.broadcast(t, src=0)
model_gb = 7 / 8 * S / 1e9                     # 环形: (W-1)/W * S
alg_bytes = S
s_desc = "src(rank0) 持有 16 GB, 广播到 8 卡"


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
    print(f"=== broadcast（广播） ===")
    print(f"[语义] {chk_msg}")
    print(f"[规模] {s_desc}; 迭代 {ITERS} 次")
    print(f"[耗时] 每次 {ms:.3f} ms | algbw {alg_bytes / (ms / 1e3) / 1e9:.0f} GB/s"
          f" | busbw(环形口径) {model_gb * 1e9 / (ms / 1e3) / 1e9:.0f} GB/s")
    print(f"[流量] NVLink 实测: 每 GPU 平均 {measured:.4f} GB/次 (min {min(d):.4f} / max {max(d):.4f})")
    print(f"[对账] 环形模型 {model_gb:.4f} GB/次/GPU -> 实测/模型 = {ratio:.3f}")
dist.barrier()
dist.destroy_process_group()

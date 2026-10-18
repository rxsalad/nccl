#!/usr/bin/env python3
"""all_to_all 稀疏路由（单 k 单进程版）: torchrun --nproc_per_node=8 a2a_sparse.py <k>

每卡把 C=2 GB/块的数据只发往 k 个目标（k=1..7；k=7 即"发其余全部卡"，与等分
a2a 的外部流量等价）。MoE dispatch 抽象：每块数据只需到达命中目标卡。
预期：每卡线上字节 = k×C——通信量 ∝ 有效流量（目标数）。

方法注意（实测教训，见文档 §7/未验证项）：变长 a2a 在"同进程内先跑过其他模式的
a2a/集合通信"后，测量窗口会携带确定性超额流量（warmup 不吸收、机制未深挖）；
**全新进程的首窗测量在所有实验中均干净**——故每 k 独立进程、单窗测量。
"""
import os, re, subprocess, sys
import torch
import torch.distributed as dist

ITERS = 5
C_BYTES = 2 * 10 ** 9        # 每目标块 2 GB（十进制）
K = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def nvln():
    """每 GPU NVLink Tx（KiB，累计值）"""
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

# ---------- 1) 语义验证（小张量, 同 k 模式, 各 rank 自报） ----------
targets = [(rank + j + 1) % W for j in range(K)]
send_sz = [4 if t in targets else 0 for t in range(W)]
recv_sz = [4 if (rank - s) % W in range(1, K + 1) else 0 for s in range(W)]
xs = torch.cat([torch.full((4,), float((rank + 1) * 10 + (t + 1)), device=dev,
                           dtype=torch.float32)
                for t in sorted(targets)])   # 输入块按目标 rank 升序（send_sz 按目标序切分）
out = torch.empty(sum(recv_sz), device=dev)
dist.all_to_all_single(out, xs, recv_sz, send_sz)
exp = torch.tensor([(s + 1) * 10 + (rank + 1)
                    for s in range(W) for _ in range(recv_sz[s])],
                   device=dev, dtype=torch.float32)
ok = torch.equal(out, exp)
if rank == 0:
    print(f"[k={K} 语义] {'PASS' if ok else 'FAIL'}")
dist.barrier()

# ---------- 2) 单窗测量（全新进程首窗） ----------
n_blk = C_BYTES // 4
send_sz = [n_blk if t in targets else 0 for t in range(W)]
recv_sz = [n_blk if (rank - s) % W in range(1, K + 1) else 0 for s in range(W)]
x = torch.randn(K * n_blk, device=dev)
out = torch.empty(K * n_blk, device=dev)


def op():
    dist.all_to_all_single(out, x, recv_sz, send_sz)


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
    d = [(y - x0) * 1024 / 1e9 / ITERS for x0, y in zip(b, a)]  # 每 GPU 每次 GB
    measured = sum(d) / W
    model = K * C_BYTES / 1e9
    print(f"[k={K}] 每次 {ms:.3f} ms | 每卡实测 {measured:.4f} GB"
          f" (分布 {[round(v, 4) for v in d]})"
          f" | 模型 k×C = {model:.0f} GB | 比值 {measured / model:.3f}")
dist.barrier()
dist.destroy_process_group()

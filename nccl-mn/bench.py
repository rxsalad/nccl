#!/usr/bin/env python3
"""NCCL 多节点微基准（NVLink + RDMA 双计数器；GB 十进制；规模自适应）。

启动（由 run.sh 驱动，每节点一条 torchrun）:
  torchrun --nnodes=<N> --nproc_per_node=8 --node-rank=<r> --master-addr=<A> \
           --master-port=<P> bench.py --op <op> [--gb N] [--iters K]

- 语义验证与 nccl-sn 相同（期望值随 W 自适应，如 all_reduce 期望 (W(W+1)/2)×[1,2,3,4]）。
- 每节点 local_rank 0 负责采集: 本节点 NVLink 每 GPU Tx（nvidia-smi）+
  本节点 RDMA 每 rail 字节（/sys/class/infiniband/mlx5_*/ports/1/counters/port_xmit_data×4，
  IB 规范 4 字节单位）。
- 输出: 每节点一行 RESULT JSON: nv_gb（8 GPU 分布）、rdma_gb（8 rail 分布）、
  wire_gb_per_gpu（NVLink+RDMA 合计/卡）、model_gb（环形模型）与比值。
"""
import argparse, fcntl, json, os, re, socket, struct, subprocess
from array import array as _arr
import torch
import torch.distributed as dist

RAILS = ["mlx5_%d" % i for i in (0, 2, 4, 6, 8, 10, 12, 14)]


def nvln():
    """本节点每 GPU NVLink Tx（KiB，累计值）"""
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


# ---- RDMA 字节计数：raw ioctl 读 netdev ethtool 统计（tx_vport_rdma_unicast_bytes，只计 RDMA）----
# 方案来自 single-node traffic-tools/roce_bytes_sampler.py（vLLM 实验已验证可用）。
SIOCETHTOOL = 0x8946
_SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_RAIL_NET = ["fabric%d" % i for i in range(0, 16, 2)]     # mlx5_0,2,…,14 的 netdev


def _ioctl(dev, buf):
    fcntl.ioctl(_SOCK.fileno(), SIOCETHTOOL,
                struct.pack("16sP", dev.encode(), buf.buffer_info()[0]))


def _names(dev):
    b = _arr("B", struct.pack("IIQI", 0x37, 0, 1 << 1, 0) + b"\0" * 4); _ioctl(dev, b)
    n = struct.unpack_from("I", b.tobytes(), 16)[0]
    b = _arr("B", struct.pack("III", 0x1b, 1, n) + b"\0" * (n * 32)); _ioctl(dev, b)
    raw = b.tobytes()[12:12 + n * 32]
    return n, [raw[i * 32:(i + 1) * 32].split(b"\0")[0].decode() for i in range(n)]


_NSTAT, _NAMES = _names(_RAIL_NET[0])
_TXI = _NAMES.index("tx_vport_rdma_unicast_bytes")
_RXI = _NAMES.index("rx_vport_rdma_unicast_bytes")
_BUFS = {d: _arr("B", struct.pack("II", 0x1d, _NSTAT) + b"\0" * (_NSTAT * 8)) for d in _RAIL_NET}


def rdma_tx():
    """本节点 8 rail 的 RDMA Tx 字节（累计值，只计 RDMA 流量）"""
    out = []
    for d in _RAIL_NET:
        _ioctl(d, _BUFS[d])
        v = struct.unpack_from("Q" * _NSTAT, _BUFS[d].tobytes(), 8)
        out.append(v[_TXI])
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--op", required=True,
                choices=["broadcast", "reduce", "all_reduce", "all_gather",
                         "reduce_scatter", "all_to_all", "rs_ag"])
ap.add_argument("--gb", type=int, default=16, help="每 rank 输入 GB（十进制）；AG 块=gb/8，RS 输出=gb/8")
ap.add_argument("--iters", type=int, default=5)
args = ap.parse_args()

rank = int(os.environ["RANK"]); local = int(os.environ["LOCAL_RANK"])
W = int(os.environ["WORLD_SIZE"])
node = rank // 8
torch.cuda.set_device(local)
dist.init_process_group("nccl")
dev = "cuda:%d" % local

# ---------- 1) 语义验证（小张量，W 自适应） ----------
S4 = W * (W + 1) // 2
op_name = args.op
ok = False
if op_name == "broadcast":
    v = torch.tensor([1., 2., 3., 4.], device=dev)
    x = v.clone() if rank == 0 else torch.zeros(4, device=dev)
    dist.broadcast(x, src=0)
    ok = torch.equal(x, v)
elif op_name == "reduce":
    x = torch.arange(1, 5, device=dev, dtype=torch.float32) * (rank + 1)
    dist.reduce(x, dst=0)
    if rank == 0:
        ok = torch.equal(x, torch.arange(1, 5, device=dev, dtype=torch.float32) * S4)
    else:
        ok = True
elif op_name == "all_reduce":
    x = torch.arange(1, 5, device=dev, dtype=torch.float32) * (rank + 1)
    dist.all_reduce(x)
    ok = torch.equal(x, torch.arange(1, 5, device=dev, dtype=torch.float32) * S4)
elif op_name == "all_gather":
    c = torch.full((4,), float(rank + 1), device=dev)
    out = [torch.empty(4, device=dev) for _ in range(W)]
    dist.all_gather(out, c)
    ok = all(torch.equal(out[j], torch.full((4,), float(j + 1), device=dev)) for j in range(W))
elif op_name == "reduce_scatter":
    x = torch.cat([torch.full((4,), float(j + 1), device=dev) for j in range(W)])
    o = torch.empty(4, device=dev)
    dist.reduce_scatter_tensor(o, x)
    ok = torch.equal(o, torch.full((4,), float(W * (rank + 1)), device=dev))
elif op_name == "all_to_all":
    x = torch.cat([torch.full((4,), float((j + 1) * 100 + (rank + 1)), device=dev)
                   for j in range(W)])                      # rank r 的第 j 块 → rank j
    o = torch.empty_like(x)
    dist.all_to_all_single(o, x)
    ok = all(torch.equal(o[j * 4:(j + 1) * 4],
                         torch.full((4,), float((rank + 1) * 100 + (j + 1)), device=dev))
             for j in range(W))
elif op_name == "rs_ag":
    x = torch.cat([torch.full((4,), float(j + 1), device=dev) for j in range(W)])
    o = torch.empty(4, device=dev)
    outs = [torch.empty(4, device=dev) for _ in range(W)]
    dist.reduce_scatter_tensor(o, x)
    dist.all_gather(outs, o)
    ok = all(torch.equal(outs[j], torch.full((4,), float(W * (j + 1)), device=dev))
             for j in range(W))
flag = torch.tensor([1.0 if ok else 0.0], device=dev)
dist.all_reduce(flag)
chk = "PASS" if flag.item() == W else "FAIL(%d/%d)" % (int(flag.item()), W)

# ---------- 2) 大张量基准 ----------
S = args.gb * 10 ** 9            # 每 rank 输入字节
C = S // 8                       # AG 输入块 / RS 输出段（gb=16 时 = 2 GB）
n = S // 4                       # fp32 元素数
if op_name == "broadcast":
    t = torch.randn(n, device=dev) if rank == 0 else torch.empty(n, device=dev)
    op = lambda: dist.broadcast(t, src=0)
    model = (W - 1) / W * S
elif op_name == "reduce":
    t = torch.randn(n, device=dev)
    op = lambda: dist.reduce(t, dst=0)
    model = (W - 1) / W * S
elif op_name == "all_reduce":
    t = torch.randn(n, device=dev)
    op = lambda: dist.all_reduce(t)
    model = 2 * (W - 1) / W * S
elif op_name == "all_gather":
    c = torch.randn(C // 4, device=dev)
    out = [torch.empty(C // 4, device=dev) for _ in range(W)]
    op = lambda: dist.all_gather(out, c)
    model = (W - 1) * C
elif op_name == "reduce_scatter":
    t = torch.randn(n, device=dev)
    o = torch.empty(n // W, device=dev)
    op = lambda: dist.reduce_scatter_tensor(o, t)
    model = (W - 1) * (S // W)
elif op_name == "all_to_all":
    t = torch.randn(n, device=dev)
    o = torch.empty(n, device=dev)
    op = lambda: dist.all_to_all_single(o, t)
    model = (W - 1) / W * S
elif op_name == "rs_ag":
    t = torch.randn(n, device=dev)
    o = torch.empty(n // W, device=dev)
    outs = [torch.empty(n // W, device=dev) for _ in range(W)]
    def op():
        dist.reduce_scatter_tensor(o, t)
        dist.all_gather(outs, o)
    model = 2 * (W - 1) * (S // W)

for _ in range(3):
    op()
torch.cuda.synchronize(); dist.barrier(); torch.cuda.synchronize()
b_nv = nvln() if local == 0 else None
b_rd = rdma_tx() if local == 0 else None
e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
dist.barrier()
e0.record()
for _ in range(args.iters):
    op()
e1.record(); torch.cuda.synchronize()
ms = e0.elapsed_time(e1) / args.iters
dist.barrier(); torch.cuda.synchronize()
a_nv = nvln() if local == 0 else None
a_rd = rdma_tx() if local == 0 else None

if local == 0:
    nv = [(y - x) * 1024 / 1e9 / args.iters for x, y in zip(b_nv, a_nv)]      # 每 GPU 每次 GB
    rd = [(y - x) / 1e9 / args.iters for x, y in zip(b_rd, a_rd)]              # 每 rail 每次 GB
    nv_total = sum(nv)                          # 本节点 NVLink 每次 GB
    rd_total = sum(v for v in rd if v is not None)
    wire_per_gpu = (nv_total + rd_total) / 8     # 折到每卡
    res = dict(node=node, op=op_name, W=W, gb=args.gb, iters=args.iters, chk=chk,
               ms=round(ms, 3),
               algbw=round(S / (ms / 1e3) / 1e9, 0) if op_name != "all_gather" else round(W * C / (ms / 1e3) / 1e9, 0),
               busbw_ring=round(model / (ms / 1e3) / 1e9, 0),
               nv_gb=[round(v, 3) for v in nv], rdma_gb=[round(v, 3) if v is not None else None for v in rd],
               nv_total=round(nv_total, 2), rdma_total=round(rd_total, 2),
               wire_per_gpu=round(wire_per_gpu, 3), model_per_gpu=round(model / 1e9, 2),
               ratio=round(wire_per_gpu / (model / 1e9), 3))
    print("RESULT " + json.dumps(res, ensure_ascii=False), flush=True)
dist.barrier()
dist.destroy_process_group()

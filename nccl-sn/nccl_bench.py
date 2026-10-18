#!/usr/bin/env python3
"""NCCL 单节点（8×B300, NV18 全互联）集合通信微基准。

用法（Pod 内执行, torchrun 拉起 8 进程）:
  torchrun --nproc_per_node=8 nccl_bench.py --op all_reduce --mib 128 --iters 20

每种操作: warmup(3) → 全组同步 → NVLink Tx 快照 → iters 次连续迭代 → 快照;
rank 0 输出一行 JSON("RESULT {...}"): 实测每 GPU 每次发送量 vs 环形模型预测、
algbw/busbw、每 GPU 分布(min/max)。计数器为节点累计值, 取差值即可。

环形模型（N=8, 每卡每次发送字节数, 一次操作）:
  broadcast / reduce:  (N-1)/N  × S          S = 每 rank 输入
  all_reduce:          2(N-1)/N  × S
  all_gather:          (N-1)     × C          C = 每 rank 输入块, 输出 8C
  reduce_scatter:      (N-1)     × C_out      输入 8×C_out
  all_to_all(等分):    (N-1)/N   × S
  rs_ag:               2(N-1)    × C          RS + AG 各 (N-1)×C
"""
import argparse, json, os, re, subprocess, sys
import torch
import torch.distributed as dist

MIB = 2**20
N = 8  # world size（本实验固定单节点 8 卡）


def nvln():
    """每 GPU NVLink Tx（KiB, 累计值, 所有链路求和）→ 8 元素列表"""
    out = subprocess.run(['nvidia-smi', 'nvlink', '-gt', 'd'],
                         capture_output=True, text=True).stdout
    g, tx = -1, {}
    for line in out.splitlines():
        m = re.match(r'GPU (\d+):', line)
        if m:
            g = int(m.group(1))
            tx[g] = 0
            continue
        m = re.search(r'Data Tx: (\d+) KiB', line)
        if m and g >= 0:
            tx[g] += int(m.group(1))
    return [tx[i] for i in range(len(tx))]


def make_op(op, mib, device):
    """返回 (执行函数, 每卡环形模型发送字节, 算法层字节)"""
    numel = mib * MIB // 4          # float32
    chunk = numel // N
    S = numel * 4                   # 输入字节
    C = chunk * 4                   # 块字节
    if op == 'broadcast':
        t = torch.randn(numel, device=device)
        return (lambda: dist.broadcast(t, src=0)), int(S * (N - 1) / N), S
    if op == 'reduce':
        t = torch.randn(numel, device=device)
        return (lambda: dist.reduce(t, dst=0)), int(S * (N - 1) / N), S
    if op == 'all_reduce':
        t = torch.randn(numel, device=device)
        return (lambda: dist.all_reduce(t)), int(S * 2 * (N - 1) / N), S
    if op == 'all_gather':
        c = torch.randn(chunk, device=device)
        out = [torch.empty(chunk, device=device) for _ in range(N)]
        return (lambda: dist.all_gather(out, c)), C * (N - 1), C * N
    if op == 'reduce_scatter':
        t = torch.randn(numel, device=device)
        o = torch.empty(chunk, device=device)
        return (lambda: dist.reduce_scatter_tensor(o, t)), C * (N - 1), C * N
    if op == 'all_to_all':
        t = torch.randn(numel, device=device)
        o = torch.empty(numel, device=device)
        return (lambda: dist.all_to_all_single(o, t)), int(S * (N - 1) / N), S
    if op == 'rs_ag':
        # reduce_scatter + all_gather: 输入 8C, RS 后每卡 1C, AG 再拼回 8C
        # 字节上应 ≈ all_reduce(输入 8C)
        t = torch.randn(numel, device=device)
        o = torch.empty(chunk, device=device)
        outs = [torch.empty(chunk, device=device) for _ in range(N)]

        def f():
            dist.reduce_scatter_tensor(o, t)
            dist.all_gather(outs, o)
        return f, 2 * C * (N - 1), C * N
    raise ValueError(f'unknown op {op}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--op', required=True)
    ap.add_argument('--mib', type=int, default=128,
                    help='每 rank 输入 MiB; all_gather/reduce_scatter 为块大小')
    ap.add_argument('--iters', type=int, default=20)
    args = ap.parse_args()

    rank = int(os.environ['RANK'])
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    device = f'cuda:{local}'

    op, model_bytes, algo_bytes = make_op(args.op, args.mib, device)

    for _ in range(3):
        op()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()

    tx_b = nvln() if rank == 0 else None
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    dist.barrier()
    ev0.record()
    for _ in range(args.iters):
        op()
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / args.iters
    dist.barrier()
    torch.cuda.synchronize()
    tx_a = nvln() if rank == 0 else None

    if rank == 0:
        d = [(a - b) / MIB / args.iters for a, b in zip(tx_a, tx_b)]  # 每 GPU 每次 GiB
        sent = sum(d) / N * 2**30
        res = dict(op=args.op, mib=args.mib, iters=args.iters,
                   host=os.environ.get('HOSTNAME', '?'),
                   ms=round(ms, 3),
                   algbw_gbps=round(algo_bytes / (ms / 1e3) / 1e9, 1),
                   busbw_gbps=round(model_bytes / (ms / 1e3) / 1e9, 1),
                   tx_gib_per_gpu=round(sum(d) / N, 4),
                   tx_gib_min=round(min(d), 4), tx_gib_max=round(max(d), 4),
                   model_gib_per_gpu=round(model_bytes / 2**30, 4),
                   ratio_vs_ring=round(sent / model_bytes, 3),
                   ncl_algo=os.environ.get('NCCL_ALGO', 'default'),
                   )
        print('RESULT ' + json.dumps(res, ensure_ascii=False), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

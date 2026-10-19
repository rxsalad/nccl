# NCCL Collective Communication Measurements: Single-Node → 2-Node → 4-Node

Empirical study of NCCL 2.29.7 collective communication algorithms on B300 GPUs with NVLink and RoCE, covering all 7 core operations at three scales: single-node (8 GPUs), 2-node (16 GPUs), and 4-node (32 GPUs).

## Key Findings

| Question | Answer |
|---|---|
| Is the ring byte model accurate? | **Exact** — ring-family ops all match model 1.000 at every scale (single/2/4-node) |
| What determines performance? | **Per-rail RDMA load**, not data volume — 2 GB/rail → ~41 ms, 8 GB/rail → ~177 ms, at any node count |
| Which algorithm does each op use? | AR→**NVLS** (2-node only); AG/RS→Ring; bcast/reduce→Ring (with dead edge); a2a→P2P direct |
| Why is 2-node all_reduce special? | **NVLS two-level**: intra-node NVSwitch reduce + cross-node head-pair RoCE exchange — 320 GB wire = 2/3 of ring model (480 GB), busbw 721 GB/s |
| Why doesn't NVLS work at 4 nodes? | PyTorch image's graph search fails (timeout + bandwidth model penalty); **solution: SPCX tuner** (coreweave image, platform-verified 768 GB/s = 100% theoretical) |
| How much does manual RS+AG cost vs AR? | 2-node: **2.06×** (losing NVLS); 4-node: **only 1.07×** (both use Ring) |
| Why is a2a always slowest? | **Direct P2P rail-lock**: each GPU's cross-node traffic locked to its own NIC's rail — 4× more per-rail load than ring-family duty-spreading |

## Repository Structure

```
nccl/
├── README.md                          ← this file
├── LICENSE
├── basic-test-en.md                   ← English: environment baseline (dual-plane, connectivity)
├── nccl-sn-en.md                      ← English: single-node (8 GPU, NVLink)
├── nccl-2n-en.md                      ← English: 2-node (16 GPU, NVLink + RoCE)
├── nccl-4n-en.md                      ← English: 4-node (32 GPU, NVLink + RoCE)
├── nccl-sn/                           ← Chinese: single-node
│   ├── nccl-sn.md                     ← main document (637 lines)
│   ├── sn-codes/                      ← one standalone .py per operation
│   │   ├── broadcast.py, reduce.py, all_gather.py, reduce_scatter.py
│   │   ├── all_reduce.py, rs_ag.py, all_to_all.py, a2a_sparse.py
│   ├── scan_channels.sh               ← channel count sweep (1→32)
│   ├── nccl_bench.py                  ← single-run verification utility
│   └── sn-results/                    ← raw measurement data
└── nccl-mn/                           ← Chinese: multi-node
    ├── basic-test.md                  ← environment baseline (dual-plane, 8×8 connectivity)
    ├── nccl-2n.md                     ← 2-node main document (712 lines)
    ├── nccl-4n.md                     ← 4-node main document (344 lines)
    ├── bench.py                       ← W-adaptive benchmark (7 ops, dual counters)
    ├── run.sh                         ← multi-node launcher
    ├── env_inventory.sh               ← environment inventory script
    ├── 8-gpu-8-fabric-pod.yaml        ← Kubernetes StatefulSet (4 Pods × 8 GPU × 8 RoCE)
    └── mn-results-2n-16gb.txt         ← raw 2-node measurement data
```

## Documents (read in order)

| # | Document | Content |
|---|---|---|
| 1 | `basic-test` | Environment: dual-plane architecture (16 NICs/node, 8/Pod), GPU↔rail affinity, 256/256 cross-rail connectivity |
| 2 | `nccl-sn` | Single-node: 7 ops byte models, algorithm fingerprints (Ring/NVLS/P2P), channel scans, NVLS topic |
| 3 | `nccl-2n` | 2-node: ring-family commonality (8 ring orders, dead-edge rules), CROSS_NIC tuning (+15%), NVLS × RoCE architecture |
| 4 | `nccl-4n` | 4-node: five-segment ring orders, NVLS 2→4 boundary (all-or-nothing + SPCX tuner), 2→4 node scaling |

Each document has: table of contents, key-findings table, per-operation sections with strict definition → principle → analogy → code → measured results → five-step analysis (bytes → algorithm → structure → cross-node flow → performance), unverified items, and reproduction guide.

## Environment

- **Hardware**: 4× nodes, each 8× NVIDIA B300 SXM6 (NV18 pairwise NVLink + NVSwitch), 16× RDMA NICs (dual-plane, 8/Pod), 400 Gb/s RoCE per rail
- **Software**: PyTorch 2.13.0+cu130, NCCL 2.29.7, CUDA 13.0
- **Platform**: Kubernetes (DOKS mkc1), StatefulSet with Multus + ip-preserving CNI
- **Topology**: GPU i ↔ mlx5_2i ↔ fabric 2i (per-GPU dedicated rail); spine-routed fabric, cross-rail penalty-free

## Quick Start

```bash
# Single-node (any node with 8 GPUs)
cd nccl-sn/sn-codes && torchrun --nproc_per_node=8 all_reduce.py

# Multi-node (requires Kubernetes cluster)
cd nccl-mn && ./run.sh 2 all_reduce --gb 16 --iters 1   # 2-node
cd nccl-mn && ./run.sh 4 all_reduce --gb 16 --iters 1   # 4-node
```

## Measurement Methodology

- **Dual counters**: NVLink per-GPU Tx (intra-node) + RDMA per-rail Tx (cross-node) — naturally separated
- **Single-shot execution**: warmup 3 → counter snapshot → 1 execution → snapshot (delta = single-run traffic); ≥3 independent runs with bitwise-identical counters (determinism)
- **Five-step analysis** per operation: ① byte reconciliation → ② algorithm determination → ③ concurrency structure → ④ cross-node flow → ⑤ performance
- **Bandwidth metrics**: algbw/busbw as group-level values (t = slowest node), per nccl-tests convention

## Performance Summary (16 GB per rank, single-shot)

### 2-Node (W=16)

| Op | Algorithm | Time | busbw (GB/s) | Wire (GB) |
|---|---|---|---|---|
| broadcast | Ring | 43.9 ms | 342 | 240 |
| reduce_scatter | Ring | 38.9 ms | 386 | 240 |
| **all_reduce** | **NVLS** | **41.6 ms** | **721** | **320** |
| rs_ag | Ring+Ring | 85.5 ms | 351 | 480 |
| all_to_all | P2P | 177.1 ms | 85 | 240 |

### 4-Node (W=32)

| Op | Algorithm | Time | busbw (GB/s) | Wire (GB) |
|---|---|---|---|---|
| broadcast | Ring | 45.2 ms | 343 | 497 |
| reduce_scatter | Ring | 43.0 ms | 360 | 497 |
| **all_reduce** | **Ring** (NVLS dropped) | 89.6 ms | 346 | 994 |
| rs_ag | Ring+Ring | 96.1 ms | 323 | 994 |
| all_to_all | P2P | 251.3 ms | 62 | 506 |

## License

See [LICENSE](LICENSE).

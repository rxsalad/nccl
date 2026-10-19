# 4-Node NCCL Measurements: NVLink + RoCE (4 Pods × 8 B300, W=32)

**Environment and methodology fully inherited from `nccl-mn/nccl-2n.md`**: 4 Pods (pod-0..3, see `basic-test.md`; W = 32, rank = node# × 8 + GPU#), standard environment variables (`nccl-2n.md` §0.2, unchanged), dual counters, **single-shot execution** protocol, five-step template and five-step analysis — shared mechanisms not repeated; **this document records only 4-node new findings and 2→4 node structural differences**.

<a id="toc"></a>
**Table of Contents** (each chapter has ↑ return link)

- [0. Methods & Commonality](#ch-0)
- [1. broadcast](#ch-1) | [2. reduce](#ch-2) | [3. all_gather](#ch-3) | [4. reduce_scatter](#ch-4)
- [5. all_reduce — NVLS abandoned, falls back to Ring](#ch-5) | [6. rs_ag combo](#ch-6) | [7. all_to_all](#ch-7)
- [8. Summary](#ch-8) | [9. Special Topic: NVLS 2→4 Node Boundary](#ch-9)
- [Unverified Items](#ch-10) | [Ops Notes](#ch-11)

**Key Findings at a Glance**:

| Question | Conclusion |
|---|---|
| Does the ring byte model hold at 4 nodes? | **Yes** — five-segment ring orders + 4-hop ladder, ring-family ops all 64/64 exact match |
| Does 4-node all_reduce still use NVLS? | **No** — heads allocated but NVLS comm not established, falls back to Ring (busbw halved 721→346); **root cause: cross-node graph search failure disables intra-node NVSwitch too (all-or-nothing); PyTorch image lacks SPCX tuner** |
| Is timing still determined by per-rail load? | **Yes** — 1.9 GB→43 ms, 2→44, 3.9→90, 7.8→190, 12→251 |
| How much does rs_ag differ from AR? | **Only +7%** — no NVLS to lose, only call boundary remains (vs 2-node 2.06×) |
| What about a2a? | Per-rail **12 GB** rail-locked → 251 ms, slowest in project |
| Why doesn't NVLS work at 4 nodes? | **PyTorch image lacks SPCX tuner** — graph search timeout + bandwidth model penalty; coreweave image solves it (platform verified 768 GB/s) |

<a id="ch-0"></a>
## 0. Methods & Commonality

[↑ Return to TOC](#toc)

Tools, dual counters, single-shot execution, five-step template, bandwidth metrics all same as `nccl-2n.md` §0.1/§0.3 (≥3 independent runs per op with bitwise-identical counters; algbw/busbw use slowest node). All 7 ops verified size invariance (`--gb 1` shape identical) and `NCCL_ALGO=Tree` forcing ineffective (bytes bitwise identical to default).

### 0.1 Ring-Family Commonality (4-node version; mechanisms see nccl-2n §0.4, only differences here)

[↑ Return to TOC](#toc)

- **Channels and rings**: 16 coll channels, each a 32-rank ring, 1 GB per ring (channel count same as 2-node — multi-node graph search result, halved from single-node 32, no performance impact: bottleneck is per-rail RDMA); **8 distinct ring orders, paired** (ch k and k+8 same order), bitwise identical across ops (AR vs broadcast comm verified).
- **Five-segment template**: Each order k = `n0 prefix [0 7 … (k+1)] ∥ n1 block ∥ n2 block ∥ n3 block ∥ n0 suffix [k … 1] →(0)` — n1/n2/n3 each a **complete 8-rank block** (entry GPU k, exit GPU k+1), only node 0 is split.
- **Cross-node ladder (4 hops)**: Each order k's 4 cross-node edges use **the same rule 4 times** — `n0.g(k+1)→n1.gk`, `n1.g(k+1)→n2.gk`, `n2.g(k+1)→n3.gk`, `n3.g(k+1)→n0.gk` (sender always uses its own PXB-affine NIC; NET lines verified 320/320).

| k | Ring order |
|---|---|
| 0 | [0 7 6 5 4 3 2 1] ∥ [8 15 14 13 12 11 10 9] ∥ [16 23 22 21 20 19 18 17] ∥ [24 31 30 29 28 27 26 25] →(0) |
| 1 | [0 7 6 5 4 3 2] ∥ [9 8 15 14 13 12 11 10] ∥ [17 16 23 22 21 20 19 18] ∥ [25 24 31 30 29 28 27 26] ∥ [1] →(0) |
| 2 | [0 7 6 5 4 3] ∥ [10 9 8 15 14 13 12 11] ∥ [18 17 16 23 22 21 20 19] ∥ [26 25 24 31 30 29 28 27] ∥ [2 1] →(0) |
| 3 | [0 7 6 5 4] ∥ [11 10 9 8 15 14 13 12] ∥ [19 18 17 16 23 22 21 20] ∥ [27 26 25 24 31 30 29 28] ∥ [3 2 1] →(0) |
| 4 | [0 7 6 5] ∥ [12 11 10 9 8 15 14 13] ∥ [20 19 18 17 16 23 22 21] ∥ [28 27 26 25 24 31 30 29] ∥ [4 3 2 1] →(0) |
| 5 | [0 7 6] ∥ [13 12 11 10 9 8 15 14] ∥ [21 20 19 18 17 16 23 22] ∥ [29 28 27 26 25 24 31 30] ∥ [5 4 3 2 1] →(0) |
| 6 | [0 7] ∥ [14 13 12 11 10 9 8 15] ∥ [22 21 20 19 18 17 16 23] ∥ [30 29 28 27 26 25 24 31] ∥ [6 5 4 3 2 1] →(0) |
| 7 | [0] ∥ [15 14 13 12 11 10 9 8] ∥ [23 22 21 20 19 18 17 16] ∥ [31 30 29 28 27 26 25 24] ∥ [7 6 5 4 3 2 1] →(0) |

- **Dead-edge rules (same as 2-node)**:

| Op | Dead edge | Location | Idle GPU / dead rail |
|---|---|---|---|
| broadcast | wrap edge pointing to src | k=0: 25→0 (**cross-node**); k=1..7: 1→0 | n0 **GPU1**; **n3 rail 1** |
| reduce | dst outgoing edge 0→next(0) | k=0..6: 0→7 (intra); k=7: 0→15 (**cross-node**) | n0 **GPU0**; **n0 rail 0** |
| AG / RS / AR / rs_ag | none | — | none (all uniform) |

- **Per-rail total formula**: Each node sends exactly 1 cross-edge per order (sender GPU (k+1) mod 8) × 2 channels × channel share; dead edge's rail minus 2 shares.
- **Total wire byte conservation**: broadcast/reduce/RS = (W−1)×S/W = **496 GB**; AG = (W−1)×C = **1984 GB**; AR/rs_ag ring = 2×(W−1)×S/W = **992 GB**.

<a id="ch-1"></a>
## 1. broadcast

[↑ Return to TOC](#toc)

**Definition**: `out_r = x_src` (src = rank 0 = n0.0).

**Principle**: Ring-family commonality §0.1 — dead edge = wrap edge pointing to src (k=0 order's 25→0 happens to be cross-node → n3 rail 1 ≈ 0). Total = 16 rings × 31 active edges × 1 GB = **496 GB**; cross-node = 4-segment ladder 16+16+16+14 = **62 GB** + NVLink 434 GB.

**Results** (measured, single-shot; 4 independent runs bitwise-identical, 43.3–45.2 ms):

| | GPU0 | GPU1 | GPU2–7 | Total |
|---|---|---|---|---|
| node 0 (src side) model / measured | 14 / 14.001 | **0** / **0.001** | 14 / 14.002 | 98 / 98.01 |
| node 1/2/3 model / measured | 14 / 14.002 | 14 / 14.002 | 14 / 14.002 | 112 / 112.01 |

| | rail 0 | rail 1 | rail 2–7 | Total |
|---|---|---|---|---|
| node 0 model / measured | 2 / 2.039 | 2 / 2.042 | 2 / 2.041 | 16 / 16.33 |
| node 1/2 model / measured | 2 / 2.040 | 2 / 2.041 | 2 / 2.041 | 16 / 16.32 |
| node 3 model / measured | 2 / 2.041 | **0** / **0.002** | 2 / 2.041 | 14 / 14.29 |

**Summary** (times 43.69 / 44.46 / 44.79 / 45.22 ms): wire **497.31 GB**. Group (t = 45.22 ms): algbw **354** / busbw **343 GB/s**. Semantics PASS (32/32).

**Analysis**: ① 497.31 = 31×S + 0.3% ✓. ② Algorithm = Ring (source/refutation/simulation 64/64, same as nccl-2n §1②). ③ Isomorphic to 2-node — GPU1 idle, n3 rail 1 dead; only 3-segment → 5-segment, 2-hop → 4-hop ladder. ④ 4-segment ladder 16/16/16/14 GB — n3→n0 segment 14 GB (k=0 order 2 return edges dead); relay nodes save no bandwidth (n1/n2 each relay 16 GB). ⑤ 45.22 ms ≈ lower bound 41 + ~4 ms; **nearly identical to 2-node (43.9 ms)** — per-rail load unchanged, 4 hops pipelined. `NCCL_ALGO=Tree` forcing ineffective ✓.

<a id="ch-2"></a>
## 2. reduce

[↑ Return to TOC](#toc)

**Definition**: `out_dst[i] = Σ_r x_r[i]` (only dst = rank 0 holds result).

**Principle**: broadcast's mirror — dead edge = dst outgoing edge (k=7 order's 0→15 **cross-node n0→n1** → n0 rail 0 ≈ 0, n0 GPU0 idle). Total 496 GB; cross-node 62 GB (n0→n1 segment 14, other three segments 16 each all active).

**Results** (measured; multiple runs bitwise-identical, 42.4–46.8 ms):

| | GPU0 | GPU1–7 | Total |
|---|---|---|---|
| node 0 (dst side) model / measured | **0** / **0.001** | 14 / 14.002 | 98 / 98.01 |
| node 1/2/3 model / measured | 14 / 14.002 | 14 / 14.002 | 112 / 112.01 |

| | rail 0 | rail 1–7 | Total |
|---|---|---|---|
| node 0 model / measured | **0** / **0.002** | 2 / 2.041 | 14 / 14.29 |
| node 1/2/3 model / measured | 2 / 2.040 | 2 / 2.041 | 16 / 16.33 |

**Summary** (times 43.68 / 42.44 / 42.60 / 43.38 ms): wire **497.30 GB**. Group (t = 43.68 ms): algbw **366** / busbw **355 GB/s**. Semantics PASS.

**Analysis**: ① 497.30 = 31×S + 0.26% ✓. ② Algorithm = Ring (simulation 64/64). ③ Mirror signature identical to 2-node. ④ 4-segment ladder 14/16/16/16 GB — exact swap of broadcast's 16/16/16/14. ⑤ 43.68 ms ≈ lower bound 41 + ~3 ms. `NCCL_ALGO=Tree` forcing ineffective ✓.

<a id="ch-3"></a>
## 3. all_gather

[↑ Return to TOC](#toc)

**Definition**: Concatenate without sum; input block C = 2 GB/rank, output W×C = **64 GB/rank**.

**Principle**: Ring-family + block-assembly rounds — each rank's block traverses 31 hops, each edge carries 31/32 channel share (3.875 GB), **no dead edge**. Per-GPU = (W−1)×C = **62 GB**; total = **1984 GB**.

**Results** (measured; multiple runs bitwise-identical, 189.6–190.8 ms):

| | GPU/rail 0–7 (uniform) | Total |
|---|---|---|
| Each node NVLink model / measured | 54.25 / 54.253 | 434 / 434.03 |
| Each node RDMA model / measured | 7.75 / 7.904 | 62 / 63.23 |

**Summary** (times 189.92 / 190.00 / 189.88 / 189.69 ms): wire **1989.0 GB**. Group (t = 190.00 ms): algbw = W×C ÷ t = **337** / busbw = **326 GB/s**. Semantics PASS.

**Analysis**: ① 1989.0 = 1984 + 0.25% ✓. ② Algorithm = Ring (simulation 64/64 uniform; no NVLS comm line — NVLS would give ≈2 GB/rank, measured 62 GB ✗); `NCCL_ALGO=Tree` identical to default ✓. ③ No dead edge, fully symmetric uniform. ④ 4-segment ladder each 62 GB bidirectional symmetric. ⑤ 190.0 ms = lower bound 158 + **~32 ms** (4-node residual significantly larger than 2-node's +12 ms). **vs 2-node**: wire 4.13×, time 2.13× — per-rail 7.75/3.75 = 2.07×, time scales linearly with rail load.

<a id="ch-4"></a>
## 4. reduce_scatter

[↑ Return to TOC](#toc)

**Definition**: Sum without concatenate; input 16 GB/rank, output S/W = **0.5 GB/rank**.

**Principle**: Ring-family + "reduce-while-forwarding" rounds — no dead edge, each edge carries 0.96875 GB. Per-GPU = (W−1)×S/W = **15.5 GB**; total = **496 GB**.

**Results** (measured; multiple runs bitwise-identical, 40.5–45.6 ms):

| | GPU/rail 0–7 (uniform) | Total |
|---|---|---|
| Each node NVLink model / measured | 13.5625 / 13.564 | 108.5 / 108.51 |
| Each node RDMA model / measured | 1.9375 / 1.976 | 15.5 / 15.81 |

**Summary** (times 42.65 / 43.00 / 42.97 / 42.83 ms): wire **497.28 GB**. Group (t = 43.00 ms): algbw **372** / busbw **360 GB/s**. Semantics PASS.

**Analysis**: ① 497.28 = 496 + 0.26% ✓. ② Algorithm = Ring (simulation 64/64; no NVLS comm line); `NCCL_ALGO=Tree` identical ✓. ③ No dead edge uniform. ④ 4-segment ladder each 15.5 GB symmetric. ⑤ 43.0 ms = lower bound 39.5 + ~3.5 ms; **same magnitude as 2-node (38.9 ms)**; fastest for same total (496 GB).

<a id="ch-5"></a>
## 5. all_reduce — 4-Node NVLS Abandoned, Falls Back to Ring

[↑ Return to TOC](#toc)

**Definition**: Element-wise sum, every rank holds complete result.

**Principle (the 4-node story)**: Implementation = **Ring** (not NVLS!):

- **Log evidence**: Has `NVLS Head h: h, h+8, h+16, h+24` (heads allocated as usual) **but no `NVLS comm` line** — NVLS resources not established, tuner falls back to Ring. Root cause §9.
- Ring implementation = RS phase + AG phase fused in single call: per-GPU = 2×(W−1)×S/W = **31 GB**; total = **992 GB** (vs 2-node NVLS 320 GB — **3.1× ring-model equivalent**); per-node NVLink 217 + RDMA 31.5.

**Results** (measured; 5 independent runs bitwise-identical counters, **bimodal timing** 89 / 131 ms clusters (one 121 ms) — see ⑤):

| | GPU/rail 0–7 (uniform) | Total |
|---|---|---|
| Each node NVLink model / measured | 27.125 / 27.127 | 217 / 217.01 |
| Each node RDMA model / measured | 3.875 / 3.952 | 31 / 31.62 |

**Summary** (run 1 times 89.19 / 89.61 / 89.54 / 89.19 ms): wire **994.5 GB** (= ring model 992 + 0.25%; vs 2-node NVLS 320 GB — **4-node carries 3.1× more**). Group (t = 89.61 ms): algbw **179** / busbw **346 GB/s** (2-node NVLS was 721 — **busbw halved after losing NVLS**). Semantics PASS.

**Analysis**: ① 994.5 = 992 + 0.25% ✓. ② Algorithm = Ring (bytes = ring model 64/64; no NVLS comm line; `NCCL_ALGO=Tree` bytes identical to default ✓). ③ No dead edge uniform. ④ 4-segment ladder each 31 GB symmetric. ⑤ **Two major findings**:
  - **Bimodal timing**: 5 runs 89 / 121 / 131 ms three clusters, bytes bitwise identical — lower bound 79 ms (per-rail 3.875 GB @ ~49 GB/s), fast cluster = 79+10, slow = 79+52; mechanism unverified.
  - **NVLS 2→4 node boundary**: 2-node 41.6 ms (NVLS, 320 GB, busbw 721) → 4-node 89–133 ms (Ring, 994 GB, busbw 346) — 3.1× more bytes, busbw halved (root cause and solution §9).

<a id="ch-6"></a>
## 6. reduce_scatter + all_gather Combo (= all_reduce)

[↑ Return to TOC](#toc)

**Definition**: RS then AG, semantically identical to all_reduce.

**Principle**: Manual two steps, same set of rings (§0.1); each phase 15.5 GB/rank, total **31 GB**; total 992 GB (**identical bytes to §5's Ring AR** — 4-node AR is already Ring).

**Results** (measured; multiple runs bitwise-identical, 91.8–96.1 ms):

| | GPU/rail 0–7 (uniform) | Total |
|---|---|---|
| Each node NVLink model / measured | 27.125 / 27.127 | 217 / 217.01 |
| Each node RDMA model / measured | 3.875 / 3.952 | 31 / 31.62 |

**Summary** (times 96.06 / 95.81 / 95.52 / 95.63 ms): wire **994.5 GB**. Group (t = 96.06 ms): algbw **167** / busbw **323 GB/s**. Semantics PASS.

**Analysis**: ① 994.5 = 992 + 0.25% ✓, step-additivity holds. ② Algorithm = Ring + Ring (no NVLS comm line — **no NVLS to lose**); `NCCL_ALGO=Tree` identical ✓. ③ Both phases no dead edge uniform. ⑤ **Key comparison**: 96.1 ms vs AR fast cluster 89.6 ms = **only +7%** — 2-node's 2.06× (NVLS 41.6 vs manual 85.5) collapses to 1.07×: **the cost of a call boundary depends on whether there's NVLS to lose**; when algorithm is the same (Ring vs Ring+Ring), single-call fusion saves only ~7%.

<a id="ch-7"></a>
## 7. all_to_all

[↑ Return to TOC](#toc)

**Definition**: W×W transpose; each segment S/W = 0.5 GB.

**Principle**: P2P direct — each rank 31 direct pairs (**7 NVLink + 24 RDMA**: 3 remote nodes × 8 ranks); NIC rule same as ring family (both ends use own rail, 6336 NET lines verified). Per-GPU = NVLink **3.5 GB** + RDMA **12 GB** (24 peers × 0.5, **all via own rail — each rail locked to 12 GB**); total = **496 GB**.

**Results** (measured; multiple runs bitwise-identical, 249.9–251.4 ms):

| | GPU/rail 0–7 (uniform) | Total |
|---|---|---|
| Each node NVLink model / measured | 3.5 / 3.502 | 28 / 28.01 |
| Each node RDMA model / measured | 12 / 12.324 | 96 / 98.59 |

**Summary** (times 249.90 / 250.42 / 251.11 / 251.25 ms): wire **506.4 GB** (= 496 + 2.1%). Group (t = 251.25 ms): algbw **64** / busbw (accounting) **62 GB/s**. Semantics PASS.

**Analysis**: ① 506.4 = 496 + 2.1% ✓. ② Algorithm = P2P direct (model 64/64 + 6336 NET direct lines); `NCCL_ALGO=Tree` identical ✓. ③ **Rail locking worsens at 4 nodes** — per-GPU 24 GB locked to own rail (2-node was 8 GB). ⑤ 251.25 ms = lower bound 245 (per-rail 12 GB @ ~49 GB/s) + ~6 ms — **slowest in project** (per-rail 12 GB = 6× broadcast); vs 2-node (172 ms) = 1.46×.

<a id="ch-8"></a>
## 8. Summary: Seven Operations One Table (W=32, 4 Nodes)

[↑ Return to TOC](#toc)

| Op | Input → Output (per rank / total) | Algorithm | Time | algbw | busbw | NVLink/GPU | RDMA/rail | Wire total |
|---|---|---|---|---|---|---|---|---|
| broadcast | 16 GB (src only) → 16 GB/rank (512) | Ring | 45.22 ms | 354 | 343 | 14 (GPU1=0) | 2.0 (n3 rail1=0) | 497 |
| reduce | 16 GB/rank (512) → 16 GB (dst only) | Ring | 43.68 ms | 366 | 355 | 14 (GPU0=0) | 2.0 (n0 rail0=0) | 497 |
| all_gather | 2 GB/rank (64) → 64 GB/rank (**2048**) | Ring | 190.00 ms | 337 | 326 | 54.25 uniform | 7.75 | 1989 |
| reduce_scatter | 16 GB/rank (512) → 0.5 GB/rank (16) | Ring | 43.00 ms | **372** | 360 | 13.5625 uniform | 1.9375 | 497 |
| all_reduce | 16 GB/rank (512) → 16 GB/rank (512) | **Ring (NVLS dropped)** | 89.61 ms (bimodal 89–133) | 179 | 346 | 27.125 uniform | 3.875 | **994** |
| rs_ag | same as all_reduce | Ring + Ring | 96.06 ms | 167 | 323 | 27.125 uniform | 3.875 | 994 |
| all_to_all | 16 GB/rank (512) → 16 GB/rank (512, transposed) | P2P | 251.25 ms | 64 | 62 | 3.5 uniform | **12 (rail-locked)** | 506 |

- **Data volume ≠ time; per-rail load is what matters**:

| Per-rail RDMA | 1.9375 | 2.0 | 3.875 | 7.75 | 12.0 |
|---|---|---|---|---|---|
| Rail lower bound | 39.5 ms | 41 ms | 79 ms | 158 ms | 245 ms |
| Measured | RS 43.0 | bcast 45.2 / reduce 43.7 | AR 89.6 / rs_ag 96.1 | AG 190 | a2a 251 |

- **2 → 4 node time comparison**:

| | bcast | reduce | AG | RS | **AR** | rs_ag | a2a |
|---|---|---|---|---|---|---|---|
| 2-node (ms) | 43.9 | 45.1 | 89.0 | 38.9 | **41.6 (NVLS)** | 85.5 | 172 |
| 4-node (ms) | 45.2 | 43.7 | 190.0 | 43.0 | **89.6 (Ring)** | 96.1 | 251 |
| Ratio | 1.03 | 0.97 | 2.13 | 1.11 | **2.15** | 1.12 | 1.46 |

  **broadcast/reduce/RS barely change with node count** (per-rail load unchanged, 4 hops pipelined); **AG/a2a scale with per-rail load** (2.07× / 1.46×); **AR's 2.15× = rail load 1.94× × losing NVLS**.

- **Structure taxonomy**: Ring family 6 + P2P 1; NVLS family **disappears** at 4 nodes (§9).
- **busbw ranking**: AR 346 ≈ reduce 355 ≈ bcast 343 > AG 326 > rs_ag 323 > **a2a 62**.

<a id="ch-9"></a>
## 9. Special Topic: NVLS 2→4 Node Boundary

[↑ Return to TOC](#toc)

### 9.1 Phenomenon

2-node AR uses NVLS (busbw 721, wire 320 GB); 4-node AR falls back to Ring (busbw 346, wire 994 GB). Log evidence:

| | 2-node | 4-node |
|---|---|---|
| NVLS multicast available | ✓ | ✓ |
| NVLS Head allocation | `h, h+8` (2 members) | `h, h+8, h+16, h+24` (4 members) ✓ |
| **NVLS comm established** | ✓ | **✗ (no line)** |
| Connected NVLS tree | ✓ | **✗** |
| AR actual algorithm | NVLS | **Ring** |

### 9.2 Root Cause (three-layer evidence, not a hardware limitation)

1. **Graph search failure** (`init.cc:1273` → `search.cc`): `ncclTopoSearchRec` for NVLS pattern returns 0 channels at 4 nodes → `nvlsSupport = 0` → NVLS comm not established. Head tree goes from 1 level (2 nodes) to 2 levels, search space doubles, cannot find solution satisfying bandwidth constraints within timeout.
2. **Bandwidth model penalty** (`tuning.cc:285`): `nNodes <= 2 ? bwIntra : bwInter` — ≤2 nodes uses NVLink bandwidth for NVLS estimate, >2 nodes uses RoCE bandwidth (much slower), tuner slashes NVLS expected bandwidth to RoCE level.
3. **Platform document proves 4-node NVLS works**: Same fabric, same 4 Pods, coreweave image (SPCX tuner v5) + nccl-tests measured **busbw 768.64 GB/s = 100% of theoretical limit**.

**Root cause**: SPCX tuner is an NCCL plugin (`libnccl-tuner.so`) that replaces default graph search/tuning logic. PyTorch image lacks this plugin (log shows `TUNER/Plugin: Could not find`), must use generic search — generic search times out / constraints too tight for 4-node NVLS. **Solution**: Use an NCCL build with SPCX tuner (e.g., coreweave image).

### 9.2.1 Why cross-node failure also disables intra-node NVSwitch: NVLS is all-or-nothing

**Key design limitation**: NCCL treats NVLS as an **indivisible atomic package** — intra-node NVSwitch reduction + cross-node head tree exchange + NVSwitch multicast is one integral unit; there is no "intra-node NVLS + cross-node Ring" hybrid mode:

```
NCCL selectable algorithms (atomic units):
  Ring  = all Ring (intra-node ring + cross-node ring)
  Tree  = all Tree
  NVLS  = NVSwitch reduce + head tree + NVSwitch multicast (one integral)
  ─────────────────────────────────────────────────────
  Does not exist: NVSwitch reduce + Ring exchange + NVSwitch multicast ← no such "assembly"
```

Source code proof (`init.cc:1273`):

```c
if (graphs[NCCL_ALGO_NVLS]->nChannels == 0)
  comm->nvlsSupport = comm->nvlsChannels = 0;   // one-vote veto
```

NVLS graph search treats intra-node and cross-node as **one integral graph** — head tree can't find solution at 4 nodes ⇒ entire NVLS graph returns 0 channels ⇒ `nvlsSupport = 0` ⇒ **perfectly functional intra-node NVSwitch (chip reduce/multicast) is also discarded**. Hardware is fine, but NCCL's algorithm is indivisible — cannot do "intra-node NVLS + cross-node Ring".

### 9.2.2 How 4-node NVLS should theoretically work (architecture unchanged, only exchange deepens)

| Step | 2-node (measured) | 4-node (theoretical, SPCX verified feasible) |
|---|---|---|
| ① Upward reduce | NVSwitch 8→1, get partial | **Identical** |
| ② Cross-node exchange | head h ══ p ══▶ peer head (**1-to-1**) | head group of **4 heads** via tree exchange (**1-to-3**) |
| ③ Fusion | head locally adds **2** partials → final | head locally adds **4** partials → final |
| ④ Downward multicast | head multicasts final → 8 GPUs | **Identical** |

Only difference: ② changes from pairwise to tree aggregation (depth log₂4 = 2, per-rail traffic ×2). SPCX tuner's value is **a different search logic that can build the 4-node head tree** — the architecture itself is fully isomorphic to 2-node.

### 9.3 Consequences and Application-Level Implications

- **Bytes**: AR jumps from 320 GB to 994 GB (**3.1×**);
- **busbw**: 721 → 346 (**halved**) — AR drops from "project bandwidth king" back to ordinary ring-family member;
- **rs_ag vs AR**: 2-node 2.06× → 4-node **1.07×** — "the cost of a call boundary" is essentially "whether there's NVLS to lose";
- **Application level**: On this cluster, **all_reduce's VIP treatment is only valid within 2 nodes** — ≥4-node AR has no essential difference from manual RS+AG (994 GB same bytes, +7% time); for scale-out budget using ring model (2×(W−1)/W×S), or switch to coreweave image to unlock NVLS (768 GB/s).

<a id="ch-10"></a>
## Unverified Items

[↑ Return to TOC](#toc)

- **4-node AR bimodal timing** (89 / 121 / 131 ms three clusters, bytes bitwise identical): mechanism unverified — candidates: runtime protocol/algorithm variant selection, communicator initialization measurement differences.
- AG 4-node ~32 ms residual (2-node was +12 ms): candidate — 31-hop block-assembly round pipeline depth.
- Ring-family mechanism unverified items (ring order construction, c and c+8 reuse, a2a rail-lock optimization, etc.) **all same as nccl-2n unverified items**.
- TODO: a2a sparse routing 4-node version; AR bimodal statistical distribution.

<a id="ch-11"></a>
## Ops Notes (Reproduction)

[↑ Return to TOC](#toc)

```bash
cd /root/work/data/distributed-inference/combinations/nccl-mn
./run.sh 4 <op> --gb 16 --iters 1                       # Baseline (single-shot)
./run.sh 4 <op> --gb 1  --iters 1                        # Size invariance
NCCL_ALGO=Tree ./run.sh 4 <op> --gb 16 --iters 1        # ALGO forcing (bytes should match default)
# AR: run multiple times to observe bimodal (89–133 ms, bytes unchanged)
# Evidence capture (same as nccl-2n ops notes, --nnodes=4 --node-rank=<0..3>):
#   grep points: "NVLS Head" (4-member groups) and "NVLS comm" (should be absent — 4-node doesn't build NVLS)
```

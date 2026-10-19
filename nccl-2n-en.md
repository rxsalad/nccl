# 2-Node NCCL Measurements: NVLink + RoCE (2 Pods × 8 B300, W=16)

**Environment**: See `nccl-mn/basic-test.md` (pod-0 + pod-1, each 8×B300 + 8×400 Gb/s RoCE rails, GPU i ↔ fabric 2i). **Methodology inherited from nccl-sn** (`nccl-sn/nccl-sn.md`), document structure aligned: **§0 Methods → §1–7 Operations → §8 Summary → §9 NVLS×RoCE → Unverified → Appendix → Ops**. **New in this document**: NVLink (intra-node) / RoCE (cross-node) dual counters, 2-node algorithm determination (ring-order byte-exact model + NVLS mechanism decode), rail mapping verification, `NCCL_CROSS_NIC` tuning.

<a id="toc"></a>
**TOC** (↑ return links at each chapter start)

- [0. Methods & Commonality](#ch-0)
- [1. broadcast](#ch-1) | [2. reduce](#ch-2) | [3. all_gather](#ch-3) | [4. reduce_scatter](#ch-4)
- [5. all_reduce (NVLS)](#ch-5) | [6. rs_ag combo](#ch-6) | [7. all_to_all](#ch-7)
- [8. Summary](#ch-8) | [9. NVLS × RoCE Topic](#ch-9)
- [Unverified](#ch-10) | [Appendix](#ch-11) | [Ops](#ch-12)

**Key Findings**:

| Question | Conclusion |
|---|---|
| Does the ring byte model hold multi-node? | **Yes** — ring-family 5 ops all 32/32 exact match (§0.4: ring orders, NIC rules, dead edges) |
| What does multi-node all_reduce use? | **NVLS two-level**: intra-node NVSwitch reduce/multicast + cross-node head-pair RoCE exchange — per-GPU S + S/8, wire 320 GB = 2/3 ring model (§5/§9) |
| What determines timing? | **Per-rail RDMA load**: 1.875 GB→38.9 ms, 2 GB→41.6, 3.75→89, 8→177 (§8) |
| How to choose `NCCL_CROSS_NIC`? | This cluster has no cross-rail penalty ⇒ `=1` both ends use own rail, fastest (~15% faster than 0) (§0.2) |
| Why is a2a slowest? | **Direct P2P rail-lock**: per-GPU 8 GB cross-node traffic locked to own rail (§7) |
| Does manual RS+AG equal all_reduce? | **Bytes identical, time 2.06×** — call boundary loses NVLS + two phases each pay rail floor (§6) |

<a id="ch-0"></a>
## 0. Methods & Commonality

[↑ Return](#toc)

### 0.1 Tools & Protocol

- Tools: `bench.py` (7 ops, W-adaptive, dual counters) + `run.sh <nnodes> <op> [--gb N] [--iters K]`.
- **Notation**: W = total ranks = nodes × 8 (16 for 2 nodes); rank = node# × 8 + GPU# (shorthand **nX.Y** = node X GPU Y).
- **Dual counters**: NVLink per-GPU Tx (intra-node) + RDMA per-rail Tx bytes (cross-node) — naturally separated.
- **Flow protocol**: **Single-shot execution** (`--iters 1`: warmup 3 → counter snapshot → **1 execution** → snapshot, delta = single-run flow, no multi-run averaging); ≥3 independent runs per op, counters **bitwise identical** (determinism; path consistency per nccl-sn §0.2 — channel graphs fixed at communicator setup).
- **Bandwidth metrics**: algbw / busbw as **group-level (collective) single values**, t = slowest node (collective is synchronous; per-node RESULT values are just local event-timing view differences); definitions per nccl-sn §0.6.
- **Size**: `--gb 16` = 16 GB input per rank (decimal); all 7 ops verified **size invariance** (`--gb 1` distribution shape identical).

### 0.2 Standard Environment Variables

| Variable | Value | Official meaning | This cluster measured |
|---|---|---|---|
| `NCCL_SOCKET_IFNAME` | `eth0` | Restrict socket (bootstrap/OOB) to interface | Log `OOB eth0` ✓ |
| `NCCL_IB_TC` | `104` | RoCE TOS/DSCP (QoS) | Cluster convention, not separately tested |
| `NCCL_IB_GID_INDEX` | `3` | RoCE GID index | Matches GID table, not separately tested |
| `NCCL_CROSS_NIC` | **`1`** (non-default) | Whether ring/tree can use different NICs at each end | Both ends use own nearest NIC, in-fabric lane-change possible, ~15% faster than 0 |
| `NCCL_NET_DISABLE_INTRA` | `1` | Disallow intra-node network loopback (source-only, not in official docs) | Guarantees clean dual-counter separation ✓ |
| `NCCL_PXN_DISABLE` | `1` | Disable PXN relay | Same 43 ms but +30 GB wire ⇒ keep disabled |

- **`NCCL_CROSS_NIC` three levels vs measured**:
  - **0**: Same NIC index for ring/tree (avoids crossing network rails) — official for strict rail designs with **slow cross-rail**;
  - **1**: Allow different NICs per end — official for converged fabrics;
  - **2** (default): Prefer same, allow different if faster.
  This cluster is **spine-routed: rail-optimized with no cross-rail penalty** (basic-test §6: 256/256 cross-rail all pass, RTT no difference). Measured: 0 ⇒ both ends use **receiver's** GPU rail, sender crosses PCIe (~50 ms); **1 ⇒ both use own rail, cross-rail in fabric (43.4 ms, current standard)**; 2 ⇒ ring rebuilt as **same-GPU-index pairs** (43.3 ms). Conclusion: **when cross-rail is penalty-free, both ends using nearest NIC is fastest**.
- **`NCCL_PXN_DISABLE`**: PXN allows cross-node communication using **non-local NIC** via NVLink through intermediate GPU. Enabled: same 43 ms but +30 GB wire (one extra NVLink hop per cross-edge); `CROSS_NIC=1` achieves same effect with zero extra wire — keep disabled.
- **`NCCL_NET_DISABLE_INTRA`**: NCCL compares **direct path** vs **NIC loopback** for same-node GPU pairs. This cluster's NV18 full-mesh means direct is always optimal; setting `1` is a **methodological safeguard** — guarantees dual-counter separation (intra=NVLink, cross=RDMA) independent of tuner choices ✓.

### 0.3 Test Template (§1–§7 each op follows this)

| Step | Command / action | Acceptance criteria |
|---|---|---|
| ① Baseline ×2 | `./run.sh 2 <op> --gb 16 --iters 1` | Two runs bitwise-identical counters, `chk=PASS` |
| ② Evidence capture | Same config `NCCL_DEBUG=INFO NCCL_DEBUG_FILE=...` | Ring orders + Trees + NVLS + NET lines |
| ③ Simulation reconciliation | Model per op's byte model (ring §0.4, NVLS/P2P §5/§7) | All distribution numbers match **GB-exact** |
| ④ Two invariances | `--gb 1`; `NCCL_ALGO=Tree` forcing | Shape unchanged; bytes identical to default |
| ⑤ Five-step analysis | ① bytes → ② algorithm → ③ structure → ④ cross-node flow → ⑤ performance | Each step has "model vs measured" |

(Op differences only in: **semantics/expectations, byte model, dead-edge location**.)

### 0.4 Ring-Family Commonality (broadcast / reduce / all_gather / reduce_scatter / rs_ag shared, measured)

- **Channels and rings**: 16 coll channels, each a 16-rank ring, S/16 = 1 GB per ring. **16 channels yield only 8 distinct ring orders, paired** (ch k and k+8 identical, verified by exact log matching). Ring orders **fixed at communicator setup, operation-independent** — §1–§4, §6 all **bitwise identical**. **Channel count halved from single-node's 32**: more complex multi-node topology reduces graph search results; **no performance impact** — bottleneck shifted from NVLink to per-rail RDMA.
- 8 distinct ring orders (`Channel k/16` log, from rank 0; `∥` = cross-node block boundary, `→(0)` = wrap edge):

| k | Ring order | Outbound (send rail → recv rail) | Return (send rail → recv rail) | Wrap |
|---|---|---|---|---|
| 0 | 0 7 6 5 4 3 2 1 ∥ 8 15 14 13 12 11 10 9 →(0) | 1→8 (r1→r0) | 9→0 (r1→r0) | 9→0 (**cross**) |
| 1 | 0 7 6 5 4 3 2 ∥ 9 8 15 14 13 12 11 10 ∥ 1 →(0) | 2→9 (r2→r1) | 10→1 (r2→r1) | 1→0 |
| 2 | 0 7 6 5 4 3 ∥ 10 9 8 15 14 13 12 11 ∥ 2 1 →(0) | 3→10 (r3→r2) | 11→2 (r3→r2) | 1→0 |
| 3 | 0 7 6 5 4 ∥ 11 10 9 8 15 14 13 12 ∥ 3 2 1 →(0) | 4→11 (r4→r3) | 12→3 (r4→r3) | 1→0 |
| 4 | 0 7 6 5 ∥ 12 11 10 9 8 15 14 13 ∥ 4 3 2 1 →(0) | 5→12 (r5→r4) | 13→4 (r5→r4) | 1→0 |
| 5 | 0 7 6 ∥ 13 12 11 10 9 8 15 14 ∥ 5 4 3 2 1 →(0) | 6→13 (r6→r5) | 14→5 (r6→r5) | 1→0 |
| 6 | 0 7 ∥ 14 13 12 11 10 9 8 15 ∥ 6 5 4 3 2 1 →(0) | 7→14 (r7→r6) | 15→6 (r7→r6) | 1→0 |
| 7 | 0 ∥ 15 14 13 12 11 10 9 8 ∥ 7 6 5 4 3 2 1 →(0) | 0→15 (r0→r7) | 8→7 (r0→r7) | 1→0 |

- **Uniform pairing rule (all 8 orders)**: Each order k's two cross-node edges are "**GPU (k+1) mod 8 sends → GPU k receives**" — **both ends use own rail** (sender from rail (k+1) mod 8, receiver at rail k), different GPU numbers ⇒ **true cross-rail in fabric** (one lane change, neither side crosses PCIe; 128/128 NET lines verified).
- **Per-rail total formula**: Each node sends exactly 1 outbound per order (GPU (k+1) mod 8) × 2 channels × channel share; dead-edge rail minus 2 shares.
- **Dead-edge rules (operation semantics determine, position follows src/dst)**:

| Op | Dead edge | Location | Idle GPU |
|---|---|---|---|
| broadcast | **edge into src** (src's predecessor recv-only) | k=0: wrap 9→0 (**cross** → n1 rail 1 ≈ 0); k=1..7: 1→0 (intra) | **GPU1** (src predecessor) |
| reduce | **dst outgoing edge** (dst never sends) | k=0..6: 0→7 (intra); k=7: 0→15 (**cross** → n0 rail 0 ≈ 0) | **GPU0** (dst itself) |
| AG / RS | **no dead edge** | — | none (uniform) |

- **Idle position bound to src/dst location** (structural artifact, not by design): n0 intra-node ring order fixed as `0-7-6-5-4-3-2-1` (8 orders unchanged), each order differs only in cross-node chain splice point (each order cuts edge `(k+1)→k`); dead-edge position from operation semantics applied to this fixed ring — change src/dst, idle GPU changes accordingly (verified).

<a id="ch-1"></a>
## 1. broadcast

[↑ Return](#toc)

**Definition**: `out_r = x_src` — same as nccl-sn §1.

**Principle**: Ring (proof in ②). Dead edge = edge into src (`broadcast.h`: `nextRank == root → directRecv`), 15 active edges per ring. Total = 16 rings × 15 edges × S/16 = **15×S = 240 GB**; cross-node 30 GB (16 outbound + 14 active return) + NVLink 210 GB.

**Analogy**: 16 relay teams, each carrying 1/16 of the script, passing around a ring from the speaker, stopping just before the speaker (that member receives only). Crossing two cities via two bridges, **each takes their nearest bridge** — sender from their own bridge (rail), receiver from recipient's bridge; 7/8 teams must re-cross on return to deliver to back-row — source of node 1→node 0 reverse traffic.

**Results** (measured, single-shot; 3 runs bitwise-identical, 43.36–43.90 ms):

**NVLink per GPU (GB/run; model = ring simulation)**:

| | GPU0 | GPU1 | GPU2–7 | Total |
|---|---|---|---|---|
| node 0 (src) model / measured | 14 / 14.001 | **0** / **0.001** | 14 / 14.002 | 98 / 98.01 |
| node 1 model / measured | 14 / 14.002 | 14 / 14.002 | 14 / 14.002 | 112 / 112.01 |

**RDMA per rail (GB/run; rail = sender's own GPU#)**:

| | rail 0 | rail 1 | rail 2–7 | Total |
|---|---|---|---|---|
| node 0 model / measured | 2 / 2.039 | 2 / 2.042 | 2 / 2.041 | 16 / 16.33 |
| node 1 model / measured | 2 / 2.041 | **0** / **0.002** | 2 / 2.041 | 14 / 14.29 |

**Summary** (times 43.47 / 43.90 ms): NVLink 210.02 GB + RDMA 30.62 GB = **wire 240.64 GB**. Group (t = 43.90 ms): algbw **364** / busbw **342 GB/s**. Semantics PASS.

**Analysis**:
① 240.64 = 15×S + 0.3% ✓.
② **Algorithm = Ring** (3 evidences: source code `tuning.cc:274` only allows Ring; tree counter-evidence — if tree, node 1 RDMA ≈ 0, measured 14.29 GB ✗; ring simulation **32/32 all match**).
③ Dead edges per §0.4: k=0 wrap 9→0 cross-node → n1 rail 1 ≈ 0; k=1..7 intra → **GPU1 idle in all 8 orders** → n0 [14, **0**, 14×6]. Size invariance ✓.
④ "Backflow" (n1→n0 = 14.29 GB) = **14 active return edges** (k=1..7 × 2 ch); cause: 7/8 ring orders split n0 ranks around n1 block (rank 1 always adjacent to src).
⑤ 43.47 ms; lower bound ≈ 41 ms (per-rail 2 GB @ ~49 GB/s); actual = 41 + ~2 ms. **Backflow time cost ≈ 0** (full-duplex reverse). `NCCL_ALGO=Tree` forcing ineffective ✓.

<a id="ch-2"></a>
## 2. reduce

[↑ Return](#toc)

**Definition**: `out_dst[i] = Σ x_r[i]` — only dst holds result.

**Principle**: broadcast's mirror — dead edge = dst outgoing (k=7: 0→15 **cross n0→n1** → n0 rail 0 ≈ 0, GPU0 idle). Total 240 GB; cross-node 30 GB (n0→n1: 14, rest 16 each).

**Results** (measured; 4 runs bitwise-identical, 41.5–45.9 ms):

| | GPU0 | GPU1–7 | Total |
|---|---|---|---|
| node 0 (dst) model / measured | **0** / **0.001** | 14 / 14.002 | 98 / 98.01 |
| node 1 model / measured | 14 / 14.002 | 14 / 14.002 | 112 / 112.01 |

| | rail 0 | rail 1–7 | Total |
|---|---|---|---|
| node 0 model / measured | **0** / **0.002** | 2 / 2.041 | 14 / 14.28 |
| node 1 model / measured | 2 / 2.040 | 2 / 2.041 | 16 / 16.32 |

**Summary** (times 45.12 / 44.98 ms): wire **240.62 GB**. Group (t = 45.12 ms): algbw **355** / busbw **332 GB/s**. Semantics PASS.

**Analysis**: ① ✓. ② Ring (simulation 32/32). ③ Mirror of broadcast (idle: GPU0 = dst itself; dead rail: n0 rail 0). ④ Ladder 14/16/16/16 — exact swap of broadcast's 16/16/16/14. ⑤ Same magnitude as 2-node. `NCCL_ALGO=Tree` ineffective ✓.

<a id="ch-3"></a>
## 3. all_gather

[↑ Return](#toc)

**Definition**: Concatenate without sum; input block C = 2 GB/rank, output W×C = 32 GB/rank.

**Principle**: Ring family + block-assembly rounds — each rank's block traverses exactly 15 hops, each edge carries 15/16 channel share (1.875 GB), **no dead edge**. Per-GPU = (W−1)×C = **30 GB**; total = **480 GB**.

**Results** (measured; 3 runs bitwise-identical, 88.9–91.5 ms):

| | GPU/rail 0–7 (uniform) | Total |
|---|---|---|
| n0 NVLink model / measured | 26.25 / 26.252 | 210 / 210.01 |
| n0 RDMA model / measured | 3.75 / 3.824 | 30 / 30.60 |
| n1 NVLink model / measured | 26.25 / 26.252 | 210 / 210.01 |
| n1 RDMA model / measured | 3.75 / 3.825 | 30 / 30.60 |

**Summary** (times 88.97 / 88.93 ms): wire **481.22 GB**. Group (t = 88.97 ms): algbw = W×C÷t = **360** / busbw = **337 GB/s**. Semantics PASS.

**Analysis**: ① 481.22 = 480 + 0.25% ✓. ② Ring (32/32; no NVLS comm — if NVLS, per-GPU ≈ 2 GB, measured 30 GB ✗); `NCCL_ALGO=Tree` identical ✓. ③ No dead edge → fully symmetric uniform. ④ Out = return = 30 GB each. ⑤ 89 ms; lower bound ≈ 76.5 (per-rail 3.75 GB @ ~49 GB/s) + ~12 ms residual. **vs broadcast: wire 2×, time 2× — linear per GB**.

<a id="ch-4"></a>
## 4. reduce_scatter

[↑ Return](#toc)

**Definition**: Sum without concatenate; input 16 GB/rank, output S/W = 1 GB/rank.

**Principle**: Ring + "reduce-while-forwarding" — no dead edge, each edge 0.9375 GB. Per-GPU = (W−1)×S/W = **15 GB**; total = **240 GB**.

**Results** (measured; 4 runs bitwise-identical, 38.8–42.8 ms):

| | GPU/rail 0–7 (uniform) | Total |
|---|---|---|
| Each node NVLink model / measured | 13.125 / 13.126 | 105 / 105.01 |
| Each node RDMA model / measured | 1.875 / 1.912 | 15 / 15.30 |

**Summary** (times 38.86 / 38.77 ms): wire **240.62 GB**. Group (t = 38.86 ms): algbw **412** / busbw **386 GB/s**. Semantics PASS.

**Analysis**: ① 240.62 = 240 + 0.26% ✓. ② Ring (32/32; no NVLS comm). ③ Uniform. ④ Out = return = 15 GB each. ⑤ **38.86 ms — tightest to rail floor** (1.875 GB @ ~49 GB/s = 38.3 ms, residual only ~0.6 ms, project minimum). **Same total as broadcast but faster** (38.9 vs 43.5 ms): 6% less per-rail Tx + no dead edge.

<a id="ch-5"></a>
## 5. all_reduce — NVLS: Intra-node NVSwitch + Cross-node Head-Pair Direct

[↑ Return](#toc)

**Definition**: Element-wise sum, every rank holds complete result.

**Principle**: **NVLS** (proof in ②):
- **Up (intra)**: Each GPU sends full S = 16 GB to local NVSwitch, **in-chip reduce** (GPU zero-sum);
- **Cross (head pairs)**: 8 head pairs (log `NVLS Head h: h, h+8`) each hold 1/8 tensor (S/8 = 2 GB), **exchange partial sums via dedicated rail h bidirectionally** (NET: `7[7] -> 15[7] via NET/IB/7` — **same-GPU-index pair, same rail, no lane change**) → each head collects both nodes' partials;
- **Down (intra)**: Each head multicasts its 1/8 = 2 GB final sum **to all 8 local GPUs** via NVSwitch fanout.
- Per-GPU = **S + S/8 = 18 GB NVLink** (up 16 + multicast 2) + **per-node RDMA 16 GB** (8 heads × 2 GB each direction); wire = 2×(144+16) = **320 GB** — vs ring model 480 GB **saves 1/3**. Details §9.

**Analogy**: SN's accountant extended to two cities: each city has an accountant (NVSwitch) — everyone submits full ledger locally (chip computes), then **8 pairs of dedicated accountants** (head pairs, each managing 1/8) exchange partials via private phone lines (rails), then each city's accountant reads the full ledger to all.

**Results** (measured; 5 runs bitwise-identical, 41.57–41.66 ms — tightest in project):

| | GPU/rail 0–7 (uniform) | Total |
|---|---|---|
| n0 NVLink model / measured | 18 / 18.003 | 144 / 144.02 |
| n0 RDMA model / measured | 2 / 2.044 | 16 / 16.35 |
| n1 NVLink model / measured | 18 / 18.003 | 144 / 144.02 |
| n1 RDMA model / measured | 2 / 2.044 | 16 / 16.35 |

**Summary** (times 41.59 / 41.63 ms): wire **320.74 GB** (vs 2-node ring 480 GB — **1/3 less**). Group (t = 41.63 ms): algbw **384** / busbw **721 GB/s** (higher than actual per-GPU line rate 433 GB/s — NVLS "ring-equivalent" fingerprint). Semantics PASS.

**Analysis**: ① 320.74 = 320 + 0.23% ✓. ② **Algorithm = NVLS** (bytes = S+S/8 exact; NVLS comm line unique to AR; head-pair NET lines match per-rail 2 GB). ③ AR **doesn't use ring** — 8 head-pair direct lines, same-GPU-index, same rail (only place "same GPU = same rail" holds). ④ Bidirectional symmetric 16 GB each way, no ring no dead edge. ⑤ 41.59 ms; lower bound ≈ 41 ms; **NVLS saves bytes not time** (but busbw 721 = 2× broadcast). `NCCL_ALGO`/`NCCL_NVLS_ENABLE` forcing both ineffective (§9/Unverified).

<a id="ch-6"></a>
## 6. reduce_scatter + all_gather Combo (= all_reduce)

[↑ Return](#toc)

**Definition**: RS then AG — semantically and byte-wise identical to all_reduce.

**Principle**: Manual two calls, same rings (§0.4); RS phase (input 16 GB) + AG phase (input = RS output 1 GB/rank). Each phase 15 GB/rank, total **30 GB**; wire 480 GB (= ring AR model, 2× NVLS's 320 GB).

**Results** (measured; 4 runs bitwise-identical, 85.2–87.5 ms):

| | GPU/rail 0–7 (uniform) | Total |
|---|---|---|
| n0 NVLink model / measured | 26.25 / 26.252 | 210 / 210.01 |
| n0 RDMA model / measured | 3.75 / 3.824 | 30 / 30.59 |
| n1 NVLink model / measured | 26.25 / 26.252 | 210 / 210.01 |
| n1 RDMA model / measured | 3.75 / 3.825 | 30 / 30.60 |

**Summary** (times 85.49 / 85.24 ms): wire **481.21 GB**. Group (t = 85.49 ms): algbw **187** / busbw **351 GB/s**. Semantics PASS.

**Analysis**: ① Step-additivity exact (RS 105.01+15.30 + AG 105.00+15.29). ② Ring + Ring (no NVLS comm — **manual two-step gets no NVLS**). ⑤ **Three-price comparison**: **NVLS single 41.6 ms < manual 85.5 ms (2.06×)** — SN's "call boundary costs, algorithm choice costs more" amplified from 1.38× to 2.06×.

<a id="ch-7"></a>
## 7. all_to_all

[↑ Return](#toc)

**Definition**: W×W transpose; each segment S/W = 1 GB.

**Principle**: P2P direct — each rank 15 direct pairs (7 NVLink + 8 RDMA); NIC rule same as ring (both ends own rail, 1088 NET lines verified). Per-GPU = NVLink **7 GB** + RDMA **8 GB** (**all via own rail — each rail locked to 8 GB**); total = **240 GB**.

**Analogy**: Dealing cards — 16 houses each with 16 stacks, stack j goes only to house j; local 7 delivered in person, cross-city 8 by courier — **courier from own bridge (rail), delivered at recipient's bridge**.

**Results** (measured; 4 runs bitwise-identical, 169.5–177.0 ms):

| | GPU/rail 0–7 (uniform) | Total |
|---|---|---|
| n0 NVLink model / measured | 7 / 7.003 | 56 / 56.03 |
| n0 RDMA model / measured | 8 / 8.220 | 64 / 65.76 |
| n1 NVLink model / measured | 7 / 7.003 | 56 / 56.03 |
| n1 RDMA model / measured | 8 / 8.216 | 64 / 65.73 |

**Summary** (times 172.06 / 177.05 ms): wire **243.55 GB**. Group (t = 177.05 ms): algbw **90** / busbw (accounting) **85 GB/s**. Semantics PASS.

**Analysis**: ① 243.55 = 240 + 1.5% ✓. ② P2P direct (model 32/32; 1088 NET direct lines). ③ **Direct P2P rail-lock cost**: each GPU's 8 GB locked to own rail (vs ring family spreading duty to 8 rails, 2 GB each). ⑤ 172 ms = lower bound 163 (per-rail 8 GB @ ~49 GB/s) + ~9 ms — **slowest in project** (4× broadcast). `NCCL_ALGO=Tree` ineffective ✓.

<a id="ch-8"></a>
## 8. Summary: Seven Operations One Table (W=16, 2 Nodes)

[↑ Return](#toc)

| Op | Input → Output (per rank / total) | Algorithm | Time | algbw | busbw | NVLink/GPU | RDMA/rail | Wire total |
|---|---|---|---|---|---|---|---|---|
| broadcast | 16 GB (src) → 16 GB/rank (256) | Ring | 43.90 ms | 364 | 342 | 14 (GPU1=0) | 2.0 | 240 |
| reduce | 16 GB/rank (256) → 16 GB (dst) | Ring | 45.12 ms | 355 | 332 | 14 (GPU0=0) | 2.0 | 240 |
| all_gather | 2 GB/rank (32) → 32 GB/rank (**512**) | Ring | 88.97 ms | 360 | 337 | 26.25 uniform | 3.75 | 480 |
| reduce_scatter | 16 GB/rank (256) → 1 GB/rank (16) | Ring | 38.86 ms | **412** | 386 | 13.125 uniform | 1.875 | 240 |
| all_reduce | 16 GB/rank (256) → 16 GB/rank (256) | **NVLS** | 41.63 ms | 384 | **721** | 18 uniform | 2.0 | **320** (saves 1/3) |
| rs_ag | same as all_reduce | Ring + Ring | 85.49 ms | 187 | 351 | 26.25 uniform | 3.75 | 480 |
| all_to_all | 16 GB/rank (256) → 16 GB/rank (256, transposed) | P2P | 177.05 ms | 90 | 85 | 7 uniform | **8.0 (locked)** | 240 |

- **Data volume ≠ time; per-rail load is**:

| Per-rail RDMA | 1.875 | 2.0 | 3.75 | 8.0 |
|---|---|---|---|---|
| Rail floor | 38.3 ms | 41 ms | 76.5 ms | 163 ms |
| Measured | RS 38.9 | bcast 43.9 / AR 41.6 / reduce 45.1 | AG 89.0 / rs_ag 85.5 | a2a 177 |

- **Structure taxonomy**: Ring 5 + **NVLS 1** (AR: intra NVSwitch + cross head pairs) + **P2P 1** (a2a: direct rail-locked); AR's NVLS single-call only (rs_ag 41.6 → 85.5 = 2.06×).
- **busbw ranking**: **AR 721** (NVLS 2×) > RS 386 > rs_ag 351 > bcast 342 ≈ AG 337 ≈ reduce 332 > **a2a 85**.

<a id="ch-9"></a>
## 9. NVLS × RoCE: Two-Level Reduction Architecture

[↑ Return](#toc)

Core question: NVSwitch's multimem primitives cover only **local node's 8 GPUs** — how do two nodes' partial sums get combined? Answer: two-level structure — intra-node NVSwitch, cross-node RoCE head pairs.

### 9.1 What is a Head: Each GPU Owns 1/8 of the Tensor

Log `NVLS Head h: h, h+8` — **head group h = same-GPU-index pair** (n0 GPU h + n1 GPU h), connected via rail h point-to-point. Tensor split into **8 slices** (S/8 = 2 GB each), **slice h's cross-node merging handled by head group h**. Every GPU is a head. Why 8: spreads RDMA duty across all 8 rails (S/8 each) and multicast duty across all 8 GPUs.

### 9.2 Four-Step Journey of One Slice (slice h, S/8 = 2 GB)

```
node 0                                              node 1
① Upward reduce: 8 GPUs → NVSwitch (in-chip sum)     8 GPUs → NVSwitch
    → head h reads partial_0[h] (GPU Rx)               → head h+8 reads partial_1[h]
② Cross-node exchange (RoCE transport only):
    head h ══ 2 GB ══▶ head h+8    (rail h, point-to-point)
    head h ◀══ 2 GB ══ head h+8    (bidirectional simultaneous)
③ Fusion (GPU local compute, no NVSwitch, no network):
    head h: partial_0[h] + partial_1[h] = final[h]
    head h+8: same → final[h]
④ Downward multicast (NVSwitch fanout only):
    head h ──final[h]──▶ NVSwitch ──one-to-all──▶ 8 local GPUs
```

**Who computes, who transports**:

| Step | Computer | Transporter | Bytes/GPU |
|---|---|---|---|
| ① Upward | **NVSwitch** (in-chip sum) | GPU→NVSwitch (NVLink) | +S |
| ② Exchange | **none** (pure transport) | RoCE rail h | +S/8 RDMA |
| ③ Fusion | **head GPU** (local add) | — (no network) | — |
| ④ Multicast | **none** (pure fanout) | head→NVSwitch→8 GPUs | +S/8 NVLink |

**Counter reconciliation (all 3 match)**:
- Per-GPU NVLink Tx = S + S/8 = 18.003 GB ✓ (① up-write + ④ multicast; ② is Rx, ③ is local)
- Per-node RDMA Tx = 8 × S/8 = 16.35 GB ✓ (② 8 heads × 2 GB)
- Per-rail RDMA Tx = S/8 = 2.044 GB ✓ (head h uses rail h — **same-GPU-index pair, only place "same GPU = same rail"**)

### 9.3 Why Two-Level is Necessary

- NVSwitch can't handle cross-node (multimem registered per-node only).
- Cross-node segment is **plain point-to-point RDMA writes** — no "in-network reduction" semantics; InfiniBand SHARP-style switch reduction doesn't exist on RoCE. The "switch" for cross-node is two GPUs (heads adding).
- RoCE segment is both **necessary** and the **bottleneck** (§9.4) — NVLS's byte savings (320 vs 480 GB) are all on NVLink side, can't save time.

### 9.4 Size Behavior (1–32 GB sweep)

| S (GB) | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Time (ms) | 3.15 | 5.61 | 10.84 | 21.07 | 41.63 | 82.86 |
| busbw (GB/s) | 595 | 668 | 692 | 712 | 721 | 724 |

- **Linear fit t ≈ 2.57·S + 0.6 ms**: slope 2.57 ms/GB = per-rail S/8 @ ~49 GB/s (**rail-dominated**); intercept only ~0.6 ms — four steps fully pipelined.
- **No Ring crossover point**: busbw from 595 (1 GB) asymptotes to 729 GB/s — two-level structure effective from 1 GB upward (single-node's "size boundary" question: no boundary multi-node).

### 9.5 Comparison with nccl-sn §9

- SN's "AR wins" extended and amplified: byte savings 1/3 (480→320), busbw 2× (721); "bytes saved ≠ time saved" explained: **time determined by RoCE segment alone**, NVLink savings become pipeline headroom not wall-clock.
- Unresolved: `NCCL_ALGO`/`NCCL_NVLS_ENABLE` forcing ineffective (candidate: symmetric memory/GIN path bypassing classic algorithm selection).

<a id="ch-10"></a>
## Unverified Items

[↑ Return](#toc)

- All 7 ops completed per §0.3 template (§1–§7, summary §8). TODO: a2a sparse routing multi-node version.
- **AR path selection mystery**: `NCCL_ALGO=Tree/Ring` and `NCCL_NVLS_ENABLE=0` don't change byte structure — mechanism unverified (candidate: symmetric memory/GIN path).
- NVLS phase timing not separately measured (counters have no time dimension; linear fit only infers pipelining).
- a2a rail-lock optimization space unexplored.
- Ring order construction algorithm details unverified (all 8 orders decoded by measurement, §0.4 table).
- Ring order c/c+8 reuse mechanism unexplored.
- ~2 ms residual (broadcast/RS) and ~12 ms (AG) unattributed.
- `NCCL_CROSS_NIC` default 2 (unset) behavior untested.
- ~~Backflow coefficient~~ (solved: 14 active return edges); ~~NCCL_ALGO ineffective for broadcast~~ (solved: Ring-only candidate set); ~~n0 NVLink non-uniform~~ (solved: ring-order idle — dead/cross-edge sender).

<a id="ch-11"></a>
## Appendix: NCCL Source Code Research (v2.29.7-1)

[↑ Return](#toc)

| Question | Finding | Relation to measurements |
|---|---|---|
| broadcast/reduce algorithms | `tuning.cc:274`: Ring only; `broadcast.h`/`reduce.h` implement Ring × 3 protocols only | ✓ Algorithm = Ring; ALGO forcing ineffective |
| AG/RS algorithms | `tuning.cc:275`: {PAT, Ring, NVLS, CollNet}, no Tree | ✓ §3/§4 determination |
| AR algorithms | `tuning.cc:278`: excludes only PAT — but forcing ineffective (§5⑤) | ✓ NVLS default + forcing mystery |
| Trees log format | `init.cc:1306`: `[c] down0/down1/down2->rank->up` | ✓ Tree resources exist but unused by broadcast |
| Ring order log | `rings.cc`: `dumpLine` prints `Channel c/16 : …` | ✓ Ring simulation matches exactly |
| Cross-node rail selection | NET lines: send IB# = sender GPU#, receive IB# = receiver GPU# | ✓ Per-rail distribution predictions exact |
| Edge data direction | `broadcast.h`: root `directSend`; `nextRank==root` `directRecv`; others `directRecvCopyDirectSend` | ✓ Dead-edge positions |
| NET_DISABLE_INTRA | `paths.cc:537`: intra-node can loop via NIC if NIC paths faster | ✓ Methodological safeguard basis |

**Conclusion**: broadcast/reduce have **only Ring** in NCCL 2.29.7; ring-family 5 ops' ring orders, NIC rules, byte conservation all explained by §0.4 commonality; all_reduce uses NVLS (intra NVSwitch + cross-node head pairs, 320 GB = 2/3 ring); all_to_all uses P2P direct (per-GPU cross traffic locked to own rail, slowest in project).

<a id="ch-12"></a>
## Ops Notes (Reproduction)

[↑ Return](#toc)

```bash
cd /root/work/data/distributed-inference/combinations/nccl-mn
./run.sh 2 <op> --gb 16 --iters 1                       # Baseline (single-shot)
./run.sh 2 <op> --gb 1  --iters 1                        # Size invariance
for g in 1 2 4 8 16 32; do ./run.sh 2 all_reduce --gb $g --iters 1; done  # §9 size sweep
NCCL_ALGO=Tree ./run.sh 2 <op> --gb 16 --iters 1        # ALGO forcing (bytes should match default)
# Evidence capture (per-process, same config, after measurements):
#   grep: "coll channels" | "Channel ..\/16 :" | "Trees \[0\]" | "NVLS comm" | "NVLS Head" | "via NET/IB" | "^RESULT"
```

# Single-Node NCCL Collective Communication Benchmarking Manual: Definitions, Code & Traffic (B300×8, NVLink)

One operation per chapter, uniform structure: **strict definition → principle walkthrough (8-GPU example: what / how / cost) → intuitive analogy → single-file PyTorch code & run → analysis**. Analysis four-step method: **① byte total reconciliation → ② algorithm determination → ③ concurrency structure → ④ performance** (§7 adds ⑤ advantage measurements). Focus on NCCL itself: operation semantics, byte models, measured traffic, algorithm determination. Document structure: **§0 Methods → §1–7 Operations (components first, then combos) → §8 Summary → §9 NVLS Topic → §10 Unverified → Reproduction Guide**.

- Environment (measured): `8-gpu-8-fabric-pod-0` (deploy per `nccl-mn/8-gpu-8-fabric-pod.yaml`, image `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel`); PyTorch 2.13.0+cu130, NCCL 2.29.7; 8×B300 SXM6 pairwise NV18 (`nvidia-smi topo -m`); NVSwitch supports NVLS multicast (NCCL log `NVLS multicast support is available ... NVLS_NCHANNELS 24`).
- Convention: **measured** = this Pod's command output or logs; **inferred** = derived from mechanism; **source** = NCCL v2.29.7 GitHub code.
- Artifacts: `sn-codes/<op>.py` (one standalone script per op; plus `a2a_sparse.py`); `scan_channels.sh` (channel scan, data source for each §③); `nccl_bench.py` (utility: single-run verification). Raw output: `sn-results/sn-codes-16gb-20261017.txt` (all measured data), `sn-results/sn-codes-20261017.txt` (process log).

<a id="toc"></a>
**TOC** (↑ return links at each chapter start)

- [0. Methods & Commonality](#ch-0)
- [1. broadcast](#ch-1) | [2. reduce](#ch-2) | [3. all_gather](#ch-3) | [4. reduce_scatter](#ch-4)
- [5. all_reduce](#ch-5) | [6. RS+AG combo](#ch-6) | [7. all_to_all](#ch-7)
- [8. Summary](#ch-8) | [9. NVLS Topic](#ch-9) | [10. Unverified](#ch-10) | [Reproduction](#ch-guide)

**Key Findings**:

| Question | Conclusion |
|---|---|
| Is the ring byte model accurate? | **Exact** — ring/tree/P2P all 1.000 (6 ops + combo); "AR = RS+AG" bytes and time both verified equivalent (§6) |
| What does all_reduce use by default? | **NVLS (NVLink SHARP)** — reduction in NVSwitch chip, per-GPU **1.125×S** (ring 1.75×S), measured/model = **0.643** (§5) |
| Is NVLS only worthwhile for all_reduce? | **Yes** (forced matrix, §9.3): AG saves 85% bytes but 38% slower; RS 18% slower; bcast/reduce unsupported; AR 17% faster, 36% fewer bytes |
| Default algorithms | **AR→NVLS; AG/RS→Ring; bcast/reduce→Ring (with dead edge); a2a→P2P direct** |
| a2a advantage under sparse routing | **bytes ∝ #targets**: k=1/2/4/7 → 2/4/8/14 GB, all 1.000; k=1 = **1/7 of all_gather** (§7⑤) |

<a id="ch-0"></a>
## 0. Methods & Commonality

[↑ Return](#toc)

### 0.1 Measurement Method (3-stage per script)

1. **Semantic correctness**: small tensor (4 elements), manually verifiable expected values;
2. **Timing**: large tensor (16 GB, decimal) × 5 iterations, CUDA events. **Note**: 5-iteration running average (includes steady-state pipelining); isolated single-run ~7% slower — bytes unaffected, time slightly optimistic;
3. **Traffic reconciliation**: per-GPU NVLink cumulative Tx (`nvidia-smi nvlink -gt d`) delta ÷ iterations. Measured over full iteration window (barrier-isolated), both timing and bytes from same batch.

Run: `torchrun --nproc_per_node=8 <op>.py`.

### 0.2 Measurement Discipline (lessons learned)

1. **One NCCL task per node at a time** — GPU and NVLink counters must be exclusive; concurrent debug tasks previously caused 2× byte inflation and 5× slowdown.
2. **Variable-length a2a: separate process per pattern** — same-process mode switching introduces deterministic excess traffic.
3. **Path consistency (verified)**: channel graphs fixed at communicator setup; 5 independent single-runs produce bitwise-identical bytes.
4. **Warmup 3 iterations**: sufficient for all ops (bitwise stable); exception is variable-length a2a mode switching.

### 0.3 Conventions

- **Units**: decimal — GB = 10⁹ B, GB/s = 10⁹ B/s; NVLink counter in KiB, converted.
- **Notation**: rank = GPU = 0–7 (unified throughout).
- **Size notation**: S = per-rank input bytes; C = per-rank block/segment bytes.

### 0.4 Ring Byte Model (reconciliation baseline)

| Operation | Per-GPU bytes | W=8 | Notes |
|---|---|---|---|
| broadcast / reduce | (W−1)/W × S | 0.875×S | S = message size |
| all_reduce | 2(W−1)/W × S | 1.75×S | = RS + AG |
| all_gather | (W−1) × C | 7×C | C = input block, output 8C |
| reduce_scatter | (W−1) × C | 7×C | C = output segment, input 8C |
| all_to_all (even) | (W−1)/W × S | 0.875×S | each rank sends S/8 to 7 peers |

The **(W−1) "−1" = "don't send to self"** — each rank's own portion is local, no wire bytes.

### 0.5 Algorithm Fingerprints & Structure

| Algorithm | Per-GPU bytes | Distribution shape |
|---|---|---|
| Ring (no dead edge: AG/RS) | = ring model | All uniform |
| **Ring (with dead edge: bcast/reduce)** | = ring model | **min ≈ 0 (dead-edge sender) / max = S** |
| Tree | mean = ring model | Non-uniform: near-root up to 2× |
| NVLS | **far below ring model** | Uniform |

**Ring dead-edge effect** (source of "chain-like" distribution for bcast/reduce): Ring is a **closed loop** (0→1→…→7→0, 8 edges), but different operation semantics create **dead edges** (edges that carry no data):
- **broadcast**: dead edge = **edge into src** (7→0) — src already has data, predecessor recv-only → distribution [S×7, 0] (GPU7 = src's predecessor, doesn't send).
- **reduce**: dead edge = **dst's outgoing edge** (0→1) — dst never sends → GPU0 = dst, doesn't send.
- **AG / RS**: **no dead edge** — every block/segment must reach all other ranks → all uniform.

Dead-edge position **changes with src/dst** — structural artifact, not design. **broadcast and reduce use the same Ring graph, only dead-edge position differs**.

**Ring algorithm rounds** (AG/RS/ring-AR shared; 4-GPU example):

```
Step 1: 0─A→1  1─B→2  2─C→3  3─D→0    (everyone sends own block)
Step 2: 0─D→1  1─A→2  2─B→3  3─C→0    (forward just-received block)
Step 3: 0─C→1  1─D→2  2─A→3  3─B→0
→ N-GPU ring: N−1 steps, everyone has all blocks
```

Everyone sends/receives 1 block per step (full-duplex, no idle). Each block travels exactly **N−1 hops**; N links all saturated simultaneously. Three guarantees: **topology** (N−1 hops on a ring visits all members), **dependency** (receive-before-forward ordering), **placement** (block from rank j → output slot j). Bytes: 1 block × (N−1) steps = **(N−1)×C** = ring model.

**reduce_scatter is the "reduce-while-forwarding" variant**: each step forwards **accumulating partial sums** (receive → add own contribution → forward). Each GPU **touches all segments**, contributing one addition per segment for N−1 segments; only the "final addition" lands back on own segment. Compared to a2a-style "direct to destination": same total computation, just **ring-spread vs a2a-concentrated**.

**Channel structure**: NCCL channels are **concurrent copies of the same logical graph**:
- **Fixed once**: channel graphs computed at communicator setup; channels only determine which fraction (e.g., 1/32) to carry — **change concurrency, not algorithm shape or bytes**.
- **Isomorphic (log-verified)**: 32 channels all isomorphic — ring order `0 1 2 3 4 5 6 7` for all. Dead edges cause min≈0/max=S distribution for bcast/reduce; AG/RS uniform.
- **Trees log role**: Tree graph is an **available resource** (for tree algorithms like AR); bcast/reduce **don't use tree** (source `tuning.cc:274` Ring-only) — they use Ring channels.
- **Full-mesh no contention**: NVLink pairwise NV18 direct; every Ring hop has dedicated link.

### 0.6 Bandwidth Metrics (algbw / busbw / actual line rate)

| Metric | Definition | Question answered |
|---|---|---|
| **algbw** | S_alg ÷ t | **Semantic layer**: effective data per second |
| **busbw** | Ring model per-GPU ÷ t (= algbw × ring factor) | **Normalized layer**: wire bytes/sec if ring-implemented |
| Actual line rate | Measured per-GPU Tx ÷ t | **Physical layer**: actual wire rate |

- **S_alg convention** (nccl-tests): AR/bcast/reduce/a2a use input S; AG uses output total (8C); RS uses input total (8C).
- **Why busbw**: algbw isn't cross-op comparable — same algbw, AR moves 1.75 units on wire vs AG 0.875; busbw normalizes all ops to one scale.
- **busbw is a normalized metric, not measured**: only equals actual line rate for no-dead-edge Ring. Ring-with-dead-edge (broadcast): busbw 595 **below** actual 680. NVLS (AR): busbw 842 **above** actual 541. **Deviation direction is itself an algorithm fingerprint**.

<a id="ch-1"></a>
## 1. broadcast

[↑ Return](#toc)

**Definition**: `out_r = x_src` (∀r, including r = src) — src's tensor copied to every rank.

**Principle**: Data split into multiple parts, **multiple isomorphic Rings concurrent** (default 32 channels); Ring-internal hop-by-hop forwarding, **dead edge 7→0 carries no data** (GPU7 recv-only), Ring-external parallel. Cost: 7 copies of S = 7S, avg 7/8 × S = 0.875×S.

**Analogy**: Speaker distributing handouts — star: speaker copies 7 times (bottleneck); single-line: 7-hop serial (bottleneck); NCCL: **multiple parallel relay lines** — line length no longer bottleneck.

**Code**: `sn-codes/broadcast.py`. Run: `torchrun --nproc_per_node=8 broadcast.py`.

**Results** (measured):
```
[Timing] 23.542 ms | algbw 680 GB/s | busbw (ring) 595 GB/s
[Traffic] NVLink: avg 14.0017 GB/GPU (min 0.0010 / max 16.0020)
[Reconciliation] Ring model 14.0000 → measured/model = 1.000
```

**Analysis**:
① 14.0017 = 0.875×S + 1.7 MB overhead ✓.
② **Algorithm = Ring (with dead edge; distribution + source code)**: min≈0, max=S, mean=0.875S. Eliminations: no-dead-edge Ring ❌ (requires uniform 14 GB); binary tree ❌ (max should be 2S=32); NVLS ❌ (should be S/8=2). **Ring with dead edge ✓** (dead edge = 7→0, GPU7 doesn't send). Source: `tuning.cc:274` broadcast **Ring-only**.
③ **32 isomorphic Rings** (channel scan): linear scaling to 16 channels (26 GB/s each); single channel preserves bytes/distribution. If ring orders rotated dead-edge positions, bytes would approach uniform (~15.5 GB) — measured min≈0 excludes.
④ 23.5 ms; per-GPU line rate ≈ **680 GB/s** = 32 channels × 26 GB/s. busbw 595 < actual 680 (dead-edge Ring sends S vs ring-normalized 7/8×S).

<a id="ch-2"></a>
## 2. reduce

[↑ Return](#toc)

**Definition**: `out_dst[i] = Σ x_r[i]` — only dst holds result; others' outputs undefined.

**Principle**: Isomorphic to broadcast (same Ring, dead edge moved) — data flows **downward** (GPU7 → GPU6 → … → GPU0), **dead edge = dst outgoing 0→1** (GPU0 = dst never sends). Each receiving rank adds "received partial sum + own input" and forwards (compute-while-forwarding).

**Analogy**: 8 branch offices with ledgers → total to headquarters. Chain relay: each passes "partial sum so far + own contribution" down the line, arriving complete at HQ.

**Code**: `sn-codes/reduce.py`.

**Results** (measured):
```
[Timing] 23.332 ms | algbw 686 GB/s | busbw 600 GB/s
[Traffic] avg 14.0017 GB/GPU (min 0.0010 / max 16.0020)
[Reconciliation] 1.000
```

**Analysis**: Same mirror signature as broadcast — GPU0 (dst) doesn't send. **Compute is free**: reduce-while-forwarding (0.25 FLOP/B arithmetic intensity) — with compute, not slower but slightly faster (23.3 vs 23.5 ms).

<a id="ch-3"></a>
## 3. all_gather

[↑ Return](#toc)

**Definition**: `out_r = concat(x_0, …, x_{W-1})` (∀r) — **concatenate, no summing**.

**Principle**: **Multiple same-order Rings** — each Ring carries one fraction. Each GPU's block circulates: every hop gains one more block; 7 rounds → everyone complete. Each GPU sends 7 copies of own block (relaying for others). Cost: 7 × C = (W−1)×C per GPU.

**Analogy**: Jigsaw puzzle — 8 people each with 1 piece; after circulation, everyone has all 8 (own stays, touches 7 others' pieces).

**Code**: `sn-codes/all_gather.py`.

**Results** (measured):
```
[Timing] 25.708 ms | algbw 622 GB/s | busbw 545 GB/s
[Traffic] avg 14.0009 GB/GPU (min = max = 14.0009)
[Reconciliation] 1.000
```

**Analysis**: ② **Algorithm = Ring (uniform distribution excludes tree/dead-edge Ring; bytes = model excludes NVLS)** — same communicator, same tree resources, but tuner chose Ring for AG: algorithm selection is **per-operation**. **Variable version `all_gatherv`**: unequal block sizes, model weighted accordingly.

<a id="ch-4"></a>
## 4. reduce_scatter

[↑ Return](#toc)

**Definition**: `out_r = (Σ x_k) segment r` — sum then split by segment, **each rank gets only its 1/W segment**.

**Principle**: **Multiple same-order Rings** — partial sums circulate, merge, and land on the segment owner. Each GPU touches all segments, contributing one addition each; only the "last addition" lands on own segment. Cost: 7 × C = (W−1)×C per GPU (same as AG, reversed).

**Analogy**: Segmented accounting — total not sent to anyone; each person takes only their responsible 1/8 segment (compute on segment, then take own).

**Code**: `sn-codes/reduce_scatter.py`.

**Results** (measured):
```
[Timing] 20.234 ms | algbw 791 GB/s | busbw 692 GB/s
[Traffic] avg 14.0009 GB/GPU (min = max)
[Reconciliation] 1.000
```

**Analysis**: ② Ring (same as AG). ⑤ **algbw 791 GB/s highest in project**: same 14 GB wire bytes carried 8× algorithm-level bytes (16 GB input, 2 GB output). **Variable version `reduce_scatterv`**: dual of `all_gatherv`.

<a id="ch-5"></a>
## 5. all_reduce

[↑ Return](#toc)

**Definition**: `out_r[i] = Σ x_k[i]` (∀r) — element-wise sum, **every rank holds complete result**. Semantics = reduce + broadcast; ring implementation = RS + AG.

**Principle (two paths)**:
- **Ring two-phase** (conceptual): RS phase (reduce-while-forwarding) → each GPU holds 1/8 of the sum; AG phase → everyone complete.
- **NVLS (actual default)** — reduction in **NVSwitch chip**:
  ```
  ① Upward reduce:  8 GPUs full S ──NVLink──▶ NVSwitch (in-chip element-wise sum 8→1)
  ② Downward multicast:  each head ──S/8──▶ NVSwitch ──one-to-all──▶ all 8 GPUs
  ```

  **Who computes, who transports**:

  | Step | Computer | Transporter | Bytes/GPU |
  |---|---|---|---|
  | ① Upward | **NVSwitch** (in-chip sum) | GPU→NVSwitch (NVLink) | +S |
  | ② Multicast | **none** (pure fanout) | head→NVSwitch→8 GPUs (NVLink) | +S/8 |

  Every GPU is a head — each multicasts 1/8 of result. Per-GPU = S + S/8 = **1.125×S**. Three consequences: **bytes** — saves 0.625×S vs ring's 1.75×S (the "forwarding for others" part); **compute** — GPU zero-summing; **structure** — independent of ring/tree channels (channel knob ineffective, ③). **Execution** — block-fused pipeline, no phase boundary.

**Analogy**: 8 people co-writing a ledger, everyone needs identical copy. Ring: "segment-compute (RS), then assemble (AG)"; NVLS: **hire an accountant (NVSwitch)** — everyone submits, accountant computes and reads result to all.

**Code**: `sn-codes/all_reduce.py`.

**Results** (measured):
```
[Timing] 33.265 ms | algbw 481 GB/s | busbw (ring) 842 GB/s
[Traffic] avg 18.0064 GB/GPU (min 18.0047 / max 18.0066)
[Reconciliation] Ring model 28.0000 → measured/model = 0.643
```

**Analysis**:
① Ring model 1.75×S = 28.0; measured **18.0064** (**0.643**) — only op significantly below model.
② **Algorithm = NVLS (3 evidences)**: uniform distribution eliminates tree/dead-edge Ring; **NVLS comm line unique to all_reduce**; byte decomposition = S + S/8 = 18.0 exact.
③ **Channel knob completely flat** (1→32 all 481 GB/s) — NVLS doesn't use ring/tree channels. Also independent proof: if ring, 1-channel should collapse to ~30 GB/s.
④ 33.265 ms; per-GPU actual 18.0/33.3 ≈ **541 GB/s**; busbw 842 **above** actual — NVLS "ring-equivalent" fingerprint. **NVLS vs forced Ring**: 33.3 vs 40.2 ms (17% faster), 18.0 vs 28.0 GB (36% fewer).

<a id="ch-6"></a>
## 6. reduce_scatter + all_gather Combo (= all_reduce)

[↑ Return](#toc)

**Definition**: RS then AG — semantically and byte-wise **identical to all_reduce**.

**Principle**: Manual two-step decomposition of AR. Each step independently ring-parallel.

**Analogy**: AR's "two-step" manual version: segmented accounting (RS), then jigsaw assembly (AG). One call = all_reduce; two calls = RS+AG.

**Code**: `sn-codes/rs_ag.py`.

**Results** (measured):
```
[Timing] 45.949 ms | algbw 348 GB/s | busbw 609 GB/s
[Traffic] avg 28.0017 GB/GPU (min = max)
[Reconciliation] 1.000
```

**Analysis**: ① 28.0017 = RS(14.0009) + AG(14.0009) **exactly additive**. ② Ring + Ring (no NVLS comm — **manual two-step gets no NVLS**). ③ Channels approx. additive at all levels. ⑤ **Three prices for same semantics**: **NVLS single-call 33.3 < Ring single-call 40.2 < manual two-call 45.9 ms** — call boundary costs (+14%), algorithm choice costs more (+17%) — this is NCCL's value: same calling convention, different path chosen by the library.

<a id="ch-7"></a>
## 7. all_to_all

[↑ Return](#toc)

**Definition**: `out_r segment j = x_j segment r` — each rank's input split into W segments, **segment j sent only to rank j** (W×W matrix transpose).

**Principle**: **Point-to-point direct** — each rank opens W−1 = 7 simultaneous direct pairs; segment j goes directly to GPU j. No ring relay, no forwarding for others. Cost: 7 × S/8 = (W−1)/W × S per GPU — same as AG for balanced load; advantage in sparse/skewed.

**Analogy**: Dealing cards — 8 players each with 8 stacks, stack j only to player j. Essential difference from AG: **only send what's needed**, not everyone's full copy.

**Code**: `sn-codes/all_to_all.py`.

**Results** (measured):
```
[Timing] 20.960 ms | algbw 763 GB/s | busbw 668 GB/s
[Traffic] avg 14.0017 GB/GPU (min = max)
[Reconciliation] 1.000
```

**Analysis**: ② P2P direct (ring/tree **semantically invalid** for transpose — each block has one destination, no relay path). ⑤ a2a faster than AG (763 vs 622 algbw): same 14 GB/GPU but "7 pairs each 1-hop direct" vs "1 block 7-hop relay through everyone".

**⑤ Sparse routing advantage (k-target scan, independent process per k)**: variable a2a (`all_to_all_single` split_sizes, `a2a_sparse.py <k>`), each GPU sends 2 GB/block to only k targets — MoE dispatch abstraction:

| Targets k | 1 | 2 | 4 | 7 (=even split) |
|---|---|---|---|---|
| Per-GPU measured | **2.0002 GB** | 4.0005 | 8.0010 | 14.0017 |
| Model k×C | 2 | 4 | 8 | 14 |
| Ratio | **1.000** | **1.000** | **1.000** | **1.000** |
| Time/run | 3.582 ms | 7.102 | 13.280 | 21.122 |

**Bytes scale linearly with target count, exact at every point** — traffic ∝ **effective flow**, not AG's fixed 7×C. k=1: a2a sends only 2 GB = **1/7 of AG** — this is the mechanism of a2a's overwhelming advantage in sparse routing (MoE) scenarios.

**Methodological finding**: variable a2a (with 0 blocks) after running other patterns in same process carries **deterministic excess traffic** (k=1 up to 1.38× model); **protocol: separate process per pattern, first window**.

<a id="ch-8"></a>
## 8. Summary: Seven Operations One Table

[↑ Return](#toc)

| Op | Input scale | Time/run | algbw | busbw | Measured Tx/GPU | Ring model | Meas/model | Distribution | Default algorithm |
|---|---|---|---|---|---|---|---|---|---|
| broadcast | src 16 GB | 23.542 ms | 680 | 595 | 14.0017 GB | 14.0000 | 1.000 | **min≈0 / max=S** | Ring (dead edge) |
| reduce | 16 GB/rank | 23.332 ms | 686 | 600 | 14.0017 | 14.0000 | 1.000 | **min≈0 / max=S** | Ring (dead edge) |
| all_gather | 2→16 GB | 25.708 ms | 622 | 545 | 14.0009 | 14.0000 | 1.000 | Uniform | Ring |
| reduce_scatter | 16→2 GB | 20.234 ms | **791** | 692 | 14.0009 | 14.0000 | 1.000 | Uniform | Ring |
| all_reduce | 16 GB/rank | 33.265 ms | 481 | **842** | **18.0064** | 28.0000 | **0.643** | Uniform (1.125×S) | **NVLS** |
| rs_ag | 16 GB | 45.949 ms | 348 | 609 | 28.0017 | 28.0000 | 1.000 | Uniform | Ring + Ring |
| all_to_all | 16 GB/rank | 20.960 ms | 763 | 668 | 14.0017 | 14.0000 | 1.000 | Uniform | P2P direct |

- **busbw ranking**: **AR 842** (NVLS 2×) > RS 692 > a2a 668 > reduce 600 ≈ bcast 595 > rs_ag 609 > AG 545 — **busbw vs actual line rate deviation direction is itself an algorithm fingerprint**.
- Ring model: 6 ops all 1.000; **only exception AR's NVLS (0.643)**.
- Trees log identical across all ops (communicator-level resource) — bcast/reduce **don't use tree** (source: Ring-only); AR unique NVLS comm line.
- AR (heaviest semantics: ring = RS+AG) via NVLS: only 29% more than AG/RS's 0.875×S.
- Channel scans: Ring/P2P 26–37 GB/s/channel, linear ≤16, sub-linear 16→32; **AR (NVLS) completely unresponsive to channel knob**.
- Sparse routing (§7⑤): a2a bytes ∝ target count.

<a id="ch-9"></a>
## 9. NVLS Topic: Concept, Principle & Application

[↑ Return](#toc)

### 9.1 Concept

**NVLS (NVLink SHARP)**: Moves reduction/multicast from GPU into **NVSwitch chip** — GPUs don't forward each other or sum locally; the switch does it.
- **Lineage** (inferred): From InfiniBand SHARP (in-switch reduction), ported to NVSwitch.
- **Hardware requirement**: NVSwitch multimem primitives. Log: `NVLS multicast support is available`.
- **Resources**: NVLS comm established **only for all_reduce by default**; log `headRank r, nHeads 8, nvlsPerRankSize = total/8`.

### 9.2 Principle (general model, 16 GB measured)

```
Upward (reduce):  all GPUs ──full S──▶ NVSwitch (in-chip element-wise reduce)
Downward (multicast): "head" GPU ──one multicast write──▶ switch fans out to all
```

**Who computes, who transports** (same §5 table):

| Step | Computer | Transporter | Bytes/GPU |
|---|---|---|---|
| ① Upward | **NVSwitch** | GPU→NVSwitch (NVLink) | +S |
| ② Multicast | **none** (fanout) | head→NVSwitch→8 GPUs (NVLink) | +S/8 |

**Byte model** (per GPU):

| Op | NVLS measured | Model | vs Ring |
|---|---|---|---|
| all_reduce | 18.01 GB | **S + S/W** | 28.0 GB (saves 36%) |
| all_gather | 2.04 GB | **≈C** | 14.0 GB (saves 85%) |
| reduce_scatter | 16.04 GB | **≈S** | 14.0 GB (+15%) |

Three general properties: **compute** moved to switch (GPU zero-sum); **structure** independent of ring/tree channels; **execution** block-fused pipeline, no phase boundary.

### 9.3 Application Matrix (16 GB: default vs forced)

| Op | Default | Forced Tree | Forced NVLS | NVLS bytes/time | Verdict |
|---|---|---|---|---|---|
| broadcast | Ring (dead edge) / 23.5 ms | ✗ no Tree in candidates | ❌ error | — | N/A |
| reduce | Ring (dead edge) / 23.3 ms | ✗ | ❌ error | — | N/A |
| all_reduce | **NVLS** / 33.3 ms | 49.3 ms (28.0 GB) | ✓ 33.3 ms | 18.0 GB | **Net win**: 17% faster, 36% fewer |
| all_gather | Ring / 25.7 ms | ✗ | ✓ 35.5 ms | 2.0 GB | Saves 85% bytes but **38% slower** |
| reduce_scatter | Ring / 20.2 ms | ✗ | ✓ 23.9 ms | 16.0 GB | +15% bytes and **18% slower** |

- **AR decomposition self-consistent**: NVLS-AR(18.015) ≈ NVLS-RS(16.037) + NVLS-AG(2.041), 0.3% — "upward reduce + multicast write" two-segment model directly additive.
- **Byte savings ≠ time savings**: NVLS-AG has 1/7 the bytes but 38% slower — switch multicast throughput below NVLink ring. Only AR (merged semantics) wins — **this is why the tuner only enables NVLS for AR by default**.
- **Algorithm candidate sets (source `tuning.cc:274-278`)**: bcast/reduce → **Ring only**; AG/RS → {PAT, Ring, NVLS, CollNet} (no Tree); AR → {Ring, Tree, NVLS, CollNet}. Tree-AR forced: 49.3 ms, slowest of three (NVLS 33.3 < Ring 40.2 < Tree 49.3).

### 9.4 Boundaries & Costs

- **Hardware threshold**: Requires NVSwitch multimem.
- **Channel independence**: Channel knob unresponsive.
- **Single-call integral**: Semantically equivalent manual combos get no NVLS (§6: rs_ag 38% slower than NVLS).
- **Size boundary**: Not tested — small messages may have NVLS fixed overhead penalty.

<a id="ch-10"></a>
## 10. Unverified Items

[↑ Return](#toc)

- NVLS 1.125×S residual (16 GB: 18.0064 vs 18.0 = +0.04%).
- NVLS vs Ring size crossover point (small messages); NVLS fallback without NVSwitch untested.
- ~~Ring/Tree forcing for bcast/reduce~~ (solved: `tuning.cc:274` Ring-only; "chain" = Ring dead-edge effect).
- a2a P2P direct inferred from bytes (per-channel composition unverified).
- Variable a2a mode-switching excess traffic mechanism (§7⑤).
- broadcast vs reduce per-channel rate asymmetry (26 vs 35 GB/s single-channel).
- Running-average vs isolated-single ~7% difference only quantified for broadcast.
- Warmup 3 sufficiency; no multi-run variance statistics performed.

<a id="ch-guide"></a>
## Reproduction Guide

[↑ Return](#toc)

**Minimal reproduction** (§1–7 baseline data):

1. Deploy: `kubectl apply -f 8-gpu-8-fabric-pod.yaml` (single-node: pod-0 only);
2. Copy scripts: `kubectl cp sn-codes 8-gpu-8-fabric-pod-0:/tmp/ -c server`;
3. Per-op run: `kubectl exec ... bash -c "cd /tmp/sn-codes && torchrun --nproc_per_node=8 <op>.py"`;
4. Sparse scan: `for k in 1 2 4 7; do torchrun --nproc_per_node=8 a2a_sparse.py $k; done` — **separate process per k** (§0.2).

**Extended experiments**:

- **Channel scan** (§③ data): `kubectl exec -i ... bash -s < scan_channels.sh` (6 ops × channels 1/2/4/8/16/32).
- **Algorithm matrix** (§9.3): `NCCL_ALGO=NVLS torchrun ... <op>.py` (Tree similarly; unsupported = ❌ in matrix).
- **Evidence capture**: `NCCL_DEBUG=INFO` (grep `NVLS comm|Trees|Channel .* via`); per-process: `NCCL_DEBUG_FILE=/tmp/dbg/nccl.%p`.
- **Single-run verification**: `torchrun --nproc_per_node=8 nccl_bench.py --op <op> --mib <N> --iters 1` (bytes should match per-run; timing ~7% slower than running average).

**Notes**: One NCCL task per node at a time (§0.2); Pod rebuild requires re-copy; NVLink counters are cumulative node values, take deltas.

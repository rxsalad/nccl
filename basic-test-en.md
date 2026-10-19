# Multi-Node Environment Baseline Test: 4 Pods × 8 GPUs × 8 RoCE Rails (Single Plane)

**Date**: 2026-10-18 (revised 10-19: added dual-plane architecture note); **Target**: StatefulSet `8-gpu-8-fabric-pod` (replicas: 4, see `nccl-mn/8-gpu-8-fabric-pod.yaml`); **Method**: per-Pod inventory (`nvidia-smi` / `ibv_devinfo` / `ibstat` / `rdma link` / `ip`) + full-rail connectivity matrix (`ping -6`, 2 packets per rail, average RTT). Convention: **measured** = output from this cluster; **inferred** = derived from mechanism or estimation.

## 1. Overview

| Pod | Node | eth0 IP | Node ID (4th address segment) |
|---|---|---|---|
| pod-0 | rs-b300x8-gpu-pool-3jv0ev | 10.154.6.54 | 52e |
| pod-1 | rs-b300x8-gpu-pool-3jvdpn | 10.154.5.10 | 521 |
| pod-2 | rs-b300x8-gpu-pool-3jvd29 | 10.154.4.210 | 527 |
| pod-3 | rs-b300x8-gpu-pool-3jvd2z | 10.154.5.132 | 520 |

4 Pods across 4 GPU nodes (one Pod per node).

## 2. Per-Pod Inventory (4/4 identical, measured)

| Item | Result |
|---|---|
| GPU | **8× B300 SXM6 PC** (`nvidia-smi -L`), pairwise NV18 |
| RDMA devices (verbs visible) | **8× mlx5_{0,2,…,14}**, all **Active, 400 Gb/s** (`ibstat`) |
| RDMA NICs | **8× fabric_{0,2,…,14}**, each with global IPv6 address |
| Device ↔ NIC | `rdma link` verified 1:1: **mlx5_2i ↔ fabric 2i** |
| VRF | Each NIC has independent VRF: fabric2i → vrf(100+2i) (verified across 4 Pods) |

**Device view layering (measured)**: `rdma link` (netlink, node-level) shows 20 links — this Pod's 8 + odd-numbered mlx5_{1,3,…,15} (no netdev — **other dual-plane**, see below) + mlx5_{16..19} (with IB subnet/lid, not this Pod's); `ibv_devinfo` (verbs, Pod-level) shows only this Pod's 8 — **verbs access is Pod-isolated**.

### Dual-Plane Architecture (Critical Context)

Each node provides **16 RDMA NICs** (even-numbered 0/2/…/14 in one plane, odd-numbered 1/3/…/15 in another); **inter-plane isolation is by design**. This experiment passes only the **even plane's 8 NICs** to each Pod (YAML requests only `rdma/fabric0,2,…,14`). Therefore:

- This Pod's 8 rails are **all in the same plane** — all connectivity conclusions (§5/§6) hold within a single plane (256/256 full connectivity does not contradict plane isolation: no cross-plane pair exists in the test set);
- Odd-numbered mlx5 visible in `rdma link` are the **other plane's NICs** on the node (not allocated to this Pod, hence no netdev);
- Capacity note: this experiment uses **single-plane 8 rails** (400 Gb/s each); full node supply is dual-plane 16 rails — all subsequent bandwidth results are single-plane.

(Platform acceptance document `doks/doks-test-cn.md` (same cluster, mkc1) §1.3/1.4 tested full 16×16 matrix: same-parity reachable, cross-parity unreachable — consistent with our observations.)

## 3. Addressing Scheme (measured, generalized)

Each fabric NIC's global address: `fd02:0:<rail>:<node>::1/64` — 3rd segment identifies rail (fabric0→400, fabric2→500, …, fabric14→0), 4th segment identifies node (see §1). Same rail across nodes = different /64 (routed, not L2 same-subnet).

Full address table per Pod (pod-0 example, others same pattern):

| fabric | IPv6 address | VRF | RDMA device |
|---|---|---|---|
| fabric0 | fd02:0:400:52e::1/64 | vrf100 | mlx5_0 |
| fabric2 | fd02:0:500:52e::1/64 | vrf102 | mlx5_2 |
| fabric4 | fd02:0:600:52e::1/64 | vrf104 | mlx5_4 |
| fabric6 | fd02:0:700:52e::1/64 | vrf106 | mlx5_6 |
| fabric8 | fd02:0:300:52e::1/64 | vrf108 | mlx5_8 |
| fabric10 | fd02:0:200:52e::1/64 | vrf110 | mlx5_10 |
| fabric12 | fd02:0:100:52e::1/64 | vrf112 | mlx5_12 |
| fabric14 | fd02:0:0:52e::1/64 | vrf114 | mlx5_14 |

## 4. GPU ↔ Rail Affinity (`nvidia-smi topo -m`, measured)

- **Core fact**: **GPU i ↔ mlx5_2i ↔ fabric 2i** — each GPU has one dedicated rail (YAML declaration and topo PXB affinity cross-verified).
- **PXB peers**: Each GPU's two PXB (same PCIe bridge) NICs — exactly one per plane (GPU i ↔ NIC 2i even plane, NIC 2i+1 odd plane) — dual-plane design gives each GPU a nearest exit in both planes (this experiment uses even side only).

## 5. Connectivity Matrix (pod-0 → 3 Pods × 8 rails, measured)

**24/24 fully connected**, uniform RTT (0.099–0.132 ms, mean ≈0.12 ms) — all rails × all node pairs equivalent (spine fabric symmetry, inferred), providing "any rail to any peer at same cost" premise for multi-node experiments.

| Target | fabric0 | fabric2 | fabric4 | fabric6 | fabric8 | fabric10 | fabric12 | fabric14 |
|---|---|---|---|---|---|---|---|---|
| → pod-1 | 0.110 | 0.132 | 0.131 | 0.116 | 0.128 | 0.123 | 0.099 | 0.125 |
| → pod-2 | 0.125 | 0.127 | 0.125 | 0.111 | 0.130 | 0.117 | 0.121 | 0.111 |
| → pod-3 | 0.132 | 0.101 | 0.119 | 0.118 | 0.128 | 0.109 | 0.126 | 0.109 |

## 6. Cross-Rail 8×8 Full-Combination Connectivity (measured)

**Design**: Group A = {pod-0, pod-1}, Group B = {pod-2, pod-3}; inter-group, each Pod pair tested **8 source fabrics × 8 target fabrics** full combination (`ping -6 -c 1 -I <source> <target addr>`), 4 Pod pairs × 64 = **256 tests** — verifying spine fabric's cross-rail routing (§5 tested same-rail diagonal only).

**Result: 256/256 fully connected.**

| Pod pair | Pass/Total | RTT range |
|---|---|---|
| pod-0 → pod-2 | 64/64 | 0.073–0.158 ms |
| pod-0 → pod-3 | 64/64 | 0.064–0.143 ms |
| pod-1 → pod-2 | 64/64 | 0.072–0.146 ms |
| pod-1 → pod-3 | 64/64 | 0.073–0.122 ms |

Analysis:

- **Plane premise**: This Pod's 8 rails are all in the same plane (even-numbered, see §2) — all 256 combinations are intra-plane.
- **Cross-rail full connectivity (intra-plane)**: Any source rail → any target rail reachable — spine fabric provides cross-rail routing; rail-optimized affects **preferred path** only, not reachability.
- **No cross-rail latency penalty**: Diagonal (same-rail) vs off-diagonal (cross-rail) RTT shows no significant difference (mean difference <0.01 ms, within noise).
- Minor observation: RTT has systematic micro-variation by source fabric (fabric0 row slightly high ≈0.13 ms, fabric12/14 rows slightly low ≈0.08 ms), magnitude <0.05 ms, likely related to ping process CPU/NUMA affinity, does not affect conclusions.

## 7. Conclusions

| Check item | Result |
|---|---|
| **Dual-plane: node 16 NICs, this Pod allocated even plane 8 only** | ✓ |
| 4 Pods / 4 nodes, each 8 GPUs + 8 RDMA devices + 8 NICs | ✓ (4/4 identical) |
| Each NIC independent VRF | ✓ |
| Device↔NIC↔GPU 1:1 mapping (mlx5_2i ↔ fabric 2i ↔ GPU i) | ✓ |
| verbs Pod-isolated | ✓ |
| Full rail × full peer connectivity, RTT ≈0.12 ms uniform | ✓ (24/24) |
| Cross-rail communication (8×8 full combination) | ✓ (256/256, same/cross-rail RTT no difference; all intra-plane) |

**Environment ready.** All prerequisites met for 2-node (`nccl-2n`) / 4-node (`nccl-4n`) experiments — note bandwidth figures are **single-plane 8-rail** (full node supply is dual-plane 16 rails).

## Appendix: Inventory Method (reproducible)

```bash
# Per-Pod inventory (kubectl exec -i ... bash -s < env_inventory.sh)
nvidia-smi -L                              # GPU list
ibv_devinfo -l                             # verbs device list
ibstat <dev>                               # Port State/Rate
rdma link                                  # Device ↔ netdev mapping
ip -6 addr show <fabricN>                  # NIC IPv6
ip -d link show <fabricN>                  # Master VRF
nvidia-smi topo -m                         # GPU-NIC affinity
# Connectivity matrix (from pod-0; target addresses per §3 scheme)
ping -6 -c 2 -I fabric0 fd02:0:400:521::1
```

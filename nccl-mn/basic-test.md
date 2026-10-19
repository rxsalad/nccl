# 多节点环境基线测试：4 Pod × 8 GPU × 8 RoCE rail（单平面）

**日期**：2026-10-18（10-19 修订：补双平面架构注记）；**对象**：StatefulSet `8-gpu-8-fabric-pod`（replicas: 4，部署见 `8-gpu-8-fabric-pod.yaml`）；**方法**：逐 Pod 清点（`nvidia-smi` / `ibv_devinfo` / `ibstat` / `rdma link` / `ip`）+ 全 rail 连通性矩阵（`ping -6`，每条 rail 2 包取平均 RTT）。标注约定：**实测**＝本集群命令输出；**推断**＝由机制或估算得出。

## 1. 总览

| Pod | 节点 | eth0 IP | 节点 ID（地址第 4 段） |
|---|---|---|---|
| pod-0 | rs-b300x8-gpu-pool-3jv0ev | 10.154.6.54 | 52e |
| pod-1 | rs-b300x8-gpu-pool-3jvdpn | 10.154.5.10 | 521 |
| pod-2 | rs-b300x8-gpu-pool-3jvd29 | 10.154.4.210 | 527 |
| pod-3 | rs-b300x8-gpu-pool-3jvd2z | 10.154.5.132 | 520 |

4 Pod 分属 4 个 GPU 节点（一 Pod 独占一节点）。

## 2. 每 Pod 清点（4/4 一致，实测）

| 项 | 结果 |
|---|---|
| GPU | **8× B300 SXM6 PC**（`nvidia-smi -L`），两两 NV18 |
| RDMA 设备（verbs 可见） | **8× mlx5_{0,2,…,14}**，全部 **Active、400 Gb/s**（`ibstat`） |
| RDMA 网卡 | **8× fabric_{0,2,…,14}**，各持全局 IPv6 地址 |
| 设备 ↔ 网卡 | `rdma link` 实测一一对应：**mlx5_2i ↔ fabric 2i** |
| VRF | 每网卡独立 VRF：fabric2i → vrf(100+2i)（4 Pod 逐卡核对一致） |

**设备视图分层（实测）**：`rdma link`（netlink，节点级）可见 20 条链路——本 Pod 的 8 条 + 奇数号 mlx5_{1,3,…,15}（无 netdev——双平面另一平面，见下）+ mlx5_{16..19}（带 IB 子网/lid，非本 Pod 附件）；`ibv_devinfo`（verbs，Pod 级）只见本 Pod 的 8 个——**verbs 已按 Pod 隔离**。

### 双平面架构（重要前提）

节点提供 **16 张 RDMA NIC**（偶数号 0/2/…/14 一平面、奇数号 1/3/…/15 另一平面），**平面间不互通是设计行为**；本实验只给每 Pod 传递了**偶数平面的 8 张**（yaml 只申请 `rdma/fabric0,2,…,14`）。因此：

- 本 Pod 的 8 条 rail **全部同平面**——§5/§6 一切连通性结论都在同平面内成立（256/256 全通不与平面隔离矛盾：测试里没有跨平面对）；
- `rdma link` 里可见的奇数号 mlx5 是节点上另一平面的网卡（未分配给本 Pod，故无 netdev）；
- 容量口径：本实验为**单平面 8 rail**（每 rail 400 Gb/s）；节点完整供给为双平面 16 rail——后续所有带宽结论均为单平面口径。

（平台侧验收文档 `doks/doks-test-cn.md`（同集群 mkc1）1.3/1.4 节实测 16×16 全矩阵：同奇偶互通、跨奇偶不通——与本文观察吻合。）

## 3. 寻址规律（实测归纳）

每条 fabric 网卡的全局地址：`fd02:0:<rail>:<节点>::1/64`——第 3 段标识 rail（fabric0→400、fabric2→500、…、fabric14→0），第 4 段标识节点（见 §1 节点 ID 列）。同 rail 跨节点为不同 /64（路由互通、非同网段 L2）。

每 Pod 完整地址表（pod-0 为例，其余同型）：

| fabric | IPv6 地址 | VRF | RDMA 设备 |
|---|---|---|---|
| fabric0 | fd02:0:400:52e::1/64 | vrf100 | mlx5_0 |
| fabric2 | fd02:0:500:52e::1/64 | vrf102 | mlx5_2 |
| fabric4 | fd02:0:600:52e::1/64 | vrf104 | mlx5_4 |
| fabric6 | fd02:0:700:52e::1/64 | vrf106 | mlx5_6 |
| fabric8 | fd02:0:300:52e::1/64 | vrf108 | mlx5_8 |
| fabric10 | fd02:0:200:52e::1/64 | vrf110 | mlx5_10 |
| fabric12 | fd02:0:100:52e::1/64 | vrf112 | mlx5_12 |
| fabric14 | fd02:0:0:52e::1/64 | vrf114 | mlx5_14 |

## 4. GPU ↔ rail 亲和（`nvidia-smi topo -m`，实测）

- **核心事实**：**GPU i ↔ mlx5_2i ↔ fabric 2i**——每 GPU 一条专属 rail（yaml 声明与 topo PXB 亲和互证）。
- **PXB 伙伴**：每 GPU 的两条 PXB（同 PCIe 桥）NIC 恰好每平面一条（GPU i ↔ NIC 2i 偶平面、NIC 2i+1 奇平面）——双平面设计下每 GPU 在两个平面各有一条就近出口（本实验只用偶平面侧）。

## 5. 连通性矩阵（pod-0 → 3 Pod × 8 rail，实测）

**24/24 全通**，RTT 均匀（0.099–0.132 ms，均值 ≈0.12 ms）——全部 rail × 全部节点对等价（脊状 fabric 的对称性，推断），为多节点实验提供"任一 rail 任一对端同价"前提。

| 目标 | fabric0 | fabric2 | fabric4 | fabric6 | fabric8 | fabric10 | fabric12 | fabric14 |
|---|---|---|---|---|---|---|---|---|
| → pod-1 | 0.110 | 0.132 | 0.131 | 0.116 | 0.128 | 0.123 | 0.099 | 0.125 |
| → pod-2 | 0.125 | 0.127 | 0.125 | 0.111 | 0.130 | 0.117 | 0.121 | 0.111 |
| → pod-3 | 0.132 | 0.101 | 0.119 | 0.118 | 0.128 | 0.109 | 0.126 | 0.109 |

## 6. 跨 rail 8×8 全组合连通性（实测）

**设计**：组 A = {pod-0, pod-1}、组 B = {pod-2, pod-3}；组间每 Pod 对做 **8 源 fabric × 8 目标 fabric** 全组合 ping（`ping -6 -c 1 -I <源> <目标地址>`），4 Pod 对 × 64 = **256 条**——验证脊状 fabric 的跨 rail 路由能力（§5 只测了同 rail 对角线）。

**结果：256/256 全通。**

| Pod 对 | 通/总 | RTT 范围 |
|---|---|---|
| pod-0 → pod-2 | 64/64 | 0.073–0.158 ms |
| pod-0 → pod-3 | 64/64 | 0.064–0.143 ms |
| pod-1 → pod-2 | 64/64 | 0.072–0.146 ms |
| pod-1 → pod-3 | 64/64 | 0.073–0.122 ms |

分析：

- **平面前提**：本 Pod 8 条 rail 全部同平面（偶数号，见 §2）——本节 256 组合全在同平面内。
- **跨 rail 全通（同平面内）**：任意源 rail → 任意目标 rail 皆可达——脊状 fabric 提供跨 rail 路由，rail-optimized 只影响**优选路径**、不限制可达性。
- **跨 rail 无延迟惩罚**：对角线（同 rail）与非对角线（跨 rail）RTT 无显著差异（均值差 <0.01 ms，噪声内）。
- 次要观察：RTT 随源 fabric 有系统性微差（fabric0 行略高 ≈0.13 ms、fabric12/14 行略低 ≈0.08 ms），量级 <0.05 ms，可能与 ping 进程 CPU/NUMA 亲和有关，不影响结论。

## 7. 结论

| 检查项 | 结果 |
|---|---|
| **双平面：节点 16 NIC，本 Pod 只分配偶数平面 8 张** | ✓ |
| 4 Pod / 4 节点，各 8 GPU + 8 RDMA 设备 + 8 网卡 | ✓（4/4 一致） |
| 每网卡独立 VRF | ✓ |
| 设备↔网卡↔GPU 一一对应（mlx5_2i ↔ fabric 2i ↔ GPU i） | ✓ |
| verbs 按 Pod 隔离 | ✓ |
| 全 rail × 全对端连通，RTT ≈0.12 ms 均匀 | ✓（24/24） |
| 跨 rail 通信（8×8 全组合） | ✓（256/256，同/跨 rail RTT 无差异；均为同平面内） |

**环境就绪。** 满足 2 节点（nccl-2n）/ 4 节点（nccl-4n）实验的全部前提——注意带宽口径为**单平面 8 rail**（节点完整供给是双平面 16 rail）。

## 附录：清点方法（可复现）

```bash
# 逐 Pod 清点（kubectl exec -i ... bash -s < env_inventory.sh）
nvidia-smi -L                              # GPU 清单
ibv_devinfo -l                             # verbs 设备清单
ibstat <dev>                               # 端口 State/Rate
rdma link                                  # 设备 ↔ netdev
ip -6 addr show <fabricN>                  # 网卡 IPv6
ip -d link show <fabricN>                  # master VRF
nvidia-smi topo -m                         # GPU-NIC 亲和
# 连通性矩阵（pod-0 内；目标地址按 §3 规律生成）
ping -6 -c 2 -I fabric0 fd02:0:400:521::1
```

# 多机实验（GPU 显存版）：经 RDMA（RoCE）的 GPU↔GPU 通信

两台机器上各有一张 GPU：server-0 的 GPU0 与 server-1 的 GPU7。本篇用两个裸 C/CUDA 程序，让它们**经 RDMA（RoCE v2）直接交换显存里的数据**：

- `server-gpu.cu`：运行在 server-1，使用 GPU7，**被动方**；
- `client-gpu.cu`：运行在 server-0，使用 GPU0，**主动方**，发起 RDMA WRITE / READ、校验、测带宽；
- **OOB（带外）通道**：两端经 `eth0` 上的 TCP，只交换控制信息；
- **数据通道**：RoCE v2 的 RDMA，网卡经 GPUDirect RDMA 直接读写 GPU 显存。

> **请先读 [c-multi-node-cpu.md](c-multi-node-cpu.md)**。那一篇用同样的协议、同一对网卡，让两台机器经 RDMA 读写**主机内存**，并讲清了 RDMA 的通用原理：对象（QP / MR / rkey）、OOB 交换什么、"连接"指什么、QP 状态机、WRITE / READ、流量类别 `-t`。本篇**不重复**这些，而是从 **§1 原理**开始，先讲清"**CPU 内存与 GPU 显存的 RDMA 有什么不同**、**GPU RDMA 有哪几种实现方式**"，再讲两个 GPU 程序与实测结果。

单机内的 NVLink 通信见 [c-single.md](c-single.md)；本篇是它的多机续篇。

**状态**：定稿（2026-11-01 封稿，已合入注册专题与产业动态；实验数据于 2026-10-31 与 2026-11-01 在 server-0 / server-1 上测得）。文中所有"实测"都按当前源码得到，每个结论的可靠程度见附录 A，已知局限见附录 B。

<a id="how"></a>
## 怎么读

| 你想…… | 读哪里 |
|---|---|
| 先看结论 | 「摘要」 |
| 理解"注册"是什么、CPU 内存与 GPU 显存的注册有何不同、GPU RDMA 有哪几种实现（peermem / dmabuf） | **§1** |
| 搞清两台机器、与主机内存版的关系、实验流程 | §2 |
| 看程序里 GPU 特有的做法、为什么要选就近的网卡 | §3 |
| 看两个程序怎么写 | §4 |
| 编译、运行、复现实验、看实测结果 | §5 |
| 看 GPU 显存与主机内存（CPU）的 RDMA 对比 | §5.7 |
| 复现注册耗时的探查实验 | §5.8 |
| 命令速查、避坑 | §6 |
| 想知道每个结论有多可靠 | 附录 A |

<a id="toc"></a>
## 目录

- [摘要](#sum)
- [读完应该能回答](#intro)
- [约定与术语](#conv)
- **[1. 原理：CPU 与 GPU 的 RDMA 有什么不同](#1)**：[1.1 为什么要注册](#1-1)、[1.2 CPU 内存怎么注册](#1-2)、[1.3 GPU 显存为什么要走另一条路](#1-3)、[1.4 实现方式一：nvidia_peermem](#1-4)、[1.5 实现方式二：dmabuf](#1-5)、[1.6 VMM 分配的显存](#1-6)、[1.7 BAR1 与页粒度](#1-7)、[1.8 生命周期与一致性](#1-8)、[1.9 对照总表](#1-9)、[1.10 编程：怎么写、怎么选](#1-10)、[1.11 产业动态与趋势](#1-11)
- **[2. 全貌](#2)**：[2.1 两台机器与环境](#2-1)、[2.2 与主机内存版的关系](#2-2)、[2.3 运行前提：GPU 版额外需要什么](#2-3)、[2.4 一次实验的时间线](#2-4)、[2.5 与单机 CUDA IPC 的对照](#2-5)
- **[3. 程序里 GPU 特有的做法](#3)**：[3.1 注册显存：程序里怎么写](#3-1)、[3.2 GPU 与网卡的就近关系](#3-2)
- **[4. 程序：server-gpu.cu 与 client-gpu.cu](#4)**：[4.1 总览与命令行参数](#4-1)、[4.2 显存布局与数据模式](#4-2)、[4.3 关键代码骨架](#4-3)
- **[5. 实验与实测](#5)**：[5.1 准备与编译](#5-1)、[5.2 运行](#5-2)、[5.3 正确性](#5-3)、[5.4 带宽](#5-4)、[5.5 数据走了哪条路：字节计数器](#5-5)、[5.6 与单机 NVLink 的对照](#5-6)、[5.7 与主机内存（CPU）版的对照](#5-7)、[5.8 注册探查实验 reg_bench](#5-8)
- **[6. 速查：命令与坑](#6)**
- **[附录 A. 验证状态汇总](#A)**、**[附录 B. 已知局限与后续工作](#B)**

前置与相关：[c-multi-node-cpu.md](c-multi-node-cpu.md)（主机内存版，含 RDMA 通用原理）、[c-single.md](c-single.md)（单机：NVLink 与 P2P，含 CUDA IPC）、[c-build.md](c-build.md)（产物与编译）、[roce-swtich.md](roce-swtich.md)（RoCE 交换机队列配置的通用说明）。

<a id="sum"></a>
## 摘要

[↑ 返回目录](#toc)

| 问题 | 结论 | 可靠度 |
|---|---|---|
| CPU 内存与 GPU 显存的 RDMA 注册有何不同？ | **API 几乎一样**（都是 `ibv_reg_mr`），但**背后实现不同**：CPU 内存由内核钉页取总线地址；GPU 显存没有主机页，必须由 **NVIDIA 驱动**提供总线地址。GPU RDMA 有两种实现：**`nvidia_peermem`**（经 peer memory client，`ibv_reg_mr` 直接收 GPU 指针）和 **dmabuf**（CUDA 导出 fd + `ibv_reg_dmabuf_mr`，走内核通用框架） | 机制为**推断**；两种方式在本环境都**实测**可用 |
| 两种 GPU 实现方式有什么区别？ | 注册开销：128 MiB 时 dmabuf 约 0.23 ms、peermem 约 2.1 ms；VMM 显存：peermem 要设 `gpuDirectRDMACapable = 1`，dmabuf 不挑；**传输带宽无差别**（都是 29.8 / 30.0 GB/s），两端可混用 | **实测** |
| GPU 显存的 RDMA 读写正确吗？ | WRITE（server 校验）、READ（client 校验）、带宽测试之后再校验，全部 PASS，退出码 0 | **实测** |
| 带宽多少？ | WRITE 约 **29.8 GB/s**（238 Gb/s），READ 约 **30.0 GB/s**（240 Gb/s），只有网卡标称 400 Gb/s 的 60% | **实测**（单 QP、单条 64 MiB 消息） |
| 30 GB/s 是网络上限吗？ | **不是**：同一对网卡上，主机内存能跑到约 44 GB/s；与注册方式也无关 | **实测**；"与 GPU 一侧的访问路径有关"是**推断**，未验证 |
| 数据走的是 RDMA 而不是 `eth0` 吗？ | 是：RDMA 流量约 1.3 GiB，`eth0` 上只有约 1 KiB | **实测** |
| 数据经过主机内存吗？ | 不经过（GPUDirect RDMA） | **推断**（由直接注册 GPU 指针得出） |
| `-t`（流量类别）有影响吗？ | 在测量范围内，**GPU 版不敏感**（`-t 0` 与 `-t 104` 数字相同）；主机内存版的 READ 则很敏感。仍统一推荐 `-t 104` | **实测**；GPU 版样本有限 |
| 与单机 NVLink 比？ | NVLink 779.4 GB/s，约是 RDMA 的 26 倍（测量方法不同，只看量级） | **实测** |
| 产业趋势？ | NVIDIA 官方**推荐 dmabuf、把 `nvidia-peermem` 称为 legacy**；NCCL 条件满足时**默认走 dmabuf**；Ubuntu 从 6.8 内核起**明确弃用** `nvidia-peermem`。但**没有找到** NVIDIA 官方宣布弃用或移除的原文 | 官方文档与公告（2026-11-01 联网查阅，§1.11）；NCCL 在本环境是否真走 dmabuf **未核实** |

<a id="intro"></a>
## 读完应该能回答

1. 为什么 RDMA 必须先"注册"内存？注册 CPU 内存与注册 GPU 显存，有什么相同和不同？（§1.1–§1.3）
2. GPU RDMA 有哪几种实现方式？`nvidia_peermem` 与 dmabuf 各自怎么工作、有什么区别、怎么选？产业上的趋势是什么？（§1.4–§1.11）
3. 两台机器上的两张 GPU，怎样经 RDMA 交换数据？与主机内存版有什么不同？（§2）
4. `server-gpu.cu` 与 `client-gpu.cu` 各自怎么写？（§4）
5. 实测：数据对不对？带宽多少？数据确实走了 RDMA 而不是 `eth0` 吗？（§5）
6. 为什么 GPU 显存只有约 30 GB/s，而主机内存能到约 44 GB/s？（§5.7）

<a id="conv"></a>
## 约定与术语

[↑ 返回目录](#toc)

**约定**

- **实测**：在 server-0 / server-1（nvcc 13.0，8×B300，驱动 590.48.01）上的命令输出；其中 §1 里涉及注册耗时、VMM、BAR1、生命周期的实测，来自探查程序 `reg_bench`（只做注册与注销，不传数据，在 server-0 的 GPU0 / `mlx5_0` 上运行，见 §5.8）；**推断**：由机制或公开资料推导，未在本机观察；**代码事实**：直接来自源码；**未测 / 未验证 / 未查明**：既没有证据也没有测量，或有现象没有原因。全部结论汇总见附录 A。
- 带宽单位：十进制 GB/s；`Gb/s` = GB/s × 8；`MiB` 为 2^20 字节；网卡标称 400 Gb/s = 50 GB/s。耗时 `ms` 为毫秒，`reg_bench` 的耗时取 5 次的中位数。
- 文中 server-0 / server-1 指 StatefulSet `server` 的两个 Pod，分别在两个不同的 GPU 节点上。
- 本篇的"GPU"指 `cudaMalloc` 或 `cuMemCreate` 分配的显存；"主机内存"指普通进程内存。
- RDMA 的通用术语（PD / MR / QP / CQ / WR / WC、lkey / rkey、GID、OOB 等）见 [c-multi-node-cpu.md §2.1](c-multi-node-cpu.md#2-1)，本篇直接使用。

**本篇用到的术语**

| 术语 | 含义 |
|---|---|
| **MR** | Memory Region，内存区域：注册之后得到的对象，网卡凭它访问那段内存 |
| **pin（钉页）** | 让这些内存页在物理上固定不动：不被换出、不被迁移 |
| **MTT** | Memory Translation Table，网卡内部的翻译表：从"注册时用的地址"到"总线（物理）地址" |
| **总线地址 / IOVA** | 网卡做 DMA 时使用的地址，与进程的虚拟地址不是一回事 |
| **GPUDirect RDMA** | 网卡直接读写 GPU 显存，不经过主机内存。有 `nvidia_peermem` 与 dmabuf 两种实现方式 |
| **BAR1** | GPU 暴露给 PCIe 总线的一段地址窗口；其他 PCIe 设备（如网卡）通过它访问显存 |
| **nvidia_peermem** | NVIDIA 驱动附带的内核模块，让 RDMA 子系统认得 GPU 指针（"peer memory client"） |
| **dmabuf** | Linux 内核通用的"设备间共享缓冲区"机制，用一个文件描述符代表一段内存 |
| **VMM** | CUDA 虚拟内存管理 API（`cuMemCreate`、`cuMemMap` 等），可以把"物理显存"和"虚拟地址"分开管理 |
| **ODP** | On-Demand Paging，按需分页：注册时不钉页，网卡访问到时再由内核处理缺页 |
| **UVA** | 统一虚拟地址：`cudaMalloc` 返回的设备指针在 64 位进程里是全局唯一的虚拟地址 |
| **fabric** | 本集群里专用于 RDMA 的网口（`fabric0`、`fabric2`……），与普通的 `eth0` 区分开 |

<a id="1"></a>
## 1. 原理：CPU 与 GPU 的 RDMA 有什么不同

[↑ 返回目录](#toc)

RDMA 的第一步，是把一段内存**注册**（`ibv_reg_mr`）给网卡。注册 CPU 内存和注册 GPU 显存，在 API 上几乎一样，但**背后的实现、耗时、限制、坑**差别很大。本章先讲清这些，后面的程序与实测才好理解。

<a id="1-1"></a>
### 1.1 为什么要注册

网卡做 DMA 时，**不认进程的虚拟地址**，只认**总线地址**（物理地址或经 IOMMU 翻译后的 IOVA）。要让网卡安全地读写你的内存，有三个问题必须解决：

| 问题 | 为什么麻烦 | 注册怎么解决 |
|---|---|---|
| 地址翻译 | 进程用虚拟地址，网卡要总线地址，两者之间的映射由内核维护 | 注册时由驱动算出每一页的总线地址，写进网卡内部的翻译表（MTT） |
| 页必须固定 | 普通内存页可以被换出到磁盘、被内核迁移；网卡 DMA 期间页一动，就会读写到错误的位置 | 注册时把这些页**钉住**，注销前不会动 |
| 权限 | 不能让对端随便读写你的任意内存 | 注册时指定访问权限（本地写、远端读、远端写），并发给你两把钥匙 lkey / rkey |

一次 `ibv_reg_mr` 完成四件事：

| 步骤 | 做什么 | 代价体现在哪 |
|---|---|---|
| ① 钉页 | 确保这段内存的每一页都在物理内存（或显存）里，并固定住 | 页数越多越慢；主机内存若还没分配物理页，要先触发缺页（§1.2） |
| ② 取总线地址 | 算出每一页对应的总线地址 | 同上 |
| ③ 建翻译表 | 把这些地址写进网卡的 MTT | 与页数成正比 |
| ④ 发 key | 生成 lkey / rkey 并登记权限 | 固定开销，很小 |

注册完成后，数据路径上网卡就用这张表翻译、直接 DMA，**CPU 不再参与**。所以**注册是一次性的重开销**：好的做法是启动时注册一次、反复使用，不要每次传输都注册（§1.10）。

<a id="1-2"></a>
### 1.2 CPU 内存怎么注册

CPU 内存是注册的"标准情形"。内核里的大致路径（**推断**，依据公开的 Linux RDMA 子系统设计，本机未跟踪内核调用）：

```
ibv_reg_mr(pd, ptr, len, access)          用户态：libibverbs
   → ib_uverbs                             内核：用户态 verbs 的入口
   → mlx5_ib（网卡驱动）
   → ib_umem_get()                         把用户虚拟地址范围变成"umem"
        → pin_user_pages()                 逐页钉住（必要时先触发缺页分配物理页）
        → dma_map（IOMMU / 总线地址）
   → 把各页的总线地址写进网卡的 MTT，返回 lkey / rkey
```

关键点：**这条路依赖"主机内存页"这个概念**——每一页都有对应的内核页描述，内核知道怎么钉住它、怎么取它的物理地址。钉页的原因、"按页"的含义、为什么注册前要先写一遍每一页，详见 [c-multi-node-cpu.md §3](c-multi-node-cpu.md#3)。

**资源与限制**：

| 项 | 说明 |
|---|---|
| 占用什么 | **物理内存**。被钉住的页不能换出，注册越多，系统可用的物理内存越少 |
| 钉页数量限制 | 受 `RLIMIT_MEMLOCK`（`ulimit -l`）约束；拥有 `CAP_IPC_LOCK` 能力的进程可以绕过（公开资料） |
| 本集群的情形 | Pod 里 `ulimit -l` 为 **unlimited**（**实测**），且 `n-server*.yaml` 给了 `IPC_LOCK` 能力——所以这个限制在本环境里**测不出来**，也就没有测 |
| 页大小 | 系统页 4096 字节（**实测**）；透明大页设置为 `[always]`（**实测**），所以主机内存可能由大页支撑，翻译表项更少 |
| 网卡的 MR 上限 | `max_mr_size` 为无限制、`max_mr` 为 16,777,216 个（**实测**，`ibv_devinfo -v`） |

**实测：注册耗时**（`reg_bench lat`，分配后注册再注销，只测 `ibv_reg_mr` 与 `ibv_dereg_mr` 本身）。**"未触摸的页"**指 `posix_memalign` 之后没有写过内存就直接注册；**"已触摸的页"**指注册前先 `memset` 把每页写一遍：

| 方式 | 大小 | 注册（中位 ms） | 注销（中位 ms） |
|---|---|---|---|
| CPU（未触摸的页） | 1 MiB | 0.039 | 0.017 |
| CPU（未触摸的页） | 64 MiB | 7.097 | 0.040 |
| CPU（未触摸的页） | 1 GiB | **244.515** | 0.090 |
| CPU（未触摸的页） | 4 GiB | **1016.979** | 0.235 |
| CPU（已触摸的页） | 1 MiB | 0.040 | 0.018 |
| CPU（已触摸的页） | 64 MiB | 0.221 | 0.039 |
| CPU（已触摸的页） | 1 GiB | 3.275 | 0.074 |
| CPU（已触摸的页） | 4 GiB | **16.505** | 0.217 |

怎么读：

1. **耗时随大小增长**：已触摸的页，大小变为 4 倍（1 GiB → 4 GiB），耗时约 5 倍（3.3 → 16.5 ms），大体与页数成正比；
2. **从 64 MiB 起，"未触摸"比"已触摸"慢约 30–75 倍**：64 MiB 时 7.1 ms 对 0.22 ms（约 32 倍），1 GiB 时 244.5 ms 对 3.3 ms（约 75 倍），4 GiB 时 1017 ms 对 16.5 ms（约 62 倍）；1 MiB 时两者没有差别。差额**推断**是首次访问页时的缺页开销——钉页要求每页都有物理内存，而 `posix_memalign` 出来的内存在第一次被写之前并没有分配物理页，是注册这一步替你"补"了这件事；
3. **注销很快**：4 GiB 也只要 0.2 ms；小缓冲（1 MiB）的注册只要约 0.04 ms。

**实践含义**：大块 CPU 内存**先触摸再注册**，能避免把缺页的开销算在注册头上（用 `mmap` 加 `MAP_POPULATE` 预先分配页也应有同样效果，本篇未测）；更重要的是**注册一次、反复使用**。

**不钉页的替代：ODP**。ODP 让注册时不钉页，网卡访问到某页时，若页不在内存里，由内核处理缺页。**实测**：`mlx5_0` 的 `general_odp_caps` 为 `ODP_SUPPORT` 和 `ODP_SUPPORT_IMPLICIT`，即本卡支持 ODP，包括"隐式 ODP"（一次覆盖整个地址空间）。各传输类型（RC 等）具体支持哪些操作本篇没有逐项查看，也**没有实际使用过 ODP**。ODP 只适用于**主机内存**；GPU 显存的 ODP 不在本篇范围内（未测）。

<a id="1-3"></a>
### 1.3 GPU 显存为什么要走另一条路

§1.2 的路径依赖"主机内存页"。**GPU 显存不是主机内存**：它没有对应的主机页描述，内核的 `pin_user_pages` 对它不适用（**推断**）。所以必须有 **GPU 驱动参与**：只有 NVIDIA 驱动知道一段显存的每一页在 GPU 上对应哪个总线地址（通过 BAR1 暴露给 PCIe 总线）。

要让网卡访问显存，就是要让 GPU 驱动把这些"总线地址"交给 RDMA 子系统。这就是 **GPUDirect RDMA**，目前有两种实现方式：

| | 方式一：`nvidia_peermem` | 方式二：dmabuf |
|---|---|---|
| 思路 | 向 RDMA 子系统注册一个"peer memory client"，由它认出 GPU 指针 | CUDA 把显存导出成 dmabuf fd，用 Linux 内核通用的 dmabuf 框架交给网卡 |
| 注册 API | 与 CPU 内存**同一个** `ibv_reg_mr` | `cuMemGetHandleForAddressRange` + `ibv_reg_dmabuf_mr` |
| 需要什么 | 宿主机加载 `nvidia_peermem` 模块 | 开放内核模块、较新的内核，不需要 `nvidia_peermem` |
| 本环境 | 支持（实测） | 支持（实测） |

<a id="1-4"></a>
### 1.4 实现方式一：nvidia_peermem

**机制**（**推断**，依据 NVIDIA 与 Linux RDMA 的公开设计）：`nvidia_peermem` 内核模块向 RDMA 子系统注册为一个 **peer memory client**。当 `ibv_reg_mr` 收到一个不是普通主机页的地址时，RDMA 子系统会逐个询问已注册的 client "这段地址是不是你的"；`nvidia_peermem` 认出是 GPU 显存，就调用 NVIDIA 驱动取得这段显存的页信息与总线地址，再交给网卡。

**API**：与注册 CPU 内存**完全相同**——同一个 `ibv_reg_mr`，只是传进去的是 `cudaMalloc` 出来的指针：

```cpp
void* p;  cudaMalloc(&p, size);
ibv_mr* mr = ibv_reg_mr(pd, p, size, access);     // 和注册主机内存的调用一模一样
```

**实测**：

- `/proc/modules` 里 `nvidia_peermem` 已加载（两台宿主机都是），且它同时挂在 `nvidia`（GPU 驱动）和 `ib_uverbs`（RDMA 用户态入口）的使用者列表里，与上面的机制描述一致；
- `ibv_reg_mr(GPU 指针)` 在 `mlx5_0` 上成功，两个 GPU 程序里两端也都一次成功；
- 耗时（中位 ms；两次独立运行的结果相近）：

| 大小 | 注册 | 注销 |
|---|---|---|
| 1 MiB | 1.580 | 0.618 |
| 64 MiB | 1.668 | 0.647 |
| 1 GiB | 3.201 | 1.196 |
| 4 GiB | 9.117 | 2.905 |

与注册 CPU 内存对比：**小缓冲时 GPU 路径贵得多**（1 MiB：1.58 ms 对 0.04 ms，约 40 倍）——有约 1.6 ms 的**固定开销**；**大缓冲时反而不贵**（4 GiB：9.1 ms，已触摸的 CPU 内存是 16.5 ms，未触摸的是 1017 ms）。GPU 注册没有"缺页"这一项，因为显存在 `cudaMalloc` 时就已经分配好了。

<a id="1-5"></a>
### 1.5 实现方式二：dmabuf

**机制**（**推断**，依据公开设计）：CUDA 把一段显存**导出**成一个 **dmabuf 文件描述符**，再用 `ibv_reg_dmabuf_mr` 把这个 fd 注册给网卡。这条路走的是 **Linux 内核通用的 dmabuf 框架**，不需要 `nvidia_peermem` 这个专门的模块。公开资料给出的前提：内核 ≥ 5.12、rdma-core ≥ 34、CUDA ≥ 11.7、NVIDIA 开放内核模块——本机都满足（内核 6.12、rdma-core 39、CUDA 13.0，驱动是 "NVIDIA UNIX Open Kernel Module"，均**实测**可见，两台机器一致）。

```cpp
int fd;
cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)p, size,
                              CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);      // ① 导出 dmabuf
ibv_mr* mr = ibv_reg_dmabuf_mr(pd, 0, size, (uint64_t)p, fd, access);       // ② 注册
/* …… 用完：ibv_dereg_mr(mr); close(fd); */
```

**实测**：

- `DMA_BUF_SUPPORTED = 1`（`cuDeviceGetAttribute`）；导出与注册都成功；
- **耗时几乎与大小无关**（中位 ms）：

| 大小 | 导出 | 注册 | 注销 | 导出 + 注册 |
|---|---|---|---|---|
| 1 MiB | 0.121 | 0.034 | 0.032 | 0.155 |
| 64 MiB | 0.121 | 0.035 | 0.033 | 0.156 |
| 1 GiB | 0.123 | 0.039 | 0.033 | 0.162 |
| 4 GiB | 0.125 | 0.047 | 0.033 | 0.172 |

- 对照 `nvidia_peermem`：1 MiB 时 1.58 ms 对 0.155 ms，4 GiB 时 9.1 ms 对 0.172 ms。**dmabuf 的注册开销低得多，而且不随大小增长**。这是否因为 dmabuf 的映射推迟到网卡真正访问时才建立，本篇**没有验证**，原因未查明；
- **完整的跨机传输也实测通过**：两个 GPU 程序用 `-m dmabuf` 切换到这种方式，WRITE / READ 校验全部 PASS，**带宽与 peermem 没有差别**（都是 WRITE 29.8、READ 30.0 GB/s），两端可以混用（§5.4 表 E）。所以 dmabuf 只是"注册更便宜"，不是"传输更快"。

<a id="1-6"></a>
### 1.6 VMM 分配的显存

**VMM 是什么**：VMM = Virtual Memory Management，CUDA 的虚拟内存管理 API，把"物理显存"和"虚拟地址"分开管理。它和 `cudaMalloc` 的区别：

| | `cudaMalloc` | VMM（`cuMem*` 一组 API） |
|---|---|---|
| 一步还是多步 | 一步：分配并得到指针 | 多步：分别做"分配物理显存""预留虚拟地址""把两者映射起来""设置访问权限" |
| 涉及的 API | `cudaMalloc` | `cuMemCreate`（创建物理显存块）→ `cuMemAddressReserve`（预留一段虚拟地址）→ `cuMemMap`（映射）→ `cuMemSetAccess`（设置哪些设备可以访问） |
| 物理与虚拟的关系 | 绑定在一起，分配后不能改 | 解耦：可以先预留一大段地址，再逐块把物理显存映射上去，也能解除、换映射 |
| 谁在用 | 普通 CUDA 程序（**本篇的两个程序就是这样分配的**） | 框架的缓存分配器、NCCL 的内存分配等 |

设计动机（公开资料的一般说法，本篇没有逐项测）：缓冲可以**增长**而不改变指针（地址先预留好，需要多少再映射多少物理显存）；减少碎片；方便在进程或设备之间共享。

**VMM 显存能不能注册？** **实测**（256 MiB，分配粒度 2048 KiB）：

| 分配时的 `gpuDirectRDMACapable` | 经 `nvidia_peermem`（`ibv_reg_mr`） | 经 dmabuf（`ibv_reg_dmabuf_mr`） |
|---|---|---|
| 0（默认） | **失败**：`Bad address` | 成功 |
| 1 | 成功 | 成功 |

另外，设备属性 `GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 1`（**实测**）。

结论：

1. **用 VMM 分配、又要走 `nvidia_peermem` 注册，分配时必须设 `gpuDirectRDMACapable = 1`**，否则注册会失败；
2. **dmabuf 路线对 VMM 显存不挑**：不论这个标志设没设，都注册成功；
3. 所以**只要显存可能由 VMM 分配（很多框架的缓存分配器默认这样做），dmabuf 是更省心的选择**——这是基于上表的结论，本篇没有测试具体某个框架是否真的这样分配；VMM 显存经 dmabuf 做**完整传输**也没有测过。

**与本文的关系——NCCL 默认就用 VMM**（NCCL 官方环境变量文档，2026-11-01 查阅）：

- `NCCL_CUMEM_ENABLE`（自 2.18）：用 CUDA `cuMem*` 函数分配 NCCL 的内存；2.18 默认关闭，**从 2.19 起，系统支持时自动开启**；
- `NCCL_CUMEM_HOST_ENABLE`（自 2.23）：对主机内存也用 `cuMem*`；2.24 起，在 CUDA 驱动 ≥ 12.6、运行时 ≥ 12.2 且系统支持时默认开启；
- 文档**没有**把这两个变量与 GPUDirect RDMA、dmabuf 或 `nvidia-peermem` 联系起来说明。

把它和上面的实测放在一起，可以**推断**：NCCL 2.19 起默认走 VMM 分配，而 VMM 显存经 peermem 注册有额外条件（上表）、经 dmabuf 注册没有，加上 `NCCL_DMABUF_ENABLE` 默认为 1（§1.11），所以 NCCL 倾向走 dmabuf。**这是推断**：未见官方文档把这几点明确联系起来，也**没有在本环境用 NCCL 验证**它实际选了哪种注册方式（可以看 `NCCL_DEBUG=INFO` 的输出）。另外要注意：**本篇的程序用 `cudaMalloc`，peermem 能跑通，不代表同一环境里的 NCCL 或框架也能用 peermem 跑通**。

<a id="1-7"></a>
### 1.7 BAR1 与页粒度

**BAR1 是什么**：BAR 是 PCIe 设备的 Base Address Register（基地址寄存器），设备通过它向 PCIe 总线声明"我占用这一段地址空间"。GPU 有几个 BAR，其中 BAR0 通常是控制寄存器，**BAR1 是显存（framebuffer，即 HBM）的访问窗口**（公开资料的一般说法，本篇未验证）。往 BAR1 这段地址里读写，经 PCIe 就会落到对应的 GPU 显存上；**映射关系由 GPU 驱动管理**——哪段显存被映射进窗口、映射到窗口的哪个位置，由驱动决定。

要点（常见的误解）：

| 说法 | 修正 |
|---|---|
| "BAR1 把 HBM 暴露给 CPU" | **不只是 CPU**。凡是能发起 PCIe 访问的设备都能经 BAR1 访问显存：CPU（内存映射 I/O）、**网卡**、其他 GPU。GPUDirect RDMA 用的正是网卡这一种——网卡的 DMA 经 BAR1 读写显存，网卡拿到的"总线地址"落在 BAR1 窗口里 |
| "把全部 HBM 暴露出来" | 暴露的是一个**地址窗口**。窗口大小是 BAR1 的总量，不一定等于显存总量；窗口里同一时刻被映射的部分才能被访问 |
| "CUDA 程序访问显存都经过 BAR1" | **不是**。普通的 `cudaMemcpy` 由 GPU 的拷贝引擎走 DMA，不经过 BAR1；BAR1 主要用在其他 PCIe 设备直接访问显存的场景，如 GPUDirect RDMA（公开资料的一般说法，本篇未测） |

**本机实测**：`nvidia-smi -q -d MEMORY` 显示本机 GPU0 的 BAR1 总量为 **524288 MiB（512 GiB）**，远大于一般的显存用量，所以 BAR1 不是本机上的瓶颈。

一个出乎预料的观察（**实测**，GPU0，Used 单位 MiB）：

| 时刻 | BAR1 Used |
|---|---|
| 基线 | 629 |
| `cudaMalloc` 4 GiB，**未注册** | 4725（比基线多 4096） |
| `ibv_reg_mr` 注册 4 GiB（peermem） | 4725 |
| 注销之后 | 4725 |
| `ibv_reg_dmabuf_mr` 注册 4 GiB | 4725 |
| 注销之后 | 4725 |

即**这台机器上 BAR1 的占用在 `cudaMalloc` 时就增加了，注册与注销前后都没有变化**。一般的说法是"GPUDirect RDMA 注册会占用 BAR1"，在本机的这个驱动与配置下，**不能用 BAR1 Used 的变化来观察注册开销**，原因未查明。

**页粒度**：公开资料说 GPU 显存按 **64 KiB** 页注册，主机内存按 4 KiB（或大页）。本篇**没有直接测量**页粒度。

<a id="1-8"></a>
### 1.8 生命周期与一致性

**注册之后释放显存**（**实测**，256 MiB）：

| 方案 | `cudaFree`（MR 未注销） | 之后 `ibv_dereg_mr` | 进程 |
|---|---|---|---|
| `nvidia_peermem` | 返回成功 | 返回 0（成功） | 正常 |
| dmabuf | 返回成功 | 返回 0（成功） | 正常 |

注意这**只说明这两个 API 调用本身不报错**，**不代表之后拿这个 MR 去做 RDMA 是安全的**——本篇没有测释放之后再发起 RDMA 会怎样。规则仍然是：**先 `ibv_dereg_mr`，再 `cudaFree`**；dmabuf 方式下，注销 MR 之后还要自己 `close(fd)`。

**一致性与同步**（GPUDirect RDMA 的特殊点）：

- 注册之前对这块显存设 `CU_POINTER_ATTRIBUTE_SYNC_MEMOPS`（`server-gpu.cu` / `client-gpu.cu` 的做法）：这是 GPUDirect RDMA 的惯例，保证 CUDA 的内存操作在 RDMA 访问之前完成（§3.1）；
- 填完数据要先 `cudaDeviceSynchronize()`，再把地址和 rkey 交给对端；
- 设备属性（**实测**，并对照 `cuda.h` 里的枚举注释）：

| 属性 | 值 | 含义（来自 `cuda.h`） |
|---|---|---|
| `GPU_DIRECT_RDMA_SUPPORTED` | 1 | 支持 GPUDirect RDMA |
| `DMA_BUF_SUPPORTED` | 1 | 支持导出 dmabuf |
| `GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS` | 1 | 位 0 = `..._OPTION_HOST`：支持 `cuFlushGPUDirectRDMAWrites()`；位 1（MEMOPS）不支持 |
| `GPU_DIRECT_RDMA_WRITES_ORDERING` | 100 | `..._ORDERING_OWNER`：设备本身可以一致地消费远端写，**其他 CUDA 设备不一定** |

主机内存没有这个问题：CPU 与 DMA 的一致性由硬件和内核保证。

<a id="1-9"></a>
### 1.9 对照总表

| 维度 | CPU 内存 | GPU 显存（`nvidia_peermem`） | GPU 显存（dmabuf） |
|---|---|---|---|
| 谁钉页 / 取总线地址 | 内核（`pin_user_pages`，推断） | NVIDIA 驱动，经 peer memory client（推断） | NVIDIA 驱动导出 dmabuf，经内核通用框架（推断） |
| 注册 API | `ibv_reg_mr` | `ibv_reg_mr`（**同一个**） | `cuMemGetHandleForAddressRange` + `ibv_reg_dmabuf_mr` |
| 需要的内核模块 | 无 | `nvidia_peermem`（本机已加载，实测） | 无额外模块；需要开放内核模块与较新的内核 |
| 占用的资源 | 物理内存；受 `RLIMIT_MEMLOCK` 约束（本环境测不出） | 显存；BAR1 | 显存；BAR1 |
| 页粒度 | 4 KiB（可由大页支撑） | 64 KiB（公开资料，未测） | 同左 |
| 注册耗时（1 MiB） | 0.04 ms | 1.58 ms | 0.155 ms（导出 + 注册） |
| 注册耗时（4 GiB） | 已触摸 16.5 ms；**未触摸 1017 ms** | 9.1 ms | 0.172 ms |
| 注销耗时（4 GiB） | 0.22 ms | 2.9 ms | 0.033 ms |
| 耗时随大小 | 大体与大小成正比；未触摸页最陡 | 固定开销 + 缓慢增长 | 几乎不变 |
| VMM（`cuMemCreate`）显存 | 不适用 | 需 `gpuDirectRDMACapable = 1`，否则失败（实测） | 不挑，均成功（实测） |
| 注册后立即释放 | — | API 不报错（实测）；之后能否 RDMA 未测 | 同左 |
| 一致性 / 同步 | 硬件与内核保证 | 需 `SYNC_MEMOPS`、先同步再交地址 | 同左 |
| 不钉页的替代 | ODP（本卡支持，未使用） | 无 | 无 |
| **跨机传输带宽**（单 QP、64 MiB、`-t 104`） | READ 43.8、WRITE（深度 1）44.1 GB/s | WRITE 29.8、READ 30.0 GB/s | WRITE 29.8、READ 30.0 GB/s（与 peermem 相同） |

最后一行是**实测**（CPU 数据见 [c-multi-node-cpu.md §5.4](c-multi-node-cpu.md#5-4)，GPU 数据见本篇 §5.4）。**注册只决定"网卡能不能访问这块内存"，不决定访问有多快**：同一对网卡上，主机内存的传输带宽高于 GPU 显存，两种 GPU 注册方式之间则没有差别（§5.7）。

<a id="1-10"></a>
### 1.10 编程：怎么写、怎么选

三种方式的代码对照：

```cpp
// ① CPU 内存
void* p;  posix_memalign(&p, 4096, size);
memset(p, 0, size);                                    // 先触摸，避免把缺页算在注册头上
ibv_mr* mr = ibv_reg_mr(pd, p, size, access);

// ② GPU 显存，nvidia_peermem（-m peermem，默认）
void* p;  cudaMalloc(&p, size);
unsigned one = 1;  cuPointerSetAttribute(&one, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)p);
/* 填数据 */  cudaDeviceSynchronize();
ibv_mr* mr = ibv_reg_mr(pd, p, size, access);          // 与 ① 是同一个调用

// ③ GPU 显存，dmabuf（-m dmabuf）
void* p;  cudaMalloc(&p, size);                        // 或 VMM 分配
int fd;  cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)p, size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
ibv_mr* mr = ibv_reg_dmabuf_mr(pd, 0, size, (uint64_t)p, fd, access);
/* 用完：ibv_dereg_mr(mr); close(fd); */
```

**怎么选**（基于注册耗时、VMM 行为的实测，以及两种方式传输带宽相同这一实测）：

| 情形 | 建议 |
|---|---|
| 注册主机内存 | `ibv_reg_mr`，大块先触摸；**注册一次、反复使用** |
| 注册显存，且系统有 dmabuf 条件 | **优先 dmabuf**：注册便宜且不随大小增长，不依赖 `nvidia_peermem`，对 VMM 显存不挑；传输带宽与 peermem 相同 |
| 注册显存，没有 dmabuf 条件 | 用 `nvidia_peermem`；若显存是 VMM 分配的，分配时设 `gpuDirectRDMACapable = 1` |
| 缓冲很小、很多、频繁创建销毁 | **不要每次都注册**：先注册一块大缓冲，自己切分管理（peermem 的 1.6 ms 固定开销会很显眼） |

本项目的 `server-gpu.cu` / `client-gpu.cu` 默认用方式 ②（原因是实现最直接），并带 **`-m dmabuf`** 开关切换到方式 ③（§3.1、§4.1）。

<a id="1-11"></a>
### 1.11 产业动态与趋势：dmabuf 正在成为推荐路线

本节依据 2026-11-01 联网查到的**官方文档**与**发行版公告**（来源见节末），与本环境的实测分开写。

**结论**：产业方向是 **dmabuf 取代 `nvidia-peermem`**，但需要区分三种说法的强度：

| 说法 | 来源 | 强度 |
|---|---|---|
| **推荐用 dmabuf，而不是 `nvidia-peermem`**："NVIDIA recommends using the DMA-BUF rather than using the `nvidia-peermem` kernel module from the GPU Driver." | NVIDIA GPU Operator 文档（latest） | 官方**推荐**；该页**没有**说 `nvidia-peermem` 已弃用或计划移除，只是推荐 |
| **`nvidia-peermem` 是 legacy**，NCCL 在条件满足时**默认走 dmabuf**：`NCCL_DMABUF_ENABLE`（自 NCCL 2.13 起）默认值为 1，内核或 CUDA / 网卡驱动不支持时自动关闭 | NVIDIA GPU Operator 文档、NCCL 环境变量文档 | 官方文档的**默认行为** |
| **明确弃用**：Ubuntu 从 6.8 内核起弃用旧的 `nvidia-peermem` 接口，推荐 dma-buf（影响 Ubuntu 22.04 HWE 与 24.04 LTS；22.04 GA 与 20.04 不受影响） | Ubuntu 公告 | **下游发行版**的弃用，不是 NVIDIA 的声明 |

**两条路线的前提**（来自 NVIDIA 文档）：

| | dmabuf | `nvidia-peermem`（legacy） |
|---|---|---|
| GPU 驱动 | **必须是开放内核模块** | 任何受支持的驱动 |
| CUDA | 11.7 及以上 | 无最低版本 |
| GPU | Turing 及更新 | 所有数据中心、Quadro RTX、RTX 及更新 |
| 网卡驱动 | MLNX_OFED / DOCA-OFED **可选**，发行版自带驱动即可 | MLNX_OFED / DOCA-OFED **必需** |
| Linux 内核 | 5.12 及以上 | 无最低版本 |

Ubuntu 公告补充：dmabuf 对 Volta 及更早的 GPU 不可用；Ampere / Hopper 要求 CUDA 11.7+、驱动分支 515+ 的开放版本、NCCL 2.13.4+、UCX 1.15.0+。NVIDIA 驱动 595.84 的 README 写明：**Blackwell 及以后只支持开放内核模块**；DMABUF 支持（CUDA 分配）属于只在开放版里可用的功能；并建议所有支持的 GPU 都用开放内核模块。

**dmabuf 比 peermem "先进"吗？**——更接近**标准化与解耦的演进**，而不是数据通路上的性能突破：

- **数据通路上没有区别**：两种方式下，网卡最终都是经 PCIe 直接 DMA 访问 GPU 显存（通过 BAR1），是同一条路径。本环境**实测**两者带宽相同（WRITE 29.8、READ 30.0 GB/s），首次操作耗时也相同，所以 dmabuf 没有让数据跑得更快；
- **差别在控制面**——谁来把显存"交给"网卡：

| 方面 | `nvidia-peermem` | dmabuf |
|---|---|---|
| 接口性质 | 厂商专用的 peer memory client 回调接口；按 NVIDIA 官方文档，legacy 路线要求 OFED | Linux 内核通用的"设备间共享缓冲区"框架，导出方（GPU 驱动）与导入方（RDMA 驱动）是标准角色 |
| 耦合 | `nvidia_peermem` 要与 RDMA 栈（OFED）的特定接口绑在一起，版本与安装顺序出问题会加载失败（论坛有用户报告） | 双方只依赖内核标准接口，发行版自带驱动即可（NVIDIA 文档：OFED 对 dmabuf 是可选） |
| 可维护性 | 要随驱动、OFED 版本配套 | 由内核主线维护 |
| 本环境实测差别 | 注册 128 MiB 约 2.1 ms；VMM 显存要设 `gpuDirectRDMACapable = 1`，否则注册失败 | 注册约 0.23 ms；VMM 显存不论设没设都成功 |

- **哪些是实测、哪些是推断**：注册耗时差约 9 倍、VMM 行为不同、带宽相同，是**实测**；"dmabuf 注册便宜是因为映射推迟到网卡访问时才建立"是**推断**，未验证；"dmabuf 框架有标准的动态导入机制，导出方在显存被移动或释放时可以通知导入方撤销映射"是**公开资料的一般说法**，本篇没有读内核或驱动源码核对——peer memory client 也有自己的失效回调，所以**不能据此说 dmabuf 的生命周期管理一定更强**，只能说它是标准化的；
- **dmabuf 的代价**：要开放内核模块、内核 ≥ 5.12、CUDA ≥ 11.7、Turing 及更新的 GPU（Volta 及更早的用不了）；API 多一步——要先导出得到 fd，用完还要自己 `close`（程序里的 `dmabuf_fd`）；
- **小结**：dmabuf 是把 GPU 与 RDMA 的对接，从厂商专用的私有挂接，换成内核的通用标准接口。收益是**工程上的**：更少的外部依赖、更好的可维护性、对新型分配方式（如 VMM）更友好、注册更便宜；不是带宽更高。这与产业上"NVIDIA 推荐 dmabuf、legacy 路线仍保留给老环境"的做法一致。

**对本环境的意义**：

- 本环境两项前提都满足（驱动是开放内核模块，内核 6.12；GPU 是 Blackwell 架构的 B300，只能用开放内核模块），所以 dmabuf 正是文档推荐的路线，与本篇的实测一致：两种方式都能完整传输、带宽相同，dmabuf 注册更便宜（§5.4 表 E）；
- **NCCL 默认会走 dmabuf**（条件满足时）：这是按文档**推断**的，**本环境没有用 NCCL 日志核实**——可以看 `NCCL_DEBUG=INFO` 的输出确认 NCCL 实际选了哪种注册方式；
- **legacy 路线仍有用处**：Volta 及更早的 GPU、没有开放内核模块的驱动、内核 < 5.12 的环境，仍只能用 `nvidia-peermem`（并且需要 OFED）。

**没有查到 / 要谨慎的**：

- **没有找到 NVIDIA 宣布 `nvidia-peermem` 弃用或移除的官方原文**；官方页面都只写"推荐 dmabuf，peermem 是 legacy 选项"；
- 论坛里有用户报告，在 DGX Spark 上 `nvidia-peermem` 因驱动与 OFED 的安装顺序、编译时缺少所需符号而加载失败（**用户报告，不是官方结论**）；
- 个别开发者帖子说"两者目前没有性能差异"，与本篇实测一致，但那只是个别说法；
- 关于"上游内核没有 peer memory 接口"的说法**没有查到来源**，不要当作事实。

**来源**（2026-11-01 查阅）：

- NVIDIA GPU Operator：[GPUDirect RDMA and GPUDirect Storage](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-rdma.html)
- Ubuntu：[NVIDIA GPUDirect over Infiniband Migration Paths](https://discourse.ubuntu.com/t/nvidia-gpudirect-over-infiniband-migration-paths/44425)
- NCCL：[环境变量文档（NCCL_DMABUF_ENABLE）](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- NVIDIA 驱动 595.84：[README: kernel open](https://download.nvidia.com/XFree86/Linux-x86_64/595.84/README/kernel_open.html)

<a id="2"></a>
## 2. 全貌

[↑ 返回目录](#toc)

<a id="2-1"></a>
### 2.1 两台机器与环境

| | server-0（client） | server-1（server） |
|---|---|---|
| 节点 | `rs-b300x8-pool-3gvac0` | `rs-b300x8-pool-3gvwb1` |
| `eth0` 地址（OOB 用） | 10.133.1.12 | 10.133.0.253 |
| 使用的 GPU | GPU0 | GPU7 |
| 使用的 RDMA 设备 | `mlx5_0`（网口 `fabric0`） | `mlx5_14`（网口 `fabric14`） |
| 使用的 GID | `gid[3]` = `fd02:0:400:52e::1` | `gid[3]` = `fd02:0:0:527::1` |

以上均为**实测**。下面这些事实决定了程序怎么写：

| 事实（实测） | 对程序的影响 |
|---|---|
| `nvidia-smi topo -m`：GPU0 与 NIC0、NIC1 同为 `PXB`；GPU7 与 NIC14、NIC15 同为 `PXB`（两台机器一致） | 让每张 GPU 用**离自己最近**的网卡：client 用 `mlx5_0`，server 用 `mlx5_14`（§3.2） |
| Pod 里 `/dev/infiniband/` 只有**偶数号**的 `uverbsN`（共 8 个），且只有偶数号网卡有对应的 `fabricN` 网口 | 只能用 `mlx5_0/2/4/…/14`；`mlx5_1` 这类奇数号设备在 sysfs 里看得到，但打不开 |
| 网卡是 400 Gb/s（4X NDR）、RoCE、`active_mtu = 4096`、`max_msg_sz = 1073741824`（1 GiB） | 本程序一条消息 = 一个 64 MiB 区域，远小于 1 GiB 上限 |
| GID 表里 `gid[1]` 是链路本地地址（`fe80::…`，RoCE v2），`gid[3]` 才是 `fd02:…` 的 ULA 地址 | 代码按"RoCE v2 且非链路本地"**自动选 GID** |
| 跨 rail 的 `ping6` 也通（server-0 的 `fabric0` → server-1 的 `fabric14`），往返约 0.13 ms | 两端可以各用各的最近网卡，不必强行用同一个 rail |
| `libibverbs` 头文件已在镜像里（rdma-core 39.0）；宿主机已加载 `nvidia_peermem`；`ulimit -l` 为 unlimited | 不用额外装包就能编译；GPU 显存能直接注册 |

<a id="2-2"></a>
### 2.2 与主机内存版的关系

两个版本用**同样的协议、同样的步骤、同一对网卡**（`mlx5_0` ↔ `mlx5_14`），只有"被注册、被 RDMA 读写的内存是什么"不同。所以 RDMA 通用部分直接沿用主机内存版：

| 内容 | 在哪里讲 |
|---|---|
| RDMA 对象（PD / MR / QP / CQ / WR / WC、lkey / rkey、GID） | [c-multi-node-cpu.md §2.1](c-multi-node-cpu.md#2-1) |
| OOB 上交换什么（`conn_info`、`'W'/'B'/'Q'`） | [c-multi-node-cpu.md §2.2](c-multi-node-cpu.md#2-2) |
| 什么算"连接" | [c-multi-node-cpu.md §2.3](c-multi-node-cpu.md#2-3) |
| QP 状态机（INIT → RTR → RTS）与参数 | [c-multi-node-cpu.md §2.4](c-multi-node-cpu.md#2-4) |
| 单边操作 WRITE / READ | [c-multi-node-cpu.md §2.5](c-multi-node-cpu.md#2-5) |
| 流量类别 `-t`（代码位置、READ 为什么两端都要设） | [c-multi-node-cpu.md §2.6](c-multi-node-cpu.md#2-6) |
| **CPU 与 GPU 的注册区别、GPU RDMA 的两种实现（peermem / dmabuf）** | **本篇 §1** |
| **程序里 GPU 特有的做法、GPU 与网卡的就近关系** | **本篇 §3** |
| **kernel 填充、`cudaMemcpy` 校验** | **本篇 §4.2、§4.3** |

**代码层面的差别**：

| | GPU 显存版（本篇） | 主机内存版 |
|---|---|---|
| 语言与依赖 | CUDA C++；`nvcc`；依赖 CUDA、`libibverbs` | 纯 C；`gcc`；只依赖 `libibverbs` |
| 内存来源 | `cudaMalloc` | `posix_memalign`（4096 字节对齐） |
| 填充数据 | GPU kernel（`fill_pattern`） | CPU 循环 |
| 校验方式 | `cudaMemcpy` 拷回主机再逐个比较 | 直接读主机内存比较，不需要拷贝 |
| 注册给网卡 | `ibv_reg_mr(GPU 指针)`（peermem）或 dmabuf | `ibv_reg_mr(主机指针)`，内核标准钉页路径，不需要额外模块 |
| 注册前的准备 | `cudaDeviceSynchronize()`、`SYNC_MEMOPS` | 先把每一页写一遍（避免缺页成本算进注册） |
| OOB、QP 状态机、RC WRITE / READ、GID 选择 | 完全相同 | 完全相同 |

<a id="2-3"></a>
### 2.3 运行前提：GPU 版额外需要什么

主机内存版需要的软件与条件（`libibverbs1`、`ibverbs-providers`、`uverbs` 设备、可锁定内存、RoCE v2 GID、网络互通等）见 [c-multi-node-cpu.md §1.3](c-multi-node-cpu.md#1-3)，GPU 版同样需要。**额外**需要：

| 软件 / 条件 | 作用 | 本环境实际情况 |
|---|---|---|
| CUDA Toolkit（`nvcc`） | 编译 `.cu`；链接 `-lcuda` 需要 CUDA 驱动库 | nvcc 13.0 |
| NVIDIA 驱动 | 运行 CUDA 程序 | 590.48.01（开放内核模块），8×B300 |
| 宿主机的 `nvidia_peermem` 内核模块 | peermem 方式下让 `ibv_reg_mr` 认得 GPU 指针 | 两台宿主机的 `/proc/modules` 里都有（**实测**） |
| 开放内核模块与较新的内核 | dmabuf 方式的前提 | 两台机器的驱动都是 "NVIDIA UNIX Open Kernel Module"，内核 6.12（**实测**） |
| Pod 里能看到 GPU | `cudaSetDevice`、`cudaMalloc` | 8 张 GPU 可见 |

**推断**：`nvidia_peermem` 缺失时，peermem 方式的 `ibv_reg_mr` 对 GPU 指针会失败（程序里的报错提示就是这个猜测），dmabuf 方式则不受影响；没有逐项去掉验证。

<a id="2-4"></a>
### 2.4 一次实验的时间线

```
server-1（GPU7，mlx5_14）                              server-0（GPU0，mlx5_0）
① cudaMalloc，填数据                                  ① cudaMalloc，填数据
② 注册 GPU 显存（peermem 或 dmabuf）                   ② 注册 GPU 显存
③ 创建 CQ、QP，切到 INIT                               ③ 创建 CQ、QP，切到 INIT
④ eth0 上 listen                  ◄──── TCP connect ──  ④ 连接 server 的 eth0 地址
⑤ 收 client 的连接信息、发自己的   ⇄  交换 {qpn, psn, gid, addr, rkey, len}
⑥ QP：INIT → RTR → RTS                                 ⑥ QP：INIT → RTR → RTS
                                                       ⑦ RDMA WRITE：GPU0 [0,n) → GPU7 [0,n)
                                                          等 CQ 出现完成项
                                  ◄──── OOB 'W' ────────  ⑧ 通知 server 去校验
⑨ 校验 GPU7 [0,n)，回 'K'/'X'     ───── OOB 'K' ───────►
                                                       ⑩ RDMA READ：GPU7 [n,2n) → GPU0 [n,2n)
                                                          本端校验
                                                       ⑪ 带宽：连续多次 WRITE、READ
                                  ◄──── OOB 'B' ────────  ⑫ 带宽测试后再让 server 校验一次
                                  ◄──── OOB 'Q' ────────  ⑬ 结束
```

**两条通道别混**：OOB（`eth0`）只传"QP 号、GID、地址、rkey"和"写完了 / 校验结果"这类控制信息，约 1 KiB；数据走 RDMA，两端网卡直接在两块**显存**之间搬运，本实验约 1.3 GiB（§5.5 实测）。

<a id="2-5"></a>
### 2.5 与单机 CUDA IPC 的对照

见 [c-single.md §3.4](c-single.md#3-4)：

| | 单机：CUDA IPC | 多机：RDMA（本篇） |
|---|---|---|
| 共享显存的凭证 | IPC 句柄（64 字节） | MR 的 `addr` + `rkey` |
| 凭证怎么传 | 任意主机侧通道（文件、pipe、TCP） | **OOB（`eth0` 上的 TCP）** |
| 谁来读写对端显存 | 访问方的 **GPU kernel**（经 NVLink） | 本端的**网卡**（经 RoCE），GPU 上没有任何 kernel 参与搬运 |
| 跨机器 | **不能**（句柄只在同一台机器内有效，文档说法，未测） | 能 |

<a id="3"></a>
## 3. 程序里 GPU 特有的做法

[↑ 返回目录](#toc)

<a id="3-1"></a>
### 3.1 注册显存：程序里怎么写

`ib_setup` 里按 `-m` 的选择走两条分支之一（原理见 §1.4、§1.5）：

```cpp
unsigned* buf;  cudaMalloc(&buf, total);                          // GPU 显存（设备指针）
// -m peermem（默认）：
ib.mr = ibv_reg_mr(ib.pd, buf, total, acc);
// -m dmabuf：
cuMemGetHandleForAddressRange(&ib.dmabuf_fd, (CUdeviceptr)buf, total, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
ib.mr = ibv_reg_dmabuf_mr(ib.pd, 0, total, (uint64_t)buf, ib.dmabuf_fd, acc);
```

**要点**：

1. **`buf` 是设备指针**：在 64 位进程的统一虚拟地址（UVA）下，它是全局唯一的虚拟地址，所以 `ibv_reg_mr` 才能直接收它，OOB 里交换的 `addr` 也是这个地址。**CPU 不能直接解引用它**——要读必须 `cudaMemcpy`（校验时就是这样做的）；
2. **dmabuf 分支的参数**：`offset = 0`（从 fd 对应缓冲的起点）；`iova` 取显存指针本身，所以对端用的远端地址与 peermem 方式一致（仍是这个 UVA 地址）；`ib.dmabuf_fd` 要在注销 MR 之后自己 `close`；
3. **注册之后，网卡可以直接读写这块显存**；数据**不经过主机内存**，这一点是**推断**（由"直接注册 GPU 指针"这一机制得出，本篇没有单独测量主机内存的流量）；
4. **`SYNC_MEMOPS`**：程序对这块显存设了 `CU_POINTER_ATTRIBUTE_SYNC_MEMOPS`，保证对这块显存的同步类内存操作（如 `cudaMemcpy`、`cudaMemset`）在返回前对 RDMA 访问都已可见（公开资料的说法）。调用 `cuPointerSetAttribute(&one, attr, ptr)` 时，第一个参数是**属性值**，最后一个才是目标显存地址；
5. **时序要点**：填充数据后必须 `cudaDeviceSynchronize()`，**再**把地址和 rkey 交给对端；否则对端读到的可能是填充了一半的数据；
6. **注册耗时**：程序启动时会打印注册方式与耗时（128 MiB：peermem 约 2.1 ms，dmabuf 约 0.23 ms，§5.4 表 E）。

<a id="3-2"></a>
### 3.2 GPU 与网卡的就近关系

`nvidia-smi topo -m` 里，GPU0 与 NIC0、NIC1 同为 `PXB`，GPU7 与 NIC14、NIC15 同为 `PXB`（**实测**，两台机器一致）。`PXB` 表示两者之间经过多个 PCIe 桥、但不经过 CPU（PCIe Host Bridge）。

- **做法**：让每张 GPU 用离它最近的网卡——GPU0 用 `mlx5_0`，GPU7 用 `mlx5_14`。因为 Pod 里只有偶数号网卡可用（§2.1），所以每张 GPU 在 `PXB` 的两块网卡里取偶数号那块。
- **为什么**：GPUDirect RDMA 时，网卡经 PCIe 直接访问 GPU 显存。网卡离 GPU 越近，这条 PCIe 路径越短；若不在同一 PCIe 交换机下，访问可能要绕过 CPU 的根复合体。这是公开资料的一般性说法。
- **本篇没有测过**"换成不就近的网卡带宽会怎样"，所以不能说这个选择对带宽有多大影响。这也是 §5.7 里 30 GB/s 平台期**未查明**原因的方向之一。

**换网卡与 GPU**：`-d` 指定网卡、`-g` 指定 GPU，server 端要对应修改，例如 `./client-gpu -a <IP> -d mlx5_2 -g 1`（GPU1 附近的网卡要先用 `nvidia-smi topo -m` 确认，这里只是举例）。

<a id="4"></a>
## 4. 程序：server-gpu.cu 与 client-gpu.cu

[↑ 返回目录](#toc)

<a id="4-1"></a>
### 4.1 总览与命令行参数

两个程序各自**独立、自包含**（TCP 小工具、ibverbs 初始化等有重复，为的是每个文件能单独读懂）。源码在 `p2p/` 目录，含详细中文注释。

| | `server-gpu.cu`（401 行） | `client-gpu.cu`（488 行） |
|---|---|---|
| 运行在 | server-1 | server-0 |
| GPU / RDMA 设备（默认） | GPU7 / `mlx5_14` | GPU0 / `mlx5_0` |
| OOB | 在 `eth0` 的 IPv4 地址上**监听** | 连接 server 的 `eth0` 地址 |
| 角色 | 被动：准备显存、注册、告知地址与 rkey；响应校验请求 | 主动：发起 WRITE / READ、校验、测带宽 |
| 生命周期 | 处理完一个 client 后退出（一次性） | 跑完退出 |

**命令行参数**（`server` / `client` 共有的用同一个字母）：

| 参数 | 含义 | server 默认 | client 默认 |
|---|---|---|---|
| `-a` | server 的 `eth0` IP | — | **必填** |
| `-p` | OOB 端口 | 5555 | 5555 |
| `-d` | RDMA 设备名 | `mlx5_14` | `mlx5_0` |
| `-g` | GPU 编号 | 7 | 0 |
| `-x` | GID 下标 | 自动选择 | 自动选择 |
| `-s` | 每个区域的大小（MiB）。**两端必须相同**，不一致会在交换信息时报错退出 | 64 | 64 |
| `-t` | 流量类别（traffic_class），**两端都要设**（见 [c-multi-node-cpu.md §2.6](c-multi-node-cpu.md#2-6)） | 0 | 0 |
| `-i` | OOB 所用的网口名（只绑它的 IPv4 地址） | `eth0` | — |
| `-m` | GPU 显存注册给网卡的方式：`peermem`（默认）或 `dmabuf`（§1.4、§1.5）。两端可各自选择，互不影响 | peermem | peermem |
| `-n` | 带宽测试的迭代数 | — | 20 |
| `-w` | 带宽测试的并发深度（同时在飞的操作数） | — | 8 |

与主机内存版相比，GPU 版多 `-g`（选 GPU）和 `-m`（选注册方式），**没有** `-o`（逐次计时探查）和 `-W`（逐窗口带宽）这两个诊断选项（这也是附录 B 里的一项后续工作）。

<a id="4-2"></a>
### 4.2 显存布局与数据模式

每端在自己的 GPU 上分配一块 `2 × region` 字节的 buffer（默认 `region = 64 MiB`，即每个区域 n = 16M 个 `unsigned`），整块注册给网卡。

| | 区域 `[0, n)` | 区域 `[n, 2n)` |
|---|---|---|
| **client（GPU0）** | 要**写给对端**的数据 A：`A[i] = i*3 + 1`（RDMA WRITE 的来源） | **读回来**的落点，初始为 0（RDMA READ 的目标） |
| **server（GPU7）** | 留给 client **写入**，初始为 0（RDMA WRITE 的目标） | 给 client **读取**的数据 B：`B[i] = i*5 + 2`（RDMA READ 的来源） |

- **数据模式用 `unsigned`**：每个下标的值唯一，所以数据错位也能被发现（`0_to_7.cu` 用 `float` 时有后半段相邻下标共用同一个值的局限，见 [c-single.md §3.1](c-single.md#3-1)，这里避开了）。`buf + n` 是"跳过 n 个 `unsigned`"（= `region` 字节）；
- **填充用 GPU kernel**：`fill_pattern` 是 grid-stride 循环（`block = 256`，`grid` 上限 4096），每个线程按步长处理多个元素，所以 kernel 的启动规模不随数据量无限增长（代码事实）；
- **校验的方法**：把 GPU 上的数据 `cudaMemcpy` 回主机，逐个比较。注意——**数据是网卡直接写进 GPU 显存的，server 的 CPU 全程没有碰过它**，校验时才第一次读。

<a id="4-3"></a>
### 4.3 关键代码骨架

```cpp
// ===== server-gpu.cu（server-1，GPU7）=====
cudaSetDevice(7);  cudaMalloc(&buf, 2*region);
cuPointerSetAttribute(&one, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)buf);
cudaMemset(buf, 0, region);  launch_fill(buf + n, n, 5u, 2u);     // [0,n)=0；[n,2n)=B
cudaDeviceSynchronize();                                          // ① 数据先就绪
ib_setup(ib, "mlx5_14", gid_arg, buf, 2*region, use_dmabuf);      // ② 开设备、注册 GPU 显存（peermem 或 dmabuf）、建 QP → INIT
bind(eth0 的 IPv4:5555) → listen → accept                         // ③ OOB
recv_all(cs, &theirs, sizeof theirs);  send_all(cs, &mine, sizeof mine);   // ④ 交换 ConnInfo
ib_connect(ib, theirs, tc);                                       // ⑤ QP → RTR → RTS
for (;;) {                                                        // ⑥ 只响应校验请求
  recv_all(cs, &cmd, 1);
  if (cmd == 'W' || cmd == 'B') { bad = count_bad(buf, n, 3u, 1u); send_all(cs, bad ? "X" : "K", 1); }
  if (cmd == 'Q') break;
}

// ===== client-gpu.cu（server-0，GPU0）=====
cudaSetDevice(0);  cudaMalloc(&buf, 2*region);
launch_fill(buf, n, 3u, 1u);  cudaMemset(buf + n, 0, region);     // [0,n)=A；[n,2n)=0
cudaDeviceSynchronize();
ib_setup(ib, "mlx5_0", gid_arg, buf, 2*region, use_dmabuf);
connect(server 的 eth0 IP:5555)  /* 没起来就每 100 ms 重试，最多约 30 秒 */
send_all(s, &mine, sizeof mine);  recv_all(s, &theirs, sizeof theirs);
ib_connect(ib, theirs, tc);

post_op(ib, IBV_WR_RDMA_WRITE, l_base, r_base, theirs.rkey, region, 1);   // WRITE
wait_cq(ib, 1);
send_all(s, "W", 1);  recv_all(s, &ack, 1);                       // OOB：通知 server 校验

post_op(ib, IBV_WR_RDMA_READ, l_base + region, r_base + region, theirs.rkey, region, 2);   // READ
wait_cq(ib, 1);
bad = count_bad(buf + n, n, 5u, 2u);                              // 本端校验（cudaMemcpy 拷回再比较）

run_bw(ib, IBV_WR_RDMA_WRITE, ...);  run_bw(ib, IBV_WR_RDMA_READ, ...);   // 带宽
send_all(s, "Q", 1);
```

`run_bw` 的做法：连续发起 `iters` 次操作，始终保持最多 `depth` 个在飞，计时从第一次 post 到最后一次完成。`ib_setup` / `ib_connect` / `post_op` / `wait_cq` 的其余部分与主机内存版相同（见 [c-multi-node-cpu.md §2](c-multi-node-cpu.md#2)）。

<a id="5"></a>
## 5. 实验与实测

[↑ 返回目录](#toc)

<a id="5-1"></a>
### 5.1 准备与编译

下面的命令都在 nuc 上执行。`kubectl cp` 要求目标目录已存在，所以这里用"先建目录、再用 stdin 重定向"的方式拷贝（**实测可用**）：

```bash
cd /home/ubuntu/work/nccl/examples/p2p
kubectl exec -i server-0 -- bash -c "mkdir -p /tmp/mn && cat > /tmp/mn/client-gpu.cu" < client-gpu.cu
kubectl exec -i server-1 -- bash -c "mkdir -p /tmp/mn && cat > /tmp/mn/server-gpu.cu" < server-gpu.cu

kubectl exec server-0 -- bash -lc 'cd /tmp/mn && nvcc -O3 -arch=native client-gpu.cu -o client-gpu -libverbs -lcuda'
kubectl exec server-1 -- bash -lc 'cd /tmp/mn && nvcc -O3 -arch=native server-gpu.cu -o server-gpu -libverbs -lcuda'
```

**编译参数**（两个程序相同，只是文件名不同）：

| 参数 | 含义 |
|---|---|
| `nvcc` | CUDA 编译器：把 `.cu` 里的主机代码交给 gcc、设备代码（kernel）自己编译 |
| `-O3` | 主机代码的优化级别 3（本程序的瓶颈在 RDMA 与网卡，优化级别不影响带宽） |
| `-arch=native` | 针对本机 GPU 的架构生成设备代码（这里是 B300），不需要手写 `sm_xx` |
| `client-gpu.cu` / `server-gpu.cu` | 源文件 |
| `-o client-gpu` / `-o server-gpu` | 输出的可执行文件名 |
| `-libverbs` | 链接 RDMA 用户态库 `libibverbs`（`-l ibverbs` 的连写） |
| `-lcuda` | 链接 CUDA **驱动 API** 库 `libcuda`（用到了 `cuPointerSetAttribute` 与 `cuMemGetHandleForAddressRange`；`cudaMalloc` 等运行时 API 不需要它） |

- **`-libverbs` 与 `-lcuda` 要写在源文件之后**（链接器按顺序处理）；
- **调试版**：`nvcc -g -G -O0 -arch=native client-gpu.cu -o client-gpu.debug -libverbs -lcuda`（`server-gpu` 同理）。`-g` 生成主机调试信息，`-G` 生成设备（kernel）调试信息并关闭设备优化，用 `cuda-gdb` 调试；
- 两个程序都**一次编译通过、没有错误**（**实测**）；镜像里已有 `libibverbs` 头文件和库，不需要额外安装（**实测**，rdma-core 39.0）。

<a id="5-2"></a>
### 5.2 运行

**server 要先启动**，等它打印"OOB 监听"之后再启动 client。先查 server-1 的 `eth0` 地址：

```bash
kubectl get pod server-1 -o jsonpath='{.status.podIP}'      # 本次实测为 10.133.0.253
```

```bash
# 终端 1：server-1（GPU7）
kubectl exec -it server-1 -- bash -lc 'cd /tmp/mn && ./server-gpu -t 104'

# 终端 2：server-0（GPU0）。-a 填 server-1 的 eth0 地址
kubectl exec -it server-0 -- bash -lc 'cd /tmp/mn && ./client-gpu -a 10.133.0.253 -t 104'
```

**换 dmabuf 注册方式**：加 `-m dmabuf`（两端各自选择，可以只加一端），例如 `./server-gpu -t 104 -m dmabuf`、`./client-gpu -a 10.133.0.253 -t 104 -m dmabuf`。

**推荐 `-t 104`，两端都要设**（本集群上运行 NCCL 也必须 `NCCL_IB_TC=104`）。GPU 版在 `-t 0` 与 `-t 104` 下数字相同（§5.4），但主机内存版的 READ 对 TC 非常敏感（[c-multi-node-cpu.md §5.4](c-multi-node-cpu.md#5-4)），所以统一用 `-t 104` 最稳妥。

<a id="5-3"></a>
### 5.3 正确性

2026-11-01 的一次完整运行，`-t 104`、默认参数（默认 `-m peermem`，**实测**）。**client 输出**：

```
[client] GPU0 上 buf = 0x7f8c38000000，总 128 MiB（每区域 64 MiB）
[client] RDMA 设备 mlx5_0：active_mtu=4096，max_msg_sz=1073741824，使用 gid[3]
[client] 本端 GID = fd02:0:400:52e::1
[client] 已注册 GPU 显存（peermem）：注册耗时 2.112 ms
[client] OOB 已连接 10.133.0.253:5555
[client] 对端 GID = fd02:0:0:527::1
[client] 对端显存：addr=0x7fa688000000  rkey=0x212e2  len=134217728
[client] QP 已就绪（RTS）
[client] RDMA WRITE  GPU0 → 对端 GPU  64 MiB：耗时 2.60 ms；对端校验 PASS
[client] RDMA READ   对端 GPU → GPU0  64 MiB：耗时 2.43 ms；本端校验：不匹配 0 个 -> PASS
[client] 带宽 WRITE：20 × 64 MiB，并发深度 8：0.05 s -> 29.8 GB/s（238 Gb/s）
[client] 带宽 READ ：20 × 64 MiB，并发深度 8：0.04 s -> 30.0 GB/s（240 Gb/s）
[client] 带宽测试之后本端再校验：不匹配 0 个 -> PASS
[client] 结束，退出码 0
```

**server 输出**：

```
[server] GPU7 上 buf = 0x7fa688000000，总 128 MiB（每区域 64 MiB）
[server] RDMA 设备 mlx5_14：active_mtu=4096，max_msg_sz=1073741824，使用 gid[3]
[server] 本端 GID = fd02:0:0:527::1
[server] 已注册 GPU 显存（peermem）：addr=0x7fa688000000  rkey=0x212e2  len=134217728；注册耗时 2.166 ms
[server] OOB 监听 10.133.0.253:5555（eth0），等待 client ...
[server] 对端 GID = fd02:0:400:52e::1
[server] QP 已就绪（RTS），等待 client 的读写与校验请求
[server] 校验 client 写进来的数据（RDMA WRITE 之后）：不匹配 0 个 -> PASS
[server] 校验 client 写进来的数据（带宽测试之后）：不匹配 0 个 -> PASS
[server] 结束，退出码 0
```

结论：

1. **RDMA WRITE 正确**：GPU0 写进 GPU7 的 64 MiB 数据，由 **server** 校验，不匹配 0 个；
2. **RDMA READ 正确**：从 GPU7 读回 GPU0 的 64 MiB 数据，由 **client** 校验，不匹配 0 个；
3. **带宽测试之后两端各再校验一次**，仍然 PASS，说明连续多次读写没有破坏数据；
4. **第一次 WRITE / READ 的耗时**约 2.60 ms / 2.43 ms（64 MiB），折合约 26 / 28 GB/s，包含首次操作的开销；
5. **注册耗时**（128 MiB，peermem）：client 2.11 ms、server 2.17 ms；换 `-m dmabuf` 约 0.23 ms（§5.4 表 E）；
6. 两个进程**退出码都是 0**；server 的 `buf` 地址（`0x7fa688000000`）也正是 client 拿到的 `addr`，印证 OOB 交换无误。

<a id="5-4"></a>
### 5.4 带宽

[↑ 返回目录](#toc)

**口径**：单个 QP；一条消息 = 一个完整的 64 MiB 区域；GB/s = 总字节数 / 从第一次 post 到最后一次完成的时间；表 A–C 为 `-n 100`（2026-10-31，默认 `-t 0`，表 C 对比两种 TC），表 D、E 为 2026-11-01 的 `-t 104` 运行。

**A. 稳定性**——同一配置（`-w 8`，TC=0）重复 3 次：

| 次 | WRITE（GB/s） | READ（GB/s） |
|---|---|---|
| 1 | 29.8 | **17.9** |
| 2 | 29.8 | 30.0 |
| 3 | 29.8 | 30.0 |

**B. 并发深度的影响**（TC=0）：

| 并发深度 `-w` | WRITE（GB/s） | READ（GB/s） |
|---|---|---|
| 1 | 29.0 | 29.8 |
| 2 | 29.8 | 30.0 |
| 4 | 29.8 | 30.0 |
| 16 | 29.8 | 30.0 |

**C. 流量类别 TC**（两端同时设置，`-w 8`）：

| TC | WRITE（GB/s） | READ（GB/s） |
|---|---|---|
| 0 | 29.8 | **22.1** |
| 104 | 29.8 | 30.0 |

**D. `-t 104` 的两次完整运行**（2026-11-01，`-w 8`）：

| 迭代数 `-n` | 首次 WRITE | 首次 READ | WRITE（GB/s） | READ（GB/s） |
|---|---|---|---|---|
| 20 | 2.60 ms | 2.42 ms | 29.8 | 30.0 |
| 100 | 2.60 ms | 2.43 ms | 29.8 | 30.0 |

**E. 注册方式：peermem 与 dmabuf**（2026-11-01，`-t 104`，`-n 100 -w 8`；用 `-m` 选择，见 §4.1）：

| 组合（server / client） | 运行次数 | 注册耗时（128 MiB，client / server） | 首次 WRITE / READ | WRITE（GB/s） | READ（GB/s） | 校验 |
|---|---|---|---|---|---|---|
| peermem / peermem | 2 | 2.08–2.11 ms / 2.10–2.13 ms | 2.60–2.61 / 2.43 ms | 29.8 | 30.0 | 全部 PASS |
| dmabuf / dmabuf | 3 | **0.23 ms** / **0.24–0.25 ms** | 2.60 / 2.42–2.43 ms | 29.8 | 30.0 | 全部 PASS |
| dmabuf / peermem（混合） | 1 | 2.13 ms（peermem）/ 0.245 ms（dmabuf） | 2.61 / 2.43 ms | 29.8 | 30.0 | 全部 PASS |
| peermem / dmabuf（混合） | 1 | 0.228 ms（dmabuf）/ 2.10 ms（peermem） | 2.61 / 2.43 ms | 29.8 | 30.0 | 全部 PASS |

读法：

1. **dmabuf 路线能完成完整的跨机 WRITE / READ**，校验全部 PASS，退出码 0；
2. **两端可以混用**：一端 peermem、一端 dmabuf 也能正常通信（注册方式只影响本端如何把显存交给网卡）；
3. **注册开销差一个数量级**：128 MiB 的注册，dmabuf 约 0.23–0.25 ms（含"导出 + 注册"），peermem 约 2.1 ms，约 9 倍；
4. **传输带宽、首次耗时都没有差别**：两种方式下都是 WRITE 29.8、READ 30.0 GB/s——所以下面说的 30 GB/s 平台期**与注册方式无关**。

局限：每种组合只跑了 1–3 次；都是单 QP、`-n 100`、一条 64 MiB 消息；没有测逐窗口曲线，也没有测 VMM 分配的显存。`nvidia_peermem` 模块在这两台机器上始终是加载着的，所以"dmabuf 不依赖 `nvidia_peermem`"是机制上的结论，**没有在卸载模块的环境里验证**。

**怎么读这些数据**：

- **WRITE 非常稳定，约 29.8 GB/s（238 Gb/s）**：不随并发深度和 TC 变化。并发深度为 1（同一时刻只有一条 64 MiB 消息在飞）时也只略低（29.0），说明这不是"延迟受限"——一条大消息已经把能用的带宽用满了；
- **READ 通常是 30.0 GB/s（240 Gb/s）**，但**表 A–C 的 9 次读测量中有 2 次明显偏低**（17.9 和 22.1），两次都发生在 TC=0 的配置下；而 TC=0 的另外多次测量又都是 30.0（`-n 20` 的几次运行，READ 也都是 30.0）。这两次偏低**可能**与 TC 有关：主机内存版实验里，`-t 0` 下 READ 明显不稳、`-t 104` 下稳定（§5.7）；但 GPU 版在 `-t 104` 下只有表 D、E 这样少量样本、没有复现过偏低，所以对 GPU 版仍**不能下结论**，偏低的原因**未查明**；
- **约 30 GB/s 对应 240 Gb/s，只有网卡标称 400 Gb/s（4X NDR）的 60%**。这个平台期的原因**未查明**。换用 dmabuf 注册方式带宽不变（表 E），所以与注册方式无关；同一对网卡上换成主机内存能跑到约 44 GB/s，因此可以**排除网卡或 fabric 限速**（§5.7）；更可能与 GPU 显存一侧的访问路径有关，但这是推断，未验证。本篇没有做多 QP、改消息大小、查 PFC / ECN 配置这些实验。

<a id="5-5"></a>
### 5.5 数据走了哪条路：字节计数器

[↑ 返回目录](#toc)

"数据走的是 RDMA 而不是 `eth0`"需要证据。方法与主机内存版相同（取一次运行前后 `ethtool -S` 的 RDMA vport 计数器与 `eth0` 字节数的差值，见 [c-multi-node-cpu.md §5.5](c-multi-node-cpu.md#5-5)；`ethtool` 需要在 Pod 里装，`apt-get install ethtool`，只影响容器层）。

一次运行（`-n 20`：共 21 次 WRITE + 21 次 READ，每次 64 MiB，即各 1344 MiB 净载荷）前后的差值（**实测**）：

| 计数器 | server-0（client，`fabric0`） | server-1（server，`fabric14`） |
|---|---|---|
| RDMA 发出 `tx_vport_rdma_unicast_bytes` | **1375.326 MiB** | **1369.850 MiB** |
| RDMA 收到 `rx_vport_rdma_unicast_bytes` | **1369.850 MiB** | **1375.326 MiB** |
| `eth0` 发出（OOB） | 0.001 MiB | 0.001 MiB |
| `eth0` 收到（OOB） | 0.001 MiB | 0.001 MiB |
| `fabric` 内核网口 `tx_bytes` | 0.000 MiB | 0.000 MiB |

读法：

1. **RDMA 流量约 1.3 GiB，与理论净载荷（WRITE 1344 MiB、READ 1344 MiB）相符**，多出的约 2% 应是协议头、确认包和 READ 请求（**推断**，未拆分）；
2. **两端互相印证**：client 发出的 1375.326 MiB 恰好等于 server 收到的，client 收到的 1369.850 MiB 恰好等于 server 发出的；
3. **`eth0` 上只有约 1 KiB**——OOB 确实只传控制信息；
4. **`fabric` 的内核网口计数为 0**——RDMA 流量绕过了内核网络栈。

**这些计数器能证明什么、不能证明什么**：能证明数据确实经 RoCE 的 RDMA 通道传输、`eth0` 上没有数据；**不能**单独证明数据没有经过主机内存——那一点是从"直接注册 GPU 指针"这一机制推断出来的（§3.1）。

<a id="5-6"></a>
### 5.6 与单机 NVLink 的对照

[↑ 返回目录](#toc)

| | 单机 NVLink（[c-single.md §3.2](c-single.md#3-2)） | 多机 RoCE RDMA（本篇） |
|---|---|---|
| 路径 | GPU0 ↔ GPU7，经 NVSwitch | server-0 的 GPU0 ↔ server-1 的 GPU7，经两块网卡与以太网 fabric |
| 谁搬数据 | Copy Engine（`cudaMemcpyPeer`） | 网卡（RDMA WRITE / READ） |
| 实测带宽 | 779.4 GB/s | WRITE 约 29.8 GB/s、READ 通常约 30.0 GB/s |
| 比值 | — | NVLink 约是 RDMA 的 26 倍 |

两者的测量方法不同（一个是 CUDA 事件计时的拷贝，一个是 CPU 时钟计时的 RDMA），数字只用来说明量级差异，不是严格的同口径对比。

<a id="5-7"></a>
### 5.7 与主机内存（CPU）版的对照

[↑ 返回目录](#toc)

两个版本之间**只有"内存类型"这一个变量不同**（见 §2.2），可以同口径对照。

**实测对照**（`-n 100`，64 MiB 区域，单 QP；主机内存版数据来自 [c-multi-node-cpu.md §5.4](c-multi-node-cpu.md#5-4)；GPU 版两种注册方式带宽相同，见表 E）：

| | GPU 显存版（`-t 0` 与 `-t 104` 数字相同） | 主机内存版（`-t 104`） | 主机内存版（`-t 0`） |
|---|---|---|---|
| `ibv_reg_mr`（128 MiB） | peermem 约 2.1 ms；dmabuf 约 0.23 ms | 约 0.6–0.75 ms | 同左 |
| 首次 WRITE | 2.60 ms | 1.52 ms（偶尔 3.4–6.2 ms） | 同左 |
| 首次 READ | 2.42 ms | **1.55 ms** | **12.7–103 ms，不稳** |
| READ 带宽 | 30.0 GB/s | **43.8 GB/s**（稳定） | 24.5–24.7 GB/s |
| WRITE 带宽（深度 8） | 29.8 GB/s | 22.1 GB/s（短测试，见下） | 22.1 GB/s |
| WRITE 带宽（深度 1） | 29.0 GB/s | **44.1 GB/s**（偶尔掉到 22） | 44.1 GB/s（偶尔掉到 22） |
| 对 TC 敏感吗 | 不敏感 | READ 敏感，WRITE 不敏感 | — |
| 稳定性 | 交替 5 次完全相同（29.8 / 30.0） | READ 稳定；WRITE 有爬升与偶发双峰 | READ 不稳 |

GPU 一列是同一轮里与主机内存版**交替**各跑 5 次（`-t 0`）以及 `-t 104` 下的几次测量；GPU 版的首次 WRITE 为 2.60–2.61 ms、首次 READ 为 2.42–2.43 ms，两种 TC 下一致，样本有限。

**怎么比才公平**：

- **READ 是最干净的对照**：主机内存版在 `-t 104` 下从第一次起恒为 43.8 GB/s，GPU 版恒为 30.0 GB/s；
- **WRITE 的深度 8 一栏不能直接比**：主机内存版深度 ≥ 2 时 WRITE 从约 22 GB/s 起步并逐渐爬升，`-n 6000` 才平均到 43.3 GB/s，所以 `-n 100` 的 22.1 是**被低估**的；深度 1 时 44.1 GB/s 才是稳态。GPU 版**没有做过逐窗口曲线**，不知道它在更长的运行里是否也有爬升，所以 GPU 版的 29.8 / 29.0 同样只代表 `-n 100`；
- 单 QP、单条 64 MiB 消息，不是网卡能力的上限（多 QP、多消息尺寸都没测）。

**对照得出的结论**：

1. **GPU 显存版的"约 30 GB/s 平台期"不是网络上限**：同一对网卡、同一条链路，主机内存能跑到约 44 GB/s（350 Gb/s，约为网卡标称 400 Gb/s 的 88%），所以可以**排除"网卡或 fabric 限速"**这一类解释。它更可能与 **GPU 显存这一侧的访问路径**（网卡经 PCIe 对等访问 GPU 显存）有关。**这是推断，没有验证**——没有测不同的 GPU–网卡就近关系（§3.2），也没有测 GPU 版的逐窗口曲线；换用 dmabuf 注册方式带宽不变（§5.4 表 E），说明平台期与注册方式无关；
2. **GPU 版更稳，但上限更低**：GPU 版数字几乎不变（29.8 / 30.0），主机内存版的上限更高（约 44），但 WRITE 有爬升与偶发双峰，READ 在 `-t 0` 下不稳；
3. **主机内存版的 READ 对 TC 敏感，GPU 版不敏感**：原因**未查明**。这也意味着 §5.4 里 GPU 版 READ 偶发偏低的 2 次（17.9、22.1 GB/s，均为 `-t 0`）**可能**与此同类；但交替的 5 次 `-t 0` 与 `-t 104` 下的 GPU 测量都没有再出现偏低，所以对 GPU 版**不能下结论**；
4. **RDMA 协议开销与内存类型无关**：同样的 `-n 20` 运行，两个版本的 RDMA 字节计数器逐位相同（1375.326 / 1369.850 MiB，见 §5.5）。

**这对使用的启示**：在本集群上，主机内存 RDMA 的流量类别设置（本篇推荐的 `-t 104`，NCCL 对应 `NCCL_IB_TC=104`）影响很大；GPU 显存 RDMA 在本篇的测量范围内不受影响。无论哪种内存，**测稳态带宽时迭代数都要足够大**。

<a id="5-8"></a>
### 5.8 注册探查实验 reg_bench

[↑ 返回目录](#toc)

§1 里的注册耗时、VMM、BAR1、生命周期等实测，都来自探查程序 `p2p/reg_bench.cu`（235 行，含中文注释）。**单机即可，不需要第二台机器**，也不收发任何数据：

```bash
nvcc -O3 -arch=native reg_bench.cu -o reg_bench -libverbs -lcuda

./reg_bench attr                      # 设备属性：GPUDirect RDMA / dmabuf / VMM / flush / ordering
./reg_bench lat                       # 注册与注销耗时：CPU 未触摸、CPU 已触摸、peermem、dmabuf（4 种 × 4 个大小 × 5 次）
./reg_bench lat mlx5_0 dmabuf         # 只测名字里含 "dmabuf" 的那一种（模式片段可以是 peermem、dmabuf、CPU 等）
./reg_bench bar1                      # 注册前后的 BAR1 占用
./reg_bench vmm                       # cuMemCreate 分配的显存能否经 peermem / dmabuf 注册
./reg_bench life                      # 注册之后 cudaFree 会怎样
```

默认使用设备 `mlx5_0` 与 GPU0；设备名可以作为第二个参数。`lat` 会反复分配最大 4 GiB 的主机内存和显存，耗时较长（未精确计时；完整运行在 280 秒的超时内完成）。所有测量都**只做注册与注销，不传数据**。

<a id="6"></a>
## 6. 速查：命令与坑

[↑ 返回目录](#toc)

```bash
# 编译（在 server-0 / server-1 各自的 /tmp/mn 下）
nvcc -O3 -arch=native client-gpu.cu -o client-gpu -libverbs -lcuda
nvcc -O3 -arch=native server-gpu.cu -o server-gpu -libverbs -lcuda

# 运行
./server-gpu -t 104                           # server-1，先启动；监听 eth0:5555（推荐 -t 104，两端都要设）
./client-gpu -a <server-1 的 eth0 IP> -t 104  # server-0

# 调参
./client-gpu -a <IP> -n 100 -w 8 -t 104       # 迭代 100 次、并发深度 8、traffic_class 104（server 也要加 -t 104）
./client-gpu -a <IP> -d mlx5_2 -g 1           # 换网卡与 GPU（server 端对应修改）
./client-gpu -a <IP> -t 104 -m dmabuf         # 用 dmabuf 注册显存（两端各自选择，server 可单独加 -m dmabuf）

# 取证：RDMA 字节计数器（累计值，前后相减）
ethtool -S fabric0 | grep rdma_unicast_bytes

# 注册探查（单机）
./reg_bench attr | lat | bar1 | vmm | life
```

| # | 坑 / 注意 | 说明 |
|---|---|---|
| 1 | 用 `kubectl exec` 启动后台进程会卡住 | **实测遇到**：`kubectl exec server-1 -- bash -lc '... setsid nohup ./server-gpu > log 2>&1 < /dev/null & ...; echo started'`，`started` 已打印，但这条 exec 一直不返回，把整段脚本卡死。可靠的做法是把 **`kubectl exec ... ./server-gpu` 本身**放到 nuc 的后台（`&`），再轮询日志里的"OOB 监听"（2026-11-01 的几次运行就是这样做的）。卡住的具体原因未隔离 |
| 2 | server 要先启动 | client 对连接做了重试（每 100 ms，最多约 30 秒），所以先后顺序稍有偏差也能成功；但 server 完成 CUDA 初始化、注册显存需要时间，要等到"OOB 监听"出现再认为就绪 |
| 3 | 只能用偶数号网卡 | Pod 里只有 `mlx5_0/2/…/14` 有 `/dev/infiniband/uverbsN`；奇数号设备在 sysfs 里看得到，但 `ibv_open_device` 会失败 |
| 4 | 让 GPU 用离它最近的网卡 | 看 `nvidia-smi topo -m`：与 GPU 标 `PXB` 的 NIC。本机 GPU0 ↔ NIC0/1，GPU7 ↔ NIC14/15；再取其中偶数号那块（`mlx5_0`、`mlx5_14`）（§3.2） |
| 5 | GID 要选对 | `gid[1]` 是链路本地地址，不能跨机；要用 `fd02:…` 那条 RoCE v2（本机是 `gid[3]`）。程序默认自动选择，`-x` 可手工指定 |
| 6 | 两端的 `-s` 必须一致 | 否则在交换连接信息时报"两端区域大小不一致"后退出 |
| 7 | WRITE 对对端是"无声"的 | 对端不知道数据已到，必须经 OOB 通知它去校验 |
| 8 | 填完数据要先同步再交出地址 | `cudaDeviceSynchronize()` 之后再把 `addr` / `rkey` 发给对端（§3.1） |
| 9 | peermem 方式要靠宿主机的 `nvidia_peermem` | `ibv_reg_mr` 对 GPU 指针失败时，先检查 `/proc/modules` 里是否有 `nvidia_peermem`；也可以改用 `-m dmabuf`（不依赖该模块，这一点是**推断**，没有卸载 peermem 验证） |
| 10 | 带宽口径 | 单 QP、一条消息 = 64 MiB、CPU 时钟计时，是**这个配置**的数字，不是网卡能力的上限 |
| 11 | OOB 只绑 `eth0` 的地址 | server 用 `-i` 指定网口；监听地址不是 `0.0.0.0`，所以 client 必须连这个 IP |
| 12 | 流量类别两端都要设 | 推荐 `-t 104`；GPU 版在测量范围内不敏感，但主机内存版的 READ 对它很敏感（§5.7） |
| 13 | 未触摸的大块 CPU 内存，注册特别慢 | **实测**：4 GiB 未触摸要 1017 ms，已触摸只要 16.5 ms（§1.2）。别把这个缺页成本误当成"RDMA 注册慢" |
| 14 | 每次传输都注册 | 注册是一次性的重开销。peermem 路径还有约 1.6 ms 的固定开销，小缓冲尤其显眼（§1.4）；应注册一次、反复使用 |
| 15 | VMM 显存用 peermem 注册失败 | **实测**：`Bad address`。分配时设 `gpuDirectRDMACapable = 1`，或改用 dmabuf（§1.6） |
| 16 | 先注销，再释放 | `cudaFree` 在 MR 还没注销时不报错（实测），但这不代表之后用这个 MR 做 RDMA 是安全的（未测）（§1.8） |
| 17 | dmabuf 的 fd 要自己关 | 注销 MR 之后，别忘了 `close(fd)`（程序里是 `ib.dmabuf_fd`） |
| 18 | BAR1 Used 不能用来观察注册开销 | **实测**：本机上它在 `cudaMalloc` 时就增加了，注册与注销前后不变（原因未查明）（§1.7） |
| 19 | 本环境测不出 `RLIMIT_MEMLOCK` 的影响 | Pod 里 `ulimit -l` 为 unlimited 且有 `IPC_LOCK` 能力；换到受限的环境，钉页数量可能成为瓶颈 |

<a id="A"></a>
## 附录 A. 验证状态汇总

[↑ 返回目录](#toc)

**实测** = 在 server-0 / server-1 上跑出来的（§1 里的注册类实测来自 `reg_bench`，在 server-0 的 GPU0 / `mlx5_0` 上）；**代码事实** = 直接来自源码；**推断** = 由机制或公开资料推导；**未测 / 未验证 / 未查明** = 没有证据，或有现象没有原因。

**环境与拓扑**

| 结论 | 状态 |
|---|---|
| 两个 Pod 在两个不同的 GPU 节点上；`eth0` 地址分别为 10.133.1.12、10.133.0.253 | **实测** |
| GPU0 与 NIC0/NIC1、GPU7 与 NIC14/NIC15 同为 `PXB`；Pod 里只有偶数号网卡可用 | **实测** |
| `mlx5_0` ↔ `fabric0`、`mlx5_14` ↔ `fabric14`；400 Gb/s、RoCE、`active_mtu = 4096`、`max_msg_sz` = 1 GiB | **实测** |
| `gid[1]` 为链路本地地址、`gid[3]` 为 `fd02:…` 的 RoCE v2 地址 | **实测** |
| 跨 rail 的 `ping6` 互通，往返约 0.13 ms | **实测** |
| `libibverbs` 头文件已在镜像里；宿主机加载了 `nvidia_peermem`（依赖 `nvidia` 与 `ib_uverbs`）；`ulimit -l` 为 unlimited | **实测** |
| nvcc 13.0、驱动 590.48.01（开放内核模块）、内核 6.12、rdma-core 39、8×B300 | **实测** |
| 系统页 4096；透明大页为 `[always]`；网卡 `max_mr_size` 无限制、`max_mr` = 16,777,216、`general_odp_caps` 含 `ODP_SUPPORT` 与 `ODP_SUPPORT_IMPLICIT` | **实测** |
| 设备属性：GPUDirect RDMA、dmabuf、VMM、VMM 下的 GPUDirect RDMA 均为 1；flush 选项为 1；写有序性为 100（`OWNER`） | **实测**（含义对照 `cuda.h` 枚举注释） |
| `nvidia_peermem` 缺失时 peermem 方式的 `ibv_reg_mr` 会失败 | **推断**（没有去掉模块验证） |

**注册（CPU 与 GPU）**

| 结论 | 状态 |
|---|---|
| CPU 内存注册：已触摸 1 MiB 0.04 ms → 4 GiB 16.5 ms；未触摸 1 MiB 0.04 ms → 4 GiB 1017 ms；注销 4 GiB 约 0.22 ms | **实测** |
| 未触摸与已触摸的差额来自首次访问的缺页 | **推断**（由对照得出，未单独测缺页） |
| `nvidia_peermem` 注册：1 MiB 1.58 ms → 4 GiB 9.1 ms；注销 0.6 → 2.9 ms | **实测**（两次独立运行相近） |
| dmabuf：导出约 0.12 ms + 注册 0.03–0.05 ms，与大小几乎无关；注销约 0.033 ms | **实测** |
| dmabuf 注册之所以便宜，是因为映射推迟到网卡访问时才建立 | **未验证**，原因未查明 |
| 两种 GPU 注册方式（peermem、dmabuf）都能完成完整的跨机 WRITE / READ，校验 PASS；两端可混用 | **实测**（`-m` 开关，§5.4 表 E） |
| 128 MiB 的注册耗时：dmabuf 约 0.23–0.25 ms，peermem 约 2.1 ms | **实测**（含 dmabuf 的"导出 + 注册"） |
| 两种注册方式下传输带宽、首次耗时无差别（29.8 / 30.0 GB/s） | **实测**（`-n 100`，单 QP，每种组合 1–3 次） |
| dmabuf 方式不依赖 `nvidia_peermem` 模块 | **推断**（没有在卸载模块的环境里验证） |
| VMM 显存：`gpuDirectRDMACapable = 0` 时 peermem 失败（`Bad address`）、dmabuf 成功；`= 1` 时两者都成功 | **实测**（注册层面；VMM 显存经 dmabuf 的完整传输未测） |
| BAR1 总量 524288 MiB；`cudaMalloc` 4 GiB 使 Used 增加 4096 MiB；注册与注销前后不变 | **实测**；原因未查明 |
| 注册后 `cudaFree`：两种方案 API 都返回成功，之后 `ibv_dereg_mr` 返回 0，进程不崩溃 | **实测**；之后能否 RDMA **未测** |
| CPU 路径：`ib_umem_get` → `pin_user_pages` → DMA 映射 → MTT | **推断**（公开设计，未跟踪内核） |
| GPU peermem 路径：peer memory client → NVIDIA 驱动取页与总线地址 | **推断** |
| GPU dmabuf 路径：CUDA 导出 dmabuf → 内核通用 dmabuf 框架 | **推断** |
| GPU 显存没有主机页描述，所以 `pin_user_pages` 不适用 | **推断** |
| GPU 显存按 64 KiB 页注册 | **推断**（公开资料），**未测** |
| BAR1 是 GPU 显存的 PCIe 访问窗口，CPU、网卡、其他 GPU 等 PCIe 设备都可经它访问显存；映射由 GPU 驱动管理；`cudaMemcpy` 不经过它 | **推断**（公开资料的一般说法，未验证） |
| `SYNC_MEMOPS` 的作用；peermem 把显存页物理地址经 BAR 交给网卡 | **推断**（公开资料的说法，未验证） |
| NVIDIA 官方推荐 dmabuf 而非 `nvidia-peermem`；dmabuf 要求开放内核模块、CUDA ≥ 11.7、内核 ≥ 5.12，MLNX_OFED 可选；peermem 要求 OFED | **公开资料**（NVIDIA GPU Operator 文档，2026-11-01 联网查阅） |
| NCCL 的 `NCCL_DMABUF_ENABLE` 自 2.13 起默认为 1，不支持时自动关闭 | **公开资料**（NCCL 文档）；本环境 NCCL 实际走哪种方式**未核实** |
| Ubuntu 从 6.8 内核起弃用 `nvidia-peermem` 接口；Blackwell 只支持开放内核模块 | **公开资料**（Ubuntu 公告、NVIDIA 驱动 595.84 README） |
| NVIDIA 官方已宣布 `nvidia-peermem` 弃用或移除 | **未查到**（官方页面只称 legacy 并推荐 dmabuf） |
| NCCL 的 `NCCL_CUMEM_ENABLE`（自 2.18）：2.19 起系统支持时自动开启，用 `cuMem*`（VMM）分配 NCCL 内存 | **公开资料**（NCCL 文档，2026-11-01 联网查阅） |
| 因此 NCCL 倾向走 dmabuf 注册（VMM 显存经 peermem 需要额外标志） | **推断**；本环境没有用 NCCL 日志核实 |
| dmabuf 与 peermem 的数据通路相同、差别在控制面（接口标准化、对 OFED 的依赖、VMM 与注册开销） | 带宽相同、注册耗时与 VMM 行为的差异为**实测**；接口性质的描述为**公开资料**；dmabuf 的动态导入 / 失效通知机制**未核对源码** |

**正确性**

| 结论 | 状态 |
|---|---|
| 两端 `ibv_reg_mr(GPU 显存)` 一次成功 | **实测** |
| RDMA WRITE（GPU0 → GPU7，64 MiB）正确，由 server 校验 PASS | **实测** |
| RDMA READ（GPU7 → GPU0，64 MiB）正确，由 client 校验 PASS | **实测** |
| 带宽测试之后两端再校验仍 PASS | **实测** |
| 两个程序编译一次通过；两端退出码都是 0 | **实测**（2026-11-01 改名并加 `-m` 开关后重新编译并跑通） |

**带宽与数据路径**

| 结论 | 状态 |
|---|---|
| WRITE 约 29.8 GB/s（238 Gb/s），不随并发深度（1～16）与 TC（0、104）变化 | **实测**（22 次 WRITE 测量：2026-10-31 的 9 次 `-n 100`（表 A–C）与 3 次 `-n 20`；2026-11-01 在 `-t 104` 下的 2 次（表 D）、表 E 的 7 次 `-n 100`、§5.3 的 1 次 `-n 20`；范围 29.0–29.8） |
| READ 通常约 30.0 GB/s；表 A–C 的 9 次里有 2 次偏低（17.9、22.1），其余及 `-n 20`、`-t 104` 的几次均为 30.0；偏低的原因 | 偏低现象**实测**，原因**未查明** |
| 约 30 GB/s 的平台期（标称 400 Gb/s 的 60%）的原因 | **未查明**；不是网卡/fabric 上限（同一对网卡上主机内存达 44 GB/s，见 §5.7，**实测**），与 GPU 一侧路径有关为**推断**；与注册方式无关（§5.4 表 E，**实测**） |
| 同一对网卡上，主机内存版 READ 43.8 GB/s（`-t 104`）、WRITE 深度 1 为 44.1 GB/s | **实测**（数据与逐次计时见 [c-multi-node-cpu.md](c-multi-node-cpu.md)） |
| 主机内存版 READ 对 TC 敏感（`-t 0` 约 24.6 GB/s 且不稳，`-t 104` 稳定）；GPU 版在测量范围内不敏感 | **实测**；原因**未查明** |
| 两个版本的 RDMA 字节计数器逐位相同 | **实测**（`-n 20`） |
| RDMA 流量约 1.3 GiB，与理论净载荷相符，两端互相印证；`eth0` 上只有约 1 KiB；`fabric` 内核网口计数为 0 | **实测**（`ethtool -S` 的 vport 计数器） |
| 数据不经过主机内存（GPUDirect RDMA） | **推断**（由直接注册 GPU 指针得出） |
| 就近网卡（`PXB`）比不就近的网卡更好 | **推断**（一般性说法）；**未测**（没有换网卡对比） |
| server 的 CPU 在传输期间不参与 | **推断**（单边操作的机制）；主机内存版测过 CPU 占用（约 1%），GPU 版未测 |
| 单机 NVLink 779.4 GB/s，约为 RDMA 的 26 倍 | **实测**（测量方法不同，只看量级） |

**明确没有测的**

| 项目 |
|---|
| 多 QP；改变消息大小；不同的 GPU–网卡就近关系对带宽的影响；VMM 分配的显存经 dmabuf 做完整传输；不加载 `nvidia_peermem` 的环境里跑 dmabuf |
| 交换机侧的 PFC / ECN 配置对 RoCE 的影响；TC 对 READ 的影响（主机内存版已实测有影响，GPU 版样本太少） |
| GPU 版的逐窗口带宽曲线（不知道长运行里 WRITE 是否也有爬升）；GPU 版在 `-t 104` 下 READ 是否偶发偏低 |
| 注册之后 `cudaFree`，再用这个 MR 发起 RDMA 会怎样 |
| ODP 的实际使用（只查了网卡能力）；GPU 显存的 ODP |
| `RLIMIT_MEMLOCK` 的限制效果（本环境 unlimited 且有 `IPC_LOCK`） |
| 多 GPU、多网卡、跨 NUMA 对注册的影响；页粒度的直接测量（没有读网卡的翻译表） |
| CUDA IPC 在跨机时是否确实失败（文档说法，见 [c-single.md](c-single.md#3-4)） |
| client 或 server 异常退出时对端的行为 |

<a id="B"></a>
## 附录 B. 已知局限与后续工作

[↑ 返回目录](#toc)

| 项目 | 现状 | 建议 |
|---|---|---|
| 带宽平台期 | 约 30 GB/s，仅为 400 Gb/s 的 60%，原因未查明（§5.4） | 做多 QP、消息大小扫描，并对照主机内存（`ib_write_bw`）的结果；换不就近的网卡对比（§3.2）（主机内存版已排除网卡/fabric 限速，两种注册方式带宽相同，见 §5.7、表 E） |
| READ 偶发偏低 | 表 A–C 的 9 次里出现 2 次；主机内存版的 READ 在 `-t 0` 下同样不稳（§5.7） | 加大样本量，并固定 TC 做对照；给 `client-gpu.cu` 加逐次计时与逐窗口带宽选项（同 `client-cpu.c` 的 `-o` / `-W`） |
| 单 QP、单 client | 程序只支持一条连接、一个 client | 多 QP、多 client，再看聚合带宽 |
| dmabuf | 已实现（`-m dmabuf`），带宽与 peermem 相同（表 E） | 在不加载 `nvidia_peermem` 的环境里验证；VMM 显存的完整传输；多 QP 下再对比 |
| 注册后释放的真实后果 | 只测了 API 返回值 | 注册后释放，再发起 RDMA，观察完成项状态 |
| dmabuf 为什么便宜、BAR1 为什么在 `cudaMalloc` 时就涨 | 都只有现象，没有机制证据 | 需要读内核日志或用更底层的工具 |
| 限制类的实验 | 本环境 `ulimit -l` 不限且有 `IPC_LOCK` | 在去掉 `IPC_LOCK`、限制 `ulimit -l` 的环境里补测钉页上限 |
| 异常处理 | 简化：一端中途崩溃时，另一端会报错退出或等待超时 | 补错误处理与重连 |
| `server` 只服务一次 | 处理完一个 client 就退出 | 改成循环 `accept` |
| 环境依赖 | 本篇在 Pod 里装了 `ethtool` 取证；`n-server*.yaml` 的启动脚本没有包含它 | 需要时把 `ethtool` 加进启动脚本 |

---

*环境：server-0 与 server-1（StatefulSet `server` 的两个 Pod，`n-server2.yaml`，各 8×B300，NV18 全互联，CUDA 13.0，驱动 590.48.01，RoCE v2 / 400 Gb/s）；实验验证日期 2026-10-31、2026-11-01；代码：`p2p/server-gpu.cu`、`p2p/client-gpu.cu`、`p2p/reg_bench.cu`，均含中文注释。*

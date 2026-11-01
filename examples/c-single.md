# C/C++ 直连 CUDA（中）：单机实验——NVLink 与 P2P

一台机器内 8 张 B300 通过 NVSwitch 全互联。本篇用三个裸 C/CUDA 程序，讲清**单机内 GPU 之间经 NVLink 通信**的原理、API 与实测：

- `0_to_7.cu`：GPU0 经 `cudaMemcpyPeer` 向 GPU7 传 16 GiB（约 17.2 GB），计时并校验（方式 B）；
- `remote_access.cu`：GPU0 的 kernel 直接读写 GPU7 的显存，并对照"不授权"的后果（方式 A）；
- `ipc_tcp.cu`：两个**进程**（一个用 GPU0，一个用 GPU7）经 CUDA IPC 交换数据。

**多机（RoCE）实验另文**，本篇不涉及。

**状态**：定稿（2026-10-31）。文中所有"实测"都已按当前源码在 server-0 上重新核对过（见附录 A）；已知局限与后续工作见附录 B。

<a id="how"></a>
## 怎么读

| 你想…… | 读哪里 |
|---|---|
| 理解 NVLink 通信到底是怎么回事、CPU 起什么作用 | §1 |
| 查 P2P 相关 API 的语义与参数 | §2 |
| 看带宽实测、复现实验 | §3.1–§3.2，命令见 §5 |
| 看 kernel 如何直接读写对端显存，以及不 `enable` 的后果 | §3.3 |
| 两个进程各用一张 GPU，数据怎么交换 | §3.4 |
| 调试这些程序 | §4 与 [debug.md](debug.md) |
| 想知道每个结论有多可靠 | 附录 A |

## 目录

- [怎么读](#how)
- [读完应该能回答](#intro)
- [约定与术语](#conv)
- **[1. NVLink 底层原理：CPU 配置、GPU 传输](#1)**
  - [1.1 物理层：GPU 之间怎么连](#1-1)
  - [1.2 控制面：CPU（驱动）做了什么](#1-2)
  - [1.3 数据面总览：两种搬运方式](#1-3)
  - [1.4 方式 A：kernel 直接读写对端地址](#1-4)
  - [1.5 方式 B：`cudaMemcpyPeer`](#1-5)
  - [1.6 NVLink 上传的信息有哪几类](#1-6)
  - [1.7 小结：谁负责什么](#1-7)
- **[2. P2P 编程模型：API 与语义](#2)**
  - [2.1 前提：cudaMalloc、UVA 与指针](#2-1)
  - [2.2 三个动词：can / enable / copy](#2-2)
  - [2.3 API 参考：签名与参数](#2-3)
  - [2.4 延伸：与 RDMA、NUMA、NCCL 的关系](#2-4)
- **[3. 实验：0_to_7.cu、remote_access.cu 与跨进程 IPC](#3)**
  - [3.1 0_to_7.cu 的设计与实现要点](#3-1)
  - [3.2 0_to_7.cu 的实测结果](#3-2)
  - [3.3 remote_access.cu：方式 A 的实验](#3-3)
  - [3.4 跨进程：CUDA IPC（ipc_tcp.cu）](#3-4)
- **[4. 调试：与本文实验相关的结论](#4)**
- **[5. 速查：运行命令与坑](#5)**
- **[附录 A. 验证状态汇总](#A)**
- **[附录 B. 已知局限与后续工作](#B)**

前置与相关：[c-build.md](c-build.md)（产物与编译）、[debug.md](debug.md)（调试环境与步骤）。

<a id="intro"></a>
## 读完应该能回答

1. 两张 GPU 之间的 NVLink 通信，底层到底是谁在做什么？CPU 起什么作用？（§1）
2. `cudaDeviceCanAccessPeer`、`cudaDeviceEnablePeerAccess`、`cudaMemcpyPeer` 各自的真实语义是什么？`cudaMemcpyPeer` 返回时拷贝完成了吗？（§2）
3. 单机 NVLink 能打到多少带宽？kernel 直接读写对端显存怎么写？不 `enable` 会怎样？（§3.1–§3.3）
4. 两个**进程**各用一张 GPU 时，数据怎么交换？为什么不能直接传指针？（§3.4）

<a id="conv"></a>
## 约定与术语

**约定**

- **实测**：在 server-0 Pod（nvcc 13.0，8×B300，sm_103）上的命令输出；**推断**：由机制或公开资料推导，未在本机观察；**未验证 / 未测**：既没有证据也没有测量。全部结论的汇总见附录 A。
- **跟敲**：跟着文档在终端里逐条敲命令，亲手重现结果。
- 带宽单位：除特别注明外为十进制 GB/s；`GiB` 为 2^30 字节。

**术语**

| 术语 | 含义 |
|---|---|
| **NVLink** | GPU 之间的高速直连链路；B300 每卡 18 条 |
| **NVSwitch** | 交换芯片，把 8 张 GPU 的 NVLink 端口连成全互联 |
| **P2P**（peer-to-peer） | GPU 直接访问另一张 GPU 的显存，不经过主机内存 |
| **UVA** | 统一虚拟地址：一个进程里，所有 GPU 显存和主机内存在同一个指针空间里，指针值互不重叠 |
| **SM** | 流多处理器，GPU 上执行 kernel 的计算单元（B300 有 148 个） |
| **Copy Engine** | GPU 芯片内部专做 DMA 拷贝的硬件单元，与 SM 相互独立、不占用 SM（拷贝与计算因此可以重叠）。B300 有 4 个（`cudaDeviceProp.asyncEngineCount = 4`，**实测**） |
| **HBM** | GPU 的显存 |
| **CUDA IPC** | 让不同进程共享同一块 GPU 显存的机制（§3.4） |
| **IPC 句柄** | 驱动为某块显存签发的、64 字节的不透明凭证；不是指针，也不含数据（§3.4） |

<a id="1"></a>
## 1. NVLink 底层原理：CPU 配置、GPU 传输

**先纠正一个直觉：CPU 并不"转发"数据，也不"选择"走 NVLink。** CPU 与驱动只做**一次性的配置（控制面）**，之后的数据传输（数据面）全部发生在 GPU 与 NVLink 硬件上，CPU 不参与。

**"NVLink 通信"是三层叠加的结果**，三者常被混为一谈：

| 层次 | 是什么 | 谁提供 |
|---|---|---|
| NVLink / NVSwitch | GPU 之间的高速链路与交换芯片（物理互联） | 硬件 |
| UVA | 一个进程里各设备的指针值互不重叠 | 驱动（软件） |
| P2P 访问 | 经 `enable` 建立的页表映射，一个 GPU 的 SM（load/store）或 Copy Engine（DMA）读写另一个 GPU 的 HBM | 页表映射 + 硬件 |

- **UVA 不依赖 NVLink，也不是"多个 GPU 之间建立"的**：它由驱动在每个进程里为所有设备分配不重叠的虚拟地址区间；只有一张 GPU、没有任何 NVLink 时同样有 UVA；PCIe 上的 P2P 也基于 UVA。
- **UVA 只保证指针值唯一，不保证能访问**：能不能访问，由 `enable` 建立的页表映射决定（§2）。

**一句话**：NVLink 通信 = 经 `enable` 建立页表映射后，一个 GPU 的 SM 或 Copy Engine 经 NVLink/NVSwitch 读写另一个 GPU 的 HBM，全程不经过 CPU 和主机内存。"读写对端 HBM"有两种方式（§1.3）；NVSwitch 还支持芯片内归约（NVLS，§2.4），数据不只是"从 A 的 HBM 到 B 的 HBM"。

本节按"物理上怎么连（1.1）→ CPU 做了哪些配置（1.2）→ 数据实际怎么走（1.3–1.6）→ 汇总（1.7）"展开。

<a id="1-1"></a>
### 1.1 物理层：GPU 之间怎么连

每张 GPU 芯片上有一组 **NVLink 端口**，8 张 GPU 的端口接到 **NVSwitch 芯片**上，由 NVSwitch 连成全互联：

```
GPU0 ─┐                    ┌─ GPU4
GPU1 ─┤                    ├─ GPU5
GPU2 ─┼── NVSwitch 芯片 ───┼─ GPU6      两卡之间的流量经 NVSwitch 路由，
GPU3 ─┘   (交换/路由)      └─ GPU7      不经过 CPU、不经过 PCIe
```

跟敲（**实测**）：

```bash
$ nvidia-smi topo -m | head -4
        GPU0  GPU1  GPU2 ...  GPU7  NIC0 ...
GPU0     X    NV18  NV18 ...  NV18  PXB  ...     ← GPU 两两之间都是 NV18
$ nvidia-smi nvlink -s -i 0 | head -3
GPU 0: NVIDIA B300 SXM6 PC (UUID: ...)
         Link 0: 53.125 GB/s                      ← 每条链路的速率
$ nvidia-smi nvlink -s -i 0 | grep -c Link
18                                                ← GPU0 共 18 条 NVLink，与 NV18 一致
```

`NV18` 的含义是两卡之间经 18 条 NVLink 捆绑相连。`topo -m` 里 GPU 与网卡之间的 `PXB` / `NODE` / `SYS` 是 PCIe 路径，与 NVLink 无关。

<a id="1-2"></a>
### 1.2 控制面：CPU（驱动）做了什么

| 时机 | 谁做 | 做什么 |
|---|---|---|
| 系统启动 | 内核驱动 | 识别 GPU 与 NVSwitch；对各条 NVLink 做**链路训练**，把物理链路拉起来 |
| 系统启动 | Fabric Manager（宿主机侧） | 配置 NVSwitch 的路由表，让每张卡都能到达其他卡。本机 `nvidia-smi -q` 中 `Fabric State : Completed`（**实测**）表示这一步已完成；它是宿主机层面的事，Pod 里只能看到结果 |
| 程序调用 `cudaDeviceEnablePeerAccess` | 内核驱动（经 libcuda → `ioctl`，见 [c-build.md §1.2](c-build.md#1)） | 在**本卡的页表**里登记对端显存的映射，标明"这块地址在对端 GPU，要经 NVLink 访问"（**推断**） |

`enable` 只负责建立映射；数据具体走 NVLink 还是 PCIe，由两卡之间的物理拓扑决定，不由这个 API 选择。

<a id="1-3"></a>
### 1.3 数据面总览：两种搬运方式

映射建好之后，数据有两种方式到达对端 HBM。**原理不同，走的物理通道相同**——都经 NVLink/NVSwitch，区别在于"谁来发起、怎么发起"：

- **方式 A**：kernel 里的线程直接对对端地址做 load/store（§1.4）。`remote_access.cu` 演示它（§3.3）；`0_to_7.cu` 没有使用；
- **方式 B**：调用 `cudaMemcpyPeer`，由 Copy Engine 做 DMA（§1.5）。`0_to_7.cu` 用的就是它。

| | 方式 A：kernel 直接读写对端地址 | 方式 B：`cudaMemcpyPeer` |
|---|---|---|
| 发起者 | GPU 的 **SM**（kernel 里的线程） | GPU 的 **Copy Engine**（DMA 硬件，**推断**） |
| 怎么发起 | 普通 load/store 指令，地址落在对端 | 驱动向 Copy Engine 下发"从哪搬到哪、搬多少"的拷贝命令 |
| 占用 SM | **占用** | **不占用**，SM 可同时算别的 |
| 需要写 kernel | 要 | 不要，一行 API 调用 |
| 没有 enable | 直接 illegal address，无降级（**实测**，§3.3） | 自动降级走主机中转（降级路径的速度本篇**未测**） |
| 访问粒度 | 细粒度，可随机、不连续，可边读边算 | 整块连续内存，批量搬运 |
| 对 host 的语义 | kernel 启动是异步的 | **也是异步的**：调用约 12 µs 返回，传输在返回之后才完成（**实测**，§2.3） |
| 本篇实测带宽 | **未测**（`remote_access` 只验证功能正确，没有计时） | 779.4 GB/s（16 GiB，§3.2） |

两种方式经过同一条物理通道，所以**推断**它们的带宽上限相同——瓶颈在链路而不是发起者。但本篇只实测了方式 B 的带宽；方式 A 只验证了功能正确（§3.3），所以"上限相同"仍是推断。

**怎么选**：整块搬运且想让 SM 去做别的事 → B；搬数据的同时要计算、或访问不连续 → A。

<a id="1-4"></a>
### 1.4 方式 A：kernel 直接读写对端地址

（`0_to_7.cu` 未使用；`remote_access.cu` 演示了这种方式，代码与实测见 §3.3。`0_to_7.cu` 的程序注释也指出："若把传输改成 kernel 远程解引用，enable 就是硬前提"。）一次对端访问的路径：

```
GPU0 的 SM 执行 ld/st 指令，地址 = UVA 指针（指向 GPU7 的显存）
   ↓
GPU0 的 MMU 查页表 → 命中 enable 时填的映射项："对端显存，走 NVLink"
   ↓
请求不交给本地显存控制器，而是交给 NVLink 控制器，打成 NVLink 数据包
   ↓
经 NVSwitch 路由到 GPU7
   ↓
GPU7 的 NVLink 端口收到 → 其内存控制器读写 HBM → 结果沿原路返回
```

对 GPU0 的程序来说这就是一条**普通的 load/store 指令**，只是地址恰好落在别家；没有 CPU 介入，没有拷贝函数调用。

**线上流程：读与写不同**

| 操作 | 线上流程 |
|---|---|
| **读**（load） | 发起方发**读请求**（带地址）→ 对端读 HBM → **读响应**（带数据）返回，是一个往返 |
| **写**（store） | 发起方发**写请求**（带地址和数据）→ 对端写 HBM → 回一个**写确认** |

三点补充：

1. **读是"请求 + 响应带回数据"，写是"请求带着数据 + 确认回来"**——写时数据随请求一起发出，回来的只是确认，不是数据；
2. **处理请求的是对端的硬件，不是对端的程序**：对端 GPU 上不需要有任何 kernel 在运行，也不需要它的 CPU 参与；负责处理的是对端芯片里的 NVLink 端口和内存控制器，直接读写 HBM。这也是 P2P 访问能"绕过对方处理器"的原因，与 RDMA 的单边操作类似（§2.4）；
3. **地址翻译发生在发起方**：发起方 SM 拿着 UVA 指针，先在**自己这张卡**的 MMU 里查页表；页表项是 `enable` 时填的，它告诉硬件"这个地址属于对端，要走 NVLink"。所以"把地址通过 NVLink 发过去"之前，还有一步本地翻译与路由判断（请求中携带的具体地址格式**未验证**）。

**为什么单次往返的延迟不会拖垮带宽**（**推断**，基于 GPU 执行模型，未测）：一次读是一个往返，延迟不小；但 GPU 上有大量线程同时在飞，每个线程各发各的请求，链路上同时有成千上万个未完成请求，延迟被并发掩盖，带宽仍能被灌满。

<a id="1-5"></a>
### 1.5 方式 B：`cudaMemcpyPeer`

`0_to_7.cu` 用的就是这种。驱动让 **Copy Engine** 经 NVLink 搬数据，不占用 SM。本篇实测 779.4 GB/s（§3.2）。

**CPU 与 DMA 的角色**：

| | 做什么 | 数据是否经过 |
|---|---|---|
| CPU（驱动） | 把"从哪搬到哪、搬多少"编码成命令，写入 GPU 的命令缓冲区并通知 GPU；**下发后立即返回**（实测约 12 µs），CPU 线程不等传输；要等完成必须显式同步 | **不经过 CPU**，也不经过主机内存 |
| 执行搬运的 Copy Engine | 以 DMA 方式经 NVLink/NVSwitch 把数据搬到对端 HBM | — |
| 对端 GPU | **被动**：其 NVLink 端口与内存控制器响应读写请求，不会自己再发起一次 DMA | — |

三点澄清：

1. **"CPU 协调"与"数据经过 CPU"是两回事**：CPU 只负责下命令，下完即返回。方式 A 的 kernel 启动同样要 CPU 提交命令，所以"需要 CPU 发起"不是 A 与 B 的区别；真正的区别是由谁搬数据——A 是 SM，B 是 Copy Engine；
2. **不是"双方都走 DMA"**：实际执行搬运的是**一个** Copy Engine。具体是源端还是目的端 GPU 的引擎执行，**未验证**，公开资料的说法也不确定，不要写死；
3. **没有 `enable` 时的降级路径**：页表里没有映射，`cudaMemcpyPeer` 自动降级为 GPU → 主机内存 → GPU 两跳（仍由 DMA 完成，但绕了一圈）。**这是唯一数据会经过主机内存的情形**，不是正常路径；其速度本篇**未测**。

**一句话**：`cudaMemcpyPeer` = CPU 上的驱动下发一条拷贝命令后立即返回，由 GPU 的一个 Copy Engine 以 DMA 方式经 NVLink 把数据搬到对端 HBM；数据全程不经过 CPU 和主机内存；要知道拷贝何时完成，必须用事件或同步调用显式等待。

<a id="1-6"></a>
### 1.6 NVLink 上传的信息有哪几类

可从应用层（程序在传什么）和链路层（线上有哪些包）两个角度分。

**应用层**：

| 类别 | 内容 | 本文里的例子 | 本机验证 |
|---|---|---|---|
| 数据读写 | 普通 load/store、DMA 拷贝的数据 | `cudaMemcpyPeer` | **实测**：传输 68.8 GB 数据，计数器相应增加（见下方计数器实测） |
| 原子操作 | 对对端显存做原子加、比较交换等 | 本文未用到（`check_kernel` 的 `atomicAdd` 作用在本卡本地） | **实测**：`nvidia-smi nvlink -c` 显示 `P2P atomics supported: true` |
| 集合通信的归约/多播 | NVSwitch 芯片内做求和、把一份数据复制给多个 GPU（NVLS） | 本文仅提名字（§2.4） | **推断**，未在本机观察 |
| 访问系统内存 | GPU 经 NVLink 访问主机内存 | 本文未涉及 | **实测**：`nvlink -c` 显示 `Access to system memory supported: true`；实际能否使用、走哪条路**未验证** |

**链路层**（**推断**，依据公开资料；本机无法看到包内容）：

| 类别 | 作用 |
|---|---|
| 请求包 | 读请求、写请求（带地址和数据） |
| 响应包 | 读响应（带回数据）、写确认 |
| 流控信息 | 告知对端"我还有多少缓冲空间"（信用机制），避免溢出 |
| 链路可靠性信息 | CRC 校验，出错时请求重传 |
| 链路管理信息 | 链路训练、状态、低功耗等控制信号 |

所以一次远程写，线上不只有数据，还有头部、确认和流控。

**计数器实测**：`0_to_7.cu` 的注释也建议用 `nvidia-smi nvlink -gt d` 前后取差来确认流量真的走了 NVLink。`-gt d` 只计数据载荷，`-gt r` 另含协议开销。跑一次 `0_to_7`（2026-10-31，**实测**），传输前后各读一次再相减：

```bash
$ nvidia-smi nvlink -gt d -i 0 > d0; nvidia-smi nvlink -gt r -i 0 > r0   # 传输前
$ ./0_to_7
$ nvidia-smi nvlink -gt d -i 0 > d1; nvidia-smi nvlink -gt r -i 0 > r1   # 传输后；各链路 Tx/Rx 求和后相减
GPU0 数据 Tx 增量  ≈ 68.82 GB   ← ≈ 4 × 17.18 GB（1 次 warmup + 3 次迭代，§3.1）
GPU0 数据 Rx 增量  =  0         ← 单向传输，GPU0 只发不收
GPU0 Raw Tx 增量   ≈ 69.12 GB   ← Raw / Data ≈ 1.0044
GPU0 Raw Rx 增量   ≈  0.18 GB   ← 应答/控制流量
```

数据 Tx 增量与"4 次 × 17.18 GB"吻合，证明数据确实经 NVLink 发出；发送方向的头部等开销约 **0.4%**；GPU0 只发数据却仍有约 0.18 GB 的 Raw Rx，**推断**为写确认、流控等响应类信息（未进一步验证）。`nvidia-smi nvlink -e` 可查各链路的错误计数，对应"可靠性信息"一类，本机未跑。

<a id="1-7"></a>
### 1.7 小结：谁负责什么

| 环节 | 谁做 | 何时 |
|---|---|---|
| 链路训练、NVSwitch 路由配置 | 内核驱动与 Fabric Manager | 系统启动时 |
| 在页表里登记对端显存 | 内核驱动（由 `enable` 触发） | 程序运行时，一次 |
| 地址翻译、打包、路由、读写对端 HBM | GPU 硬件、NVSwitch | 每次访问 |
| 发起传输（load/store 或 DMA） | GPU 的 SM 或 Copy Engine | 每次访问 |
| 传输期间的 CPU | 下发命令后立即返回，不参与搬运；需要完成通知时显式同步 | — |

本节各结论哪些是实测、哪些是推断，见**附录 A**。

<a id="2"></a>
## 2. P2P 编程模型：API 与语义

<a id="2-1"></a>
### 2.1 前提：cudaMalloc、UVA 与指针

每一次 `cudaMalloc` / `cudaFree` 都是**真实的驱动调用**（**推断**，源码机制）。写裸 C 时的三个后果：

1. **首次分配慢**：涉及驱动交互；批量分配、进程内复用这类优化是上层框架的事；
2. **释放即真实**：`cudaFree` 立刻归还驱动——写"测完释放再测下一个"的测量逻辑时，边界清晰可预期；
3. **对齐**：`cudaMalloc` 返回的地址对齐满足 kernel 里向量化访问（如 `float4`，16 B）的要求。

**UVA** 让所有显存**看起来**在同一地址空间（指针都是统一的 `0x7ffd...`）。注意"看起来"：它只保证指针值唯一，**不代表任何 GPU 都能直接解引用任何指针**——能不能访问，由 §2.2 的页表映射决定（UVA 与 NVLink 的区别见 §1 开头）。

**指针只是地址（实测）**：在 GPU7、GPU0 上各 `cudaMalloc` 一块，再用 `cudaPointerGetAttributes` 查：

```
remote = 0x7f4a1d000000   local = 0x7f49e7000000   ← 数值上看不出谁在哪张卡
驱动记录的 remote: device=7  type=2 (2=device)
驱动记录的 local : device=0  type=2
```

指针值本身是个普通的 64 位整数；"属于哪张卡"记在**驱动的分配记录**里，不在指针值里。能否访问则由 `enable` 决定——指针是门牌号，`enable` 才是钥匙。**跨进程同理**：指针只在本进程的 context 里有效，另一个进程拿到同样的数值也不能用，要先走 CUDA IPC（§3.4）。

<a id="2-2"></a>
### 2.2 三个动词：can / enable / copy

| API | 动词 | 作用 |
|---|---|---|
| `cudaDeviceCanAccessPeer(&can, dev, peer)` | **查** | 问驱动"dev 的 kernel 能否直接访问 peer 的内存"，输出 0/1，不改任何状态 |
| `cudaDeviceEnablePeerAccess(peer, 0)` | **开** | 真正授权——**建立页表映射**，kernel 远程解引用的硬前提 |
| `cudaMemcpyPeer(...)` | **用**（方式 B） | 执行拷贝，不行自动降级走 host staging |

"用"有两种方式：方式 B 是调 `cudaMemcpyPeer`（`0_to_7.cu` 用的）；方式 A 是自己写 kernel 直接解引用对端指针（§1.4，`remote_access.cu` 演示）。`can` 与 `enable` 是两种方式**共同的前置步骤**。

**这组 API 是通用的 GPU P2P 接口，不专属于 NVLink**：走 NVLink 还是 PCIe 由物理拓扑决定。本机是 NVSwitch 全互联，所以走 NVLink。

**查——纯查询，反映拓扑不反映状态**：返回 1 只表示"调 enable 会成功"，enable 前后答案相同。由两卡的物理连接决定：

| 两卡之间的连接 | 返回值 |
|---|---|
| NVLink / NVSwitch 互联 | 1（本机 `0_to_7` 实测：`canAccessPeer(0->7)=1`、`canAccessPeer(7->0)=1`，§3.2） |
| PCIe，同一根复合体 | 通常 1 |
| 其他 | 0 |

**按方向问**：`canAccessPeer(0,1)` 与 `(1,0)` 是两个独立问题，非对称拓扑可能一向 1 一向 0（NVSwitch 全互联下恒等）。`0_to_7.cu` 因此两个方向都查。

**开——enable 具体做什么：建立页表映射**。GPU 的 MMU 页表里**默认没有别家内存的映射**——拿着 GPU0 的指针在 GPU7 上解引用，MMU 查不到页表项，就是 illegal access。enable 让驱动做三件事（**推断**，源码/文档机制）：

```
① 映射页表：在 dev 的 MMU 页表里登记 peer 内存区间 → 远程解引用的地址翻译走通
② 接通路由：NVLink / NVSwitch fabric 侧可达路径确认
③ 记账：每个 (dev, peer) 对一条映射，消耗页表项——
   这就是文档说"开启 peer access 可能减少可用显存"的原因
```

类比：**给对方家配一把你家的钥匙**——钥匙（页表映射）不配，地址（UVA 指针）写得再清楚也进不去：要么进不去（kernel：illegal address），要么物业带他绕正门（`cudaMemcpyPeer`：host staging）。

**"通信前必须 enable 吗"——分两级**（kernel 一行为**实测**；`cudaMemcpyPeer` 一行依据程序注释，降级路径本篇未实测）：

| 通信方式 | 必须 enable 吗 | 依据 |
|---|---|---|
| kernel 远程解引用 | **硬必须**——无降级路径，直接 illegal address | **实测**：`remote_access` 的 `-DNO_ENABLE` 版触发 `illegal memory access`（§3.3）；`0_to_7.cu` 注释也这样说明 |
| `cudaMemcpyPeer` | **功能不必须、性能必须**——不开也能完成，但自动降级走 host 慢路径 | `0_to_7.cu` 注释："不能 P2P 会自动走 host staging"；慢多少本篇**未测** |

**四个使用细节**：

1. **方向性**：授予的是 *dev 访问 peer* 单向——`0_to_7.cu` 里成对调用 `enable_peer(0,7)` + `enable_peer(7,0)`；
2. **进程内作用域**：映射挂在本进程的 CUDA context 上，别的进程不可见；
3. **幂等**：重复调用返回 `cudaErrorPeerAccessAlreadyEnabled`，不是错误（`0_to_7.cu` 的 `enable_peer` 视同成功）；
4. **真失败**：`cudaErrorPeerAccessUnsupported`——硬件不支持直连（`can` 查询返回 0 的场合），映射根本建不了。

**`can` 返回 0 的后果**（三条链全断）：enable 必然失败、`cudaMemcpyPeer` 静默降级、kernel 远程解引用无路可走——所以 `0_to_7.cu` 在 ① 先做 `can` 检查，并打印"NVLink 直连"或"不可直连, 将走 host staging"（§3.1）。

<a id="2-3"></a>
### 2.3 API 参考：签名与参数

四个 API 的完整签名（返回值都是 `cudaError_t`，成功 = `cudaSuccess`）：

```c
cudaError_t cudaDeviceCanAccessPeer (int* canAccessPeer, int device,    int peerDevice);
cudaError_t cudaDeviceEnablePeerAccess (int peerDevice, unsigned int flags);
cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
cudaError_t cudaMemcpyPeer (void* dst, int dstDevice,
                            const void* src, int srcDevice, size_t count);
```

**`cudaDeviceCanAccessPeer` —— 查**

| 参数 | 方向 | 含义 |
|---|---|---|
| `canAccessPeer` | 出参 | 写入 1（可直连）或 0（不可） |
| `device` | 入参 | 发问的设备："**我的** kernel 能否访问 peer 的内存？" |
| `peerDevice` | 入参 | 被访问的设备 |

纯查询、不改状态；问的是 device→peer 单向（§2.2"查"）。

**`cudaDeviceEnablePeerAccess` —— 开**

| 参数 | 含义 |
|---|---|
| `peerDevice` | 授予"**当前设备**"（先 `cudaSetDevice` 切到它）访问 `peerDevice` 内存的权限 |
| `flags` | 目前必须为 0（保留位，无可用选项） |

返回值三分支（`0_to_7.cu` 的 `enable_peer` 正是按这个处理的）：

- `cudaSuccess`：已开启；
- `cudaErrorPeerAccessAlreadyEnabled`：重复开启——幂等，视同成功；
- `cudaErrorPeerAccessUnsupported`：硬件不支持（对应 `can` 查询返回 0 的场合）——**真正的失败**。

`cudaDeviceDisablePeerAccess` 参数同上；未开启时返回 `cudaErrorPeerAccessNotEnabled`（同样幂等处理）。

**`cudaMemcpyPeer` —— 用**

| 参数 | 含义 |
|---|---|
| `dst` / `src` | 裸指针（UVA 下全地址空间唯一） |
| `dstDevice` / `srcDevice` | 两块内存各自的设备号——显式声明，驱动不用猜 |
| `count` | 字节数 |

三个要点：

1. **对 host 是异步的——返回不等于拷贝完成**（**实测**，2026-10-31）。对 4 GiB 做一次 `cudaMemcpyPeer(GPU0→GPU7)`，用 `std::chrono` 量调用本身，再 `cudaDeviceSynchronize()` 量剩余等待：

   ```
   run0: 调用返回耗时 13 us | 返回后再同步又等 5495 us | 4GiB@775GB/s 理论 5542 us
   run1: 调用返回耗时 11 us | 返回后再同步又等 5501 us
   run2: 调用返回耗时 12 us | 返回后再同步又等 5499 us
   ```

   调用约 12 µs 就返回，传输在返回之后继续约 5.5 ms（与理论值吻合）。所以**不能用 CPU 时钟包住 `cudaMemcpyPeer` 来计时**（只会量到约 12 µs 的启动开销），必须用 CUDA 事件加同步——这正是 `0_to_7.cu` 采用"事件计时三段式"的原因（§3.1）。非 `Async` 版与 `cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream)` 的区别只在于是否指定 stream，两者行为差异本机**未比较**；
2. **不需要 enable 也能完成**（自动降级 host staging），但**快路径要求 peer access 已开启**（§2.2 分级表；慢多少本篇未测）；
3. 方向任意（`dst`/`src` 在哪张卡都行），设备号参数与 enable 的"当前设备"语境无关。

**配套支撑 API**（三个程序用到的其余关键调用）：

| API | 作用 |
|---|---|
| `cudaGetDeviceCount(int*)` / `cudaSetDevice(int)` | 多卡控制：查卡数 / 切换"当前设备"——后续分配、kernel 启动都作用于它 |
| `cudaMalloc(void**, size_t)` / `cudaFree(void*)` | 在**当前设备**上分配 / 释放（§2.1） |
| `cudaMemset` | 在当前设备的显存上置初值（用于预清零接收缓冲与计数器） |
| `cudaEventCreate / Record / Synchronize / ElapsedTime` | GPU 侧时间戳四件套——计时三段式的底座 |
| `cudaDeviceSynchronize()` | 等**当前设备**上的全部工作完成 |
| `cudaGetLastError()` | kernel 启动错误的**唯一**获取途径——`<<<>>>` 语法不返回错误码，**启动后必须查**（`launch_fill` 里紧跟的那行 `CHECK_CUDA(cudaGetLastError())` 就是这个原因） |
| `cudaPointerGetAttributes(&attr, ptr)` | 查询一个指针属于哪张卡、是什么类型（`attr.device`、`attr.type`）——指针值本身看不出这些（§2.1） |
| `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` / `cudaIpcCloseMemHandle` | 跨进程共享显存（§3.4） |

<a id="2-4"></a>
### 2.4 延伸：与 RDMA、NUMA、NCCL 的关系

以下为类比定位（**推断**），也是后续多机（RoCE）实验的铺垫。

| 维度 | NVLink P2P | RDMA |
|---|---|---|
| 地址空间 | 全机统一（UVA） | 节点间隔离（GVA + rkey 交换） |
| 访问原语 | 普通 load/store（地址恰好落在别家） | 显式 verb 提交（`ibv_post_send`） |
| 谁发事务 | SM 指令流本身 | HCA（CPU 只 post 描述符） |
| 完成机制 | 指令退休即完成 | CQ + 轮询/中断 |
| 授权建立 | enable（页表映射） | `ibv_reg_mr`（pin + 内存窗口） |

一句话：**授权模型像 RDMA**（一侧发起、绕过对方处理器、零拷贝），**访问语义像 NUMA**（统一地址空间 + 普通 load/store）——GPU P2P 本质是"带显存的 NUMA"，NVSwitch 就是把 8 个 NUMA 节点缝起来的那张互联。NVLS（NVSwitch 芯片内归约）在 RDMA 世界**没有对应物**——IB/RoCE 交换机是哑的。

**NCCL 的视角**：NCCL 把两者当平级 transport——intra-node P2P（NVLink）、inter-node IBVerbs（RDMA，含 GPUDirect RDMA——NIC 的 DMA 直接进出显存）。`torch.cuda.can_device_access_peer` 是 `can` 的 Python 绑定；NCCL 内部调同一 driver API（`cuDeviceCanAccessPeer`）判定 intra-node 走 P2P transport 还是退 SHM——`NCCL_P2P_DISABLE=1` 的性能雪崩，机制上就是作废这条判定、逼 NCCL 走 SHM/主机路径；通信器初始化时还会对每对 GPU 调 `cuCtxEnablePeerAccess` 建同样的页表映射——Ring / Tree / NVLS 全踩在这层地基上。

<a id="3"></a>
## 3. 实验：0_to_7.cu、remote_access.cu 与跨进程 IPC

三个程序各回答一个问题（源码都在 `p2p/` 目录，含中文注释；编译命令见 [c-build.md 附录 A](c-build.md#A)）：

| 程序 | 行数 | 方式 | 数据量 | 做什么 | 验证了什么 |
|---|---|---|---|---|---|
| `0_to_7.cu` | 227 | 方式 B（`cudaMemcpyPeer`） | 16 GiB × 2（GPU0 源、GPU7 目的） | GPU0 → GPU7 传输，事件计时，GPU 侧校验 | 带宽 779.4 GB/s；数据正确；确实走了 NVLink（§3.1–§3.2） |
| `remote_access.cu` | 197 | 方式 A（kernel 直接读写） | 64 MiB × 2 | 同一进程里，GPU0 的 kernel 读写 GPU7 的显存；另有 `-DNO_ENABLE` 对照版 | 读写正确；不 `enable` 会 `illegal memory access`；不计时（§3.3） |
| `ipc_tcp.cu` | 255 | 方式 A + CUDA IPC | 16 MiB × 2 个区域 | 两个**独立进程**（server 用 GPU7，client 用 GPU0）经 IPC 句柄共享显存、双向交换数据，控制信息走 TCP | 双向交换正确；句柄的性质（§3.4） |

<a id="3-1"></a>
### 3.1 0_to_7.cu 的设计与实现要点

16 GiB = 2^34 字节 = 2^32 个 `float`。四步流程（与输出一一对应）：

| 步骤 | 部件 | 说明 |
|---|---|---|
| ① 检查 | `cudaDeviceCanAccessPeer` + `enable_peer` | 0↔7 双向可达性（NVLink 直连证据）+ 建页表映射（§2.2"开"） |
| ② 准备 | `fill_kernel` | GPU0 上 16 GiB 填 `(float)i` 模式——校验链的起点 |
| ③ 传输 | `cudaMemcpyPeer` + 事件计时三段式 | warmup 排除首次开销，3 次迭代取平均 |
| ④ 校验 | `check_kernel` + `atomicAdd` | **GPU 侧**数"≠ 模式"的元素，只回传 8 字节计数器 |

实现要点：

| 要点 | 做法 | 原因 |
|---|---|---|
| 需要 8 卡 | `cudaGetDeviceCount` < 8 直接退出 | 实验固定用 0 号和 7 号卡 |
| 双向 enable | `enable_peer(0,7)` + `enable_peer(7,0)`，已开启视为成功 | 方向性与幂等（§2.2）；对 `cudaMemcpyPeer` 不是必须，显式打开是为了语义完整（程序注释） |
| 计时三段式 | warmup 一次 + `cudaDeviceSynchronize`；事件建在目标设备（GPU7）；`record t0` → 3 次 `cudaMemcpyPeer` → `record t1` → `cudaEventSynchronize(t1)` | `cudaMemcpyPeer` 对 host 异步（§2.3），不能用 CPU 时钟；warmup 触发建路，不计成绩 |
| fill / check kernel 的启动 | block = 256，grid 封顶 4096，用 grid-stride 循环处理余量 | 封顶是**设计选择，不是硬件限制**：实测 `maxGridSize[0] = 2147483647`，2^32 个元素不封顶也只需 16,777,216 个 block；封顶后每个线程循环处理 4096 个元素；每次启动后查 `cudaGetLastError` |
| 拷贝前未对 GPU0 显式同步 | `fill_kernel` 启动后没有 `cudaDeviceSynchronize`，依赖拷贝与默认流之间的隐式顺序 | 本机 PASS，说明顺序成立，但这个依赖**未单独验证**；严格写法是在 `launch_fill` 之后加同步（程序注释已指出）。对照：`remote_access.cu` 在 GPU7 填充后有显式同步（§3.3） |
| 校验在 GPU 上做 | 在 GPU7 本地用 `check_kernel` 数"≠ `(float)i`"的元素，`atomicAdd` 累加到一个 8 字节计数器，只回传这一个数 | 不把 16 GiB 拉回 host；校验读的是 GPU7 本地显存，不占 NVLink |
| 接收缓冲预清零 | `cudaMemset(dst, 0, BYTES)` | 保证"没拷到的位置必然 ≠ 模式"，拷贝失败不会被偶然正确的残留内存掩盖 |
| `(float)i` 的精度 | 大数转 `float` 有舍入，但填充与校验两边用同一规则 | 只要拷贝没坏，`bad` 必然为 0；**值被改坏**一定会被数出来。局限见下 |

**校验的局限**（已用 float32 实际计算验证，程序注释里也写明了）：`(float)i` 在 i ≥ 2^24 后，相邻下标会共用同一个 float 值——从 2^28 起连续 1024 个下标只有 33 个不同值，从 2^31 起只有 5 个（约每 256 个下标共用一个值）。数组共 2^32 个元素，后一半都是这种情况。因此：

| 问题 | 能否检出 |
|---|---|
| 某个元素的值被改坏（任何字节出错，含变成 NaN） | **能** |
| 某部分根本没拷到 | **能**（下标 0 例外：预清零的 0 恰好等于 `(float)0`） |
| 数据错位 / 某一小块放错位置，且错位后的值落在同一片相同值里 | **大下标区可能漏检** |
| 数据走的是不是 NVLink | **不能**：经主机中转的降级路径同样能得到正确数据并 PASS，要确认路径需读 `nvidia-smi nvlink -gt d` 计数器（§1.6） |

改进办法：把模式改成 32 位无符号整数 `p[i] = (unsigned)i`——2^32 个下标恰好都能被 32 位无符号数精确表示，每个值唯一，错位就无处可藏，校验代价不变（尚未改动程序，仅记录在注释中）。

<a id="3-2"></a>
### 3.2 0_to_7.cu 的实测结果

跟敲：

```bash
$ nvcc -O3 -arch=native 0_to_7.cu -o 0_to_7 && ./0_to_7
GPU0 --NVLink--> GPU7, 17.2 GB per message

[1] P2P: canAccessPeer(0->7)=1, canAccessPeer(7->0)=1  (NVLink 直连)
[2] transfer: 3 x 17.2 GB in 66.13 ms -> 779.4 GB/s
[3] verify: 0 mismatched elements -> PASS
```

结论：

1. **779.4 GB/s**（连续运行 3 次，结果相同：66.13 ms）——16 GiB 的消息打到单向理论 ~900 GB/s 的 **87%**（~900 GB/s 取 NVLink 5 公开规格：18 条链路 × 单向 50 GB/s，**推断**，未在本机验证）。与 §2.3 的单次 4 GiB 测量（5.5 ms，约 780 GB/s 量级）一致；
2. **确实走了 NVLink**：传输前后读 `nvidia-smi nvlink -gt d` 的计数器，GPU0 的数据 Tx 增量约 68.82 GB，与 4 次 × 17.18 GB 吻合（§1.6）；
3. **校验零成本**：2^32 个元素（16 GiB）全量校验，host 只收到 8 字节结果——GPU 侧校验对大消息的实惠；
4. **单位注**：`BYTES = 16 << 30` 是 16 **GiB**（2^34 字节），十进制 17.2 GB——程序打印用十进制，注释里写清了换算，免生"16 怎么变 17.2"的疑惑。

<a id="3-3"></a>
### 3.3 remote_access.cu：方式 A 的实验

`remote_access.cu` 演示方式 A：不调用 `cudaMemcpyPeer`，而是让 **GPU0 上的 kernel 直接拿 GPU7 显存的指针做读写**。数据量 64 MiB（2^24 个 `unsigned`），**只验证功能，不计时、不测带宽**。

```bash
nvcc -O3 -arch=native remote_access.cu -o remote_access
nvcc -O3 -arch=native -DNO_ENABLE remote_access.cu -o remote_access_noenable   # 对照版：故意不 enable
./remote_access
```

两个二进制用的是**同一份源码**，区别只是编译时有没有 `-DNO_ENABLE`：加了就把 `cudaDeviceEnablePeerAccess` 那一段去掉。

**流程**：

| 步骤 | 当前设备 | 做什么 |
|---|---|---|
| 1 查询 | — | `cudaDeviceCanAccessPeer(&can, 0, 7)`，为 1 才继续 |
| 2 准备 | 7 → 0 | GPU7：`cudaMalloc` 出 `remote_buf`，`init_kernel` 填 `remote_buf[i] = i`，再 `cudaDeviceSynchronize`；GPU0：`cudaMalloc` 出 `local_buf` |
| 3 授权 | 0 | `cudaDeviceEnablePeerAccess(7, 0)`：方向是"GPU0 访问 GPU7"（`-DNO_ENABLE` 时没有这一步） |
| A1 远程读 | 0 | `remote_read`：`local[i] = remote[i] + 1`；host 上核对 |
| A2 远程写 | 0 → 7 | `remote_write`：`remote[i] = i * 2`；切到 GPU7 把 `remote_buf` 取回来核对 |

**关键代码**——kernel 里访问对端显存的写法，与访问本地显存**完全一样**：

```cpp
// 在 GPU0 上运行；remote 指向 GPU7 的显存，local 在 GPU0 本地
__global__ void remote_read(const unsigned* __restrict__ remote,
                            unsigned* __restrict__ local, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride)
    local[i] = remote[i] + 1;        // 读对端、写本地
}

__global__ void remote_write(unsigned* __restrict__ remote, size_t n) {
  /* 同样的下标与步长计算 */
  for (; i < n; i += stride)
    remote[i] = (unsigned)i * 2u;    // 直接写进对端显存
}
```

**实测结果**（2026-10-31）：

```
canAccessPeer(0->7) = 1
peer access 0->7 enabled
[A1] 远程读 : GPU0 kernel 读 GPU7 -> 不匹配 0 个 -> PASS
[A2] 远程写 : GPU0 kernel 写 GPU7 -> 不匹配 0 个 -> PASS
```

对照版（`-DNO_ENABLE`，**实测**）：

```
canAccessPeer(0->7) = 1                      ← 拓扑上仍然"能连"
!! 故意不调用 cudaDeviceEnablePeerAccess（-DNO_ENABLE）
FATAL remote_access.cu:160  cudaDeviceSynchronize(): an illegal memory access was encountered
```

**要点**：

1. **`remote` 只是地址，权限来自 `enable`**：同一个指针值，授权了能读写，没授权就非法访问。比喻：指针是门牌号，`enable` 才是钥匙。指针本身是普通的 64 位整数，看不出属于哪张卡（§2.1 的 `cudaPointerGetAttributes` 实测）；
2. **"能连"不等于"能访问"**：`can` 在两个版本里都返回 1，因为它只反映拓扑；
3. **错误在同步处才暴露**：没有 `enable` 时，kernel 启动本身不报错（`cudaGetLastError()` 通过），要到后面的 `cudaDeviceSynchronize()` 才出现 `an illegal memory access was encountered`。在 `cuda-gdb` 里则会直接停在 `CUDA Exception: Warp Illegal Address`，定位到 `remote_access.cu:57`；停下的 block / thread 是第一个被捕获到的，每次运行可能不同（调试步骤见 [debug.md §4.4](debug.md#4)）；
4. **两个"设备"要分清**：指针指向哪张卡，取决于 `cudaMalloc` 时的当前设备；kernel 在哪张卡上运行，取决于 `<<<>>>` 启动时的当前设备。本程序故意让两者不一致——数据在 GPU7，kernel 在 GPU0；
5. **显式同步**：GPU7 填完 `remote_buf` 后调用了 `cudaDeviceSynchronize`，GPU0 再去读。因为 GPU0 上的 kernel 不会自动等待 GPU7 上的工作（**推断**，跨设备的隐式顺序没有保证）。对照 `0_to_7.cu`（§3.1）；
6. **启动配置 `launch_dims`**：`block = 256`，`grid = min(ceil(n / 256), 4096)`。n = 2^24 时不封顶需要 65,536 个 block，封顶后 4096 个，每个线程用 grid-stride 循环处理 16 个元素。封顶是**设计选择，不是硬件限制**（§3.1 已给出实测的 `maxGridSize[0]`）；
7. **模式用 `unsigned`**：2^24 个下标都能被精确表示，每个值唯一，避免 `0_to_7.cu` 里 `(float)i` 的错位漏检问题（§3.1）；
8. **没测的**：方式 A 的带宽和延迟——这个程序只验证功能正确。

<a id="3-4"></a>
### 3.4 跨进程：CUDA IPC（`ipc_tcp.cu`）

前面的程序都是**一个进程**同时用 GPU0 和 GPU7。如果是**两个进程**——进程 A 用 GPU0，进程 B 用 GPU7——要交换数据，程序怎么写？

**3.4.1 概念：IPC、句柄、两个角色**

- **IPC**（Inter-Process Communication，进程间通信）是操作系统层面的概念：每个进程有自己独立的地址空间，互相看不见对方的内存；要让两个进程共享数据，必须借助专门的机制（管道、共享内存、套接字……）。
- **CUDA IPC** 是 CUDA 提供的这样一套机制：让**不同进程**共享**同一块 GPU 显存**（也可以共享事件，本篇不涉及）。
- **为什么需要它**：每个进程都有自己的 **CUDA context**，显存的映射、指针的有效性、peer 授权都挂在各自的 context 上。同一块物理显存，对另一个进程默认不可见，把指针数值发过去也没有用。
- **IPC 句柄（`cudaIpcMemHandle_t`）**不是显存本身，也不是指针的包装，而是驱动为某块显存**签发的一张凭证**，可以类比成取件码或门禁卡：拿到它的进程，可以请驱动在自己的 context 里为那块显存建立映射。
- **两个角色**：**拥有方**（用 `cudaMalloc` 分配这块显存的进程，负责取句柄、最后释放；`ipc_tcp` 里是 server，GPU7）和**访问方**（拿到句柄、把显存映射进自己 context 的进程；`ipc_tcp` 里是 client，GPU0）。

**句柄的性质**（2026-10-31，server-0，对 `cudaMalloc` 出的显存取句柄）：

| 性质 | 结果 | 依据 |
|---|---|---|
| 大小固定、内容不透明 | `sizeof(cudaIpcMemHandle_t) = 64` 字节，内部格式不公开（本文不解读具体字段） | **实测** |
| **不是**指针的包装 | buf 的 8 字节指针数值**没有**原样出现在这 64 字节里 | **实测** |
| 由显存分配本身决定 | 对**同一块**显存再取一次句柄，逐字节相同；换一块显存，句柄不同 | **实测** |
| 可经任何主机侧通道传递 | 把句柄写进一个普通文件，两个**独立进程**读到后都能打开 | **实测** |
| 同一个句柄可被多个进程**同时**打开 | 一个用 GPU0、一个用 GPU1，同时打开同一个句柄、读 GPU7 的数据，都 PASS；两个进程映射出的指针数值各不相同 | **实测** |
| 在**拥有它的进程自己**里打开会失败 | `cudaIpcOpenMemHandle` 返回 `invalid device context (201)`；与"句柄是给另一个进程用的"一致，但具体原因本篇未隔离 | **实测** |
| 不携带数据 | 它只有 64 字节；数据仍然在 GPU 显存里，由访问方的 kernel 经 NVLink 读写 | 由大小与机制直接得出 |
| **只在同一台机器内有效** | CUDA 文档的说法。本机只验证了"同一个容器里的多个进程"；跨容器、跨机器**未测** | 文档说法，**未测** |
| 拥有方 `cudaFree` 后句柄失效 | 句柄只是凭证，不延长显存寿命 | **推断**，未测 |

**3.4.2 流程：不能把指针直接发给另一个进程，要用句柄**

显存映射、UVA、peer 授权都挂在**各自进程的 CUDA context** 上，对方 context 里没有这块显存的映射，所以必须走 CUDA IPC：

| 步骤 | 谁做 | API |
|---|---|---|
| ① 分配并填好数据 | 拥有方（GPU7） | `cudaMalloc` + kernel + `cudaDeviceSynchronize` |
| ② 取句柄 | 拥有方 | `cudaIpcGetMemHandle(&h, buf)`——句柄是不透明的 64 字节 |
| ③ 把句柄送给访问方 | 拥有方 → 访问方 | **主机侧**任意方式：pipe、TCP、共享文件…… |
| ④ 打开句柄 | 访问方（GPU0） | `cudaIpcOpenMemHandle(&remote, h, cudaIpcMemLazyEnablePeerAccess)`——在访问方自己的 context 里建立映射，返回本进程里有效的指针 |
| ⑤ 读写 | 访问方 | 访问方的 kernel 经 NVLink 直接读写拥有方的显存（即方式 A，kernel 与 §3.3 完全相同） |
| ⑥ 收尾 | 访问方先、拥有方后 | 访问方：`cudaIpcCloseMemHandle`，再通知拥有方；拥有方：收到通知后才 `cudaFree` |

- `cudaIpcMemLazyEnablePeerAccess` 让驱动在需要时自动完成 GPU0 与 GPU7 之间的 peer 授权，所以**不用**再手写 `cudaDeviceEnablePeerAccess`。
- 两个进程的 GPU 工作互不知情，**同步只能靠主机侧信号**：拥有方数据就绪并同步后才发句柄；访问方读写完并同步后才通知拥有方。
- 注意**两层"传递"别混**：句柄经主机通道（文件、管道、TCP）传递，这一步是控制信息；真正的数据始终留在 GPU 显存里，由访问方的 kernel 经 NVLink 读写。

**和 §3.3 的同进程方式对比**：

| | 同进程（§3.3 `remote_access.cu`） | 跨进程（本节） |
|---|---|---|
| 对端指针从哪来 | 本进程里 `cudaSetDevice(7)` + `cudaMalloc` 直接得到 | 拥有方取句柄 → 经主机通道传过来 → 访问方 `cudaIpcOpenMemHandle` |
| 授权 | 显式 `cudaDeviceEnablePeerAccess` | `cudaIpcOpenMemHandle` 的 `LazyEnablePeerAccess` 标志 |
| 传递什么 | 不需要传递 | 传句柄（64 字节，控制信息），**不是**传数据 |
| 同步 | 同一个进程里，`cudaDeviceSynchronize` 即可 | 两个进程的 GPU 工作互不可见，只能靠主机侧信号 |
| 访问的 kernel | 方式 A | **完全相同**的方式 A |

**3.4.3 示例程序 `ipc_tcp.cu`**

两个独立进程，`server`（GPU7，先启动）与 `client`（GPU0），控制信息走 **TCP**（`Msg{n, handle}` 与 1 字节完成信号）。数据布局：server 在 GPU7 上分配 2n 个 `unsigned`——`[0, n)` 是 server 给 client 的数据（值为 `i`），`[n, 2n)` 留给 client 写入（client 写 `i*3`），即**双向交换**。

```bash
nvcc -O3 -arch=native ipc_tcp.cu -o ipc_tcp
./ipc_tcp server 5555                  # 终端 1，先启动
./ipc_tcp client 127.0.0.1 5555        # 终端 2
```

关键代码骨架：

```cpp
// ===== server（GPU7，先启动）=====
cudaSetDevice(7);  cudaMalloc(&buf, 2*n*sizeof(unsigned));  init_kernel<<<...>>>(buf, n);
cudaDeviceSynchronize();                        // ① 先把数据准备好
cudaIpcGetMemHandle(&msg.handle, buf);          // ② 取句柄
socket → setsockopt(SO_REUSEADDR) → bind(0.0.0.0:port) → listen   // ③ 然后才开始监听
cs = accept(...);
send_all(cs, &msg, sizeof msg);                 // ④ 通过 TCP 发 {n, 句柄}
recv_all(cs, &done, 1);                         // ⑤ 等 client 的完成信号
/* 校验 client 写进来的数据 */  cudaFree(buf);  // ⑥ client 已关映射，才能释放

// ===== client（GPU0）=====
connect(host:port)  /* server 没起来就每 100 ms 重试，最多约 10 秒 */
recv_all(s, &msg, sizeof msg);                  // ① 收到句柄
cudaSetDevice(0);
cudaIpcOpenMemHandle(&remote, msg.handle, cudaIpcMemLazyEnablePeerAccess);  // ② 映射到本进程
remote_read <<<...>>>(remote, local, n);        // ③ GPU0 的 kernel 经 NVLink 读写 server 的显存
remote_write<<<...>>>(remote, n);
cudaDeviceSynchronize();
cudaIpcCloseMemHandle(remote);                  // ④ 先关映射
send_all(s, &done, 1);                          // ⑤ 再通过 TCP 通知 server
```

TCP 是字节流，一次 `send` / `recv` 不保证收发完整，所以程序里用 `send_all` / `recv_all` 循环到凑够字节数。

**3.4.4 实测结果**（2026-10-31，同一个 Pod 里的两个进程，各区域 16 MiB）：

```
# ipc_tcp（server 先启动，client 后启动）
[server] GPU7 上 buf = 0x7fae0e000000，数据已就绪
[server] 监听 0.0.0.0:5555，等待 client ...
[client] 收到句柄，n = 4194304
[client] 经 IPC 映射得到 remote = 0x7f3646000000
[client] server→client（client 读 server 的数据）：不匹配 0 个 -> PASS
[server] client→server（client 写进来的数据）：不匹配 0 个 -> PASS；原数据被改动 0 处
```

另一项对照：**让 client 先于 server 启动**，client 靠重试连接最终也成功（两边退出码都是 0）。

**3.4.5 要点**

1. **TCP 只传控制信息，数据不走 TCP**：句柄、元素个数、完成信号走 TCP；真正的数据由 client 的 kernel 经 NVLink 直接读写；
2. **指针数值在两个进程里不同**：实测 server 的 `buf = 0x7fae0e000000`，client 映射出的 `remote = 0x7f3646000000`。更重要的是：有效性取决于映射是否存在，而不取决于数值——直接把 server 的指针数值发给 client 用不了（机制说明；本篇不保留对照实验）；
3. **CUDA IPC 句柄只在同一台机器（同一个 OS / 容器的 IPC 命名空间）内有效**：TCP 在这里只是传句柄的信道。如果 client 在**另一台机器**上，TCP 照样能把句柄送过去，但 `cudaIpcOpenMemHandle` 打不开它——跨机要换方案（NCCL、MPI、RDMA/RoCE，即后续多机实验）。本机的两个进程在同一个容器里；**跨容器、跨机器未测**；
4. **释放顺序**：访问方先 `cudaIpcCloseMemHandle` 再通知，拥有方收到通知后才 `cudaFree`；
5. **server 先准备数据、再 `listen`**：这样"监听"日志出现就等于数据已就绪。实测 server 启动 1.5 秒后日志仍为空，说明其 CUDA 初始化与数据准备耗时超过 1.5 秒（未精确测量），所以 client 需要重试连接；
6. **没测的**：带宽；用 GPU 侧事件（`cudaIpcGetEventHandle`）代替主机信号做同步；client 异常退出时 server 的行为（程序里异常处理很简化）；
7. **旧式 IPC 的适用范围**（来自 CUDA 文档，**未实测**）：主要针对 `cudaMalloc` 分配的显存；用虚拟内存管理（`cuMemCreate`）分配的显存要改用 `cuMemExportToShareableHandle`。实际工程里，两个进程交换 GPU 数据更常用 NCCL 或 CUDA-aware MPI，它们在底层做的就是类似的事。

<a id="4"></a>
## 4. 调试：与本文实验相关的结论

环境准备、代码拷贝、debug 构建、命令行与 VS Code 调试的完整步骤见 **[debug.md](debug.md)**，本文不再重复。这里只留与 `0_to_7.cu` 直接相关、`debug.md` 里没有的结论。

**0_to_7 的 kernel 启动参数（实测）**：

```
$ cuda-gdb -batch -ex 'break fill_kernel' -ex 'run' -ex 'print threadIdx' --args ./0_to_7.debug
CUDA thread hit Breakpoint 1.2, fill_kernel<<<(4096,1,1),(256,1,1)>>>
    (p=0x7ff960000000, n=4294967296) at 0_to_7.cu:45
$1 = {x = 0, y = 0, z = 0}
```

可见 kernel 启动配置（grid = 4096、block = 256，即 §3.1 的"封顶 4096"）、UVA 地址形式的实参，以及 `n=4294967296` 正好是 2^32，即 16 GiB 的 float 数。

`remote_access` 的调试（断点位置，以及 `NO_ENABLE` 时 `cuda-gdb` 抓到的 `Warp Illegal Address`）见 [debug.md §4.4](debug.md#4)。

<a id="5"></a>
## 5. 速查：运行命令与坑

```bash
./0_to_7                                    # 16 GiB 固定，无参数
./remote_access                             # 方式 A：64 MiB，无参数，A1/A2 应均为 PASS
./remote_access_noenable                    # 对照版：应以 illegal memory access 退出（退出码 1）
./ipc_tcp server 5555                       # 终端 1：server（GPU7），先启动
./ipc_tcp client 127.0.0.1 5555             # 终端 2：client（GPU0）
nvidia-smi topo -m                          # 拓扑：GPU 两两 NV18 = 经 18 条 NVLink
nvidia-smi nvlink -s -i 0                   # GPU0 各链路速率与状态
nvidia-smi nvlink -gt d -i 0                # 各链路数据吞吐计数器（-gt r 含协议开销）
```

**同步与异步**

| # | 坑/注意 | 说明 |
|---|---|---|
| 1 | `cudaMemcpyPeer` 返回 ≠ 拷贝完成 | **实测**：调用约 12 µs 返回，传输在其后继续（§2.3）。要计时或确认完成，用事件 + `cudaEventSynchronize` 或 `cudaDeviceSynchronize`，**不要用 CPU 时钟包住这个调用** |
| 2 | 跨设备的异步顺序没有保证 | 在 GPU7 上启动 kernel 后，要在 GPU0 上读它写的数据之前，应显式同步（`remote_access.cu` 这样做了；`0_to_7.cu` 在 GPU0 填充与拷贝之间没有，属**未验证**的隐式依赖，§3.1） |
| 3 | kernel 启动错误在同步处才暴露 | `<<<>>>` 不返回错误码：启动后查 `cudaGetLastError()`，运行期错误（如非法访问）要到 `cudaDeviceSynchronize()` 才出现（§3.3） |

**授权与指针**

| # | 坑/注意 | 说明 |
|---|---|---|
| 4 | kernel 远程解引用不 `enable` 必崩 | **实测**（§3.3）：`an illegal memory access was encountered`；在 `cuda-gdb` 下停在 `CUDA Exception: Warp Illegal Address`。`0_to_7.cu` 本身不做远程解引用，但先做 `can` 检查（§2.2） |
| 5 | `cudaMemcpyPeer` 与 peer enable 的关系 | copy API 不需要 enable 也能完成（自动 host staging），但快路径需要 enable；降级慢多少本篇未测（§2.2） |
| 6 | 指针值看不出属于哪张卡 | 对端指针和本地指针在数值上没有区别，"属于谁"记在驱动里（`cudaPointerGetAttributes`，§2.1）；能不能访问取决于 `enable`，不取决于指针（§3.3） |

**跨进程**

| # | 坑/注意 | 说明 |
|---|---|---|
| 7 | 指针不能直接传给另一个进程 | 机制说明（§3.4）：指针只在本进程的 CUDA context 里有效，把数值发给另一个进程不能用；要用 `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`（本篇不保留对照实验） |
| 8 | CUDA IPC 只在同一台机器内有效 | TCP 能把句柄传到别的机器，但句柄打不开；跨机用 NCCL / MPI / RDMA。跨容器、跨机本篇**未测**（§3.4） |
| 9 | IPC 的顺序要求 | 访问方先 `cudaIpcCloseMemHandle` 再通知，拥有方收到后才 `cudaFree`；两个进程的 GPU 工作互不可见，同步靠主机侧信号（§3.4） |
| 10 | 句柄是凭证，不是数据，也不延长显存寿命 | 句柄只有 64 字节，数据仍留在 GPU 显存里；拥有方 `cudaFree` 之后句柄就指向不存在的显存（推断，未测）（§3.4） |
| 11 | TCP 是字节流 | 一次 `send` / `recv` 不保证收发完整，要循环到凑够字节数（`send_all` / `recv_all`，§3.4） |

**度量**

| # | 坑/注意 | 说明 |
|---|---|---|
| 12 | GiB 与 GB | `0_to_7` 的 `16 << 30` 是 16 GiB，打印的是十进制 17.2 GB（§3.2） |
| 13 | 吞吐计数器是累计值 | `nvidia-smi nvlink -gt` 读到的是累计量（实测已达 ~4×10^8 KiB，TB 量级），且 `-r` 重置选项在帮助里已标记弃用；测一次传输要**前后各读一次再相减**（§1.6） |

<a id="A"></a>
## 附录 A. 验证状态汇总

本文每个结论的可靠程度汇总于此；正文里的标注与这里一致。**实测** = 在 server-0 上跑出来的；**推断** = 由机制或公开资料推导；**未测 / 未验证** = 没有证据。

定稿前已按当前源码重新编译并运行全部程序（`0_to_7` 三次；`remote_access` 的正常版与 `NO_ENABLE` 版；`ipc_tcp` 的 server 先启动、client 先启动两种顺序），并用 `cuda-gdb` 再次捕获了 `NO_ENABLE` 版的 `Warp Illegal Address`，文中的实测输出与之一致。

**硬件与拓扑**

| 结论 | 状态 |
|---|---|
| GPU 两两 NV18、GPU0 有 18 条链路、`Fabric State : Completed` | **实测** |
| B300 有 4 个 Copy Engine、148 个 SM | **实测** |
| `nvlink -c` 的能力项：P2P、P2P 原子、系统内存访问均支持 | **实测**（系统内存访问是否实际可用：未验证） |

**带宽与计数器**

| 结论 | 状态 |
|---|---|
| `0_to_7` 带宽 779.4 GB/s（16 GiB 一次，3 次迭代平均） | **实测** |
| 传输计数器：Data Tx ≈ 68.82 GB、Raw Tx ≈ 69.12 GB、Raw Rx ≈ 0.18 GB | **实测** |
| 单向理论 ~900 GB/s（18 条链路 × 单向 50 GB/s） | **推断**（公开规格，未在本机验证） |

**API 语义与行为**

| 结论 | 状态 |
|---|---|
| `cudaMemcpyPeer` 对 host 异步：调用约 12 µs 返回，传输约 5.5 ms 后完成（4 GiB） | **实测**（§2.3） |
| kernel 远程读、远程写对端显存的结果正确（`remote_access`，64 MiB，A1/A2 均 PASS） | **实测**（§3.3） |
| 不 `enable` 时 kernel 访问对端显存触发 `illegal memory access`；`cuda-gdb` 报 `CUDA Exception: Warp Illegal Address` | **实测**（§3.3） |
| 指针只是 64 位整数，"属于哪张卡"记在驱动里（`cudaPointerGetAttributes` 报 `device=7`） | **实测**（§2.1） |
| `maxGridSize[0] = 2147483647`（grid 封顶 4096 是设计选择，不是硬件限制） | **实测**（§3.1） |
| `(float)i` 在 i ≥ 2^24 后相邻下标共用同一个值（从 2^28 起连续 1024 个下标只有 33 个不同值，从 2^31 起只有 5 个） | **实测**（float32 实际计算，§3.1） |
| `cudaMemcpyPeer` 在没有 `enable` 时自动降级走主机中转、功能不受影响 | 依据程序注释，**未实测**；降级速度**未测** |

**内部机制（推断）**

| 结论 | 状态 |
|---|---|
| `enable` 在本卡页表里登记对端显存的映射；NVLink 包结构与路由方式 | **推断**（公开资料） |
| 读请求/响应、写请求/确认的线上流程；Raw Rx 回流是确认与流控 | **推断** |
| `cudaMemcpyPeer` 由 Copy Engine 执行 | **推断**（驱动在某些情况下也可能改用 kernel 完成拷贝，本机未验证） |
| 具体由源端还是目的端的 Copy Engine 执行 | **未验证** |
| 延迟被大量并发请求掩盖 | **推断** |
| NVLS（NVSwitch 芯片内归约） | **推断**，未在本机观察 |
| 请求中携带的地址格式 | **未验证** |

**跨进程 IPC**

| 结论 | 状态 |
|---|---|
| 两个进程（server 用 GPU7，client 用 GPU0）经 `cudaIpc*` 双向交换数据，结果正确；client 先于 server 启动时靠重试也成功 | **实测**（§3.4） |
| IPC 句柄为 64 字节；不含指针数值；同一块显存多次取句柄相同、不同显存不同 | **实测**（§3.4） |
| 同一个句柄经文件传给两个独立进程，可被同时打开并读到正确数据；在拥有它的进程自己里打开会报错 `invalid device context (201)` | **实测**（§3.4；报错的具体原因未隔离） |
| 拥有方 `cudaFree` 后句柄失效 | **推断**，未测 |
| 直接把指针数值发给另一个进程不能用（映射挂在各自进程的 context 上） | **推断**（机制说明；本篇不保留对照实验） |
| CUDA IPC 句柄只在同一台机器内有效；跨容器、跨机器是否可用 | **未测**（文档说法） |

**明确没有测的**

| 项目 |
|---|
| 方式 A 的带宽与延迟、A 与 B 的带宽对比 |
| 降级路径（无 `enable` 的 `cudaMemcpyPeer`）的速度 |
| 用 GPU 侧事件（`cudaIpcGetEventHandle`）做跨进程同步 |
| client 异常退出时 server 的行为 |
| 旧式 IPC 对非 `cudaMalloc` 分配显存的适用范围（文档说法） |

<a id="B"></a>
## 附录 B. 已知局限与后续工作

| 项目 | 现状 | 建议 |
|---|---|---|
| `0_to_7.cu` 的校验模式 | `(float)i` 在后一半下标上相邻元素共用同一个值，对"错位"的检出力弱（§3.1） | 改成 `(unsigned)i`（`remote_access.cu` 已采用）；改后应重新实测 PASS，并人为注入错位，确认能检出 |
| `0_to_7.cu` 拷贝前缺显式同步 | `fill_kernel` 之后没有 `cudaDeviceSynchronize`，依赖拷贝与默认流之间的隐式顺序（§3.1） | 在 `launch_fill` 之后加同步 |
| 方式 A 的带宽与延迟 | `remote_access.cu` 只验证功能，不计时（§3.3） | 加事件计时，与方式 B 同口径比较 |
| 降级路径的速度 | 没有 `enable` 的 `cudaMemcpyPeer` 慢多少未测（§2.2） | 去掉 `enable_peer` 再跑 `0_to_7` |
| CUDA IPC 的边界 | 跨容器、跨机器未测；`cudaIpcGetEventHandle` 未演示（§3.4） | 跨机部分由后续的多机（RoCE）实验承接 |
| `ipc_tcp.cu` 的健壮性 | 异常处理简化：client 中途崩溃、`SIGPIPE`、监听 `0.0.0.0` 等（程序注释已逐项说明） | 真实环境需补错误处理，并只绑 `127.0.0.1` 或加认证 |

---

*环境：server-0（StatefulSet `examples/n-server.yaml` / 调试用 `n-server1.yaml`，8×B300 SXM6, NV18 全互联, CUDA 13.0, nvcc 13.0）；实验与调试验证日期 2026-10-26/27、2026-10-31；代码：`p2p/0_to_7.cu`、`p2p/remote_access.cu`、`p2p/ipc_tcp.cu`，均含中文注释。*

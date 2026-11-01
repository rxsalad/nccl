# C/C++ 直连 CUDA：产物与编译

一个 `.cu` 文件怎么变成**只依赖驱动**的独立可执行文件？本篇回答三个问题：

1. 编出来的 978 KB 可执行文件里有什么，运行起来靠什么？（§1、§2）
2. 怎么**证明** CUDA Runtime 是静态链入、`libcuda.so` 是运行期动态加载？（§2）
3. nvcc 一条命令背后做了什么，`-arch`、`-G` 等开关怎么选？（§3）

素材来自 `p2p/0_to_7.cu`（仅作为"被编译的样例程序"，关键代码见下面的"样例程序"一节）。调试见 [debug.md](debug.md)。

## 约定

- **实测**：在 server-0 Pod（nvcc 13.0，8×B300，sm_103）上的命令输出；**推断**：由机制推导。
- **跟敲**：跟着文档在终端里逐条敲命令，亲手重现结果。命令均在 `/vllm-workspace` 下执行，二进制需先编译出来（见附录 A）。
- **版本号随节点变化**：文中出现的驱动版本（`590.48.01`）是写文档时那个节点的；Pod 重建后可能调度到别的节点，版本可能不同。以你机器上的实际输出为准。

## 本篇用到的工具

| 工具 | 用途 | 是否执行目标文件 |
|---|---|---|
| `readelf` | 查看 ELF **文件里写了什么**：头、section、动态依赖（`-d`）、导入符号（`--dyn-syms`） | 否（纯静态查看，永远安全） |
| `nm` | 看符号表：某函数是已定义（`t`/`T`）还是未定义（`U`） | 否 |
| `strings` | 抽取文件里的可读字符串 | 否 |
| `ldd` | 列出启动时**会装载**的动态库（含间接依赖） | 某些实现会执行目标文件的 loader，**不要对不可信二进制使用** |
| `/proc/PID/maps` | 看进程**运行后**实际映射了哪些库 | — |
| `cuobjdump` / `nvdisasm` | 查看 fatbin 内容、反汇编 SASS | 否 |
| `nvcc --dryrun` | 只打印 nvcc 将执行的命令 | 否 |

一句话区分：`readelf` 看文件里**写了什么**，`ldd` 看启动时**会装什么**，`/proc/PID/maps` 看运行后**装了什么**。

<a id="sample"></a>
## 样例程序：0_to_7.cu 的关键代码

本文拆解的二进制，由 `p2p/0_to_7.cu`（227 行，含中文注释）编出。它做的事：GPU0 通过 `cudaMemcpyPeer` 向 GPU7 传 16 GiB，计时，再在 GPU7 上校验。**设计细节与实测带宽**见 [c-single.md §3.1](c-single.md#3-1)，本篇只关心"这份源码编出了什么"，所以这里只给出后文会反复引用的几段关键代码（源码行号以当前文件为准）。

**① 填充 kernel `fill_kernel`**（第 44–48 行；§3.2 的 SASS 对照表就是它）：

```cpp
__global__ void fill_kernel(float* __restrict__ p, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程 id
  size_t stride = (size_t)gridDim.x * blockDim.x;            // grid 总步长
  for (; i < n; i += stride) p[i] = (float)i;                 // 填下标模式
}
```

这是典型的 **grid-stride 循环**：线程数可以少于元素数，每个线程隔 `stride` 个取一个，直到越界。

**② 校验 kernel `check_kernel`**（§3.2 要点 4 里的 `atomicAdd` 就是它）：

```cpp
__global__ void check_kernel(const float* __restrict__ p, size_t n,
                            unsigned long long* bad) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  unsigned long long local = 0;               // 本线程数到的不匹配数
  for (; i < n; i += stride)
    if (p[i] != (float)i) local++;
  if (local) atomicAdd(bad, local);            // 各线程计数原子累加到全局
}
```

**③ kernel 的启动**（`<<<>>>` 在 §3.1 的流水线里会被改写成 `cudaLaunchKernel`，在 §1.2 的调用链里属于 ① 层）：

```cpp
static void launch_fill(float* p, size_t n) {
  int block = 256;
  size_t grid = (n + block - 1) / block;
  if (grid > 4096) grid = 4096;
  fill_kernel<<<(int)grid, block>>>(p, n);
  CHECK_CUDA(cudaGetLastError());  // <<<>>> 不返回错误码，启动错误只能这样查
}
```

**④ `main` 的骨架**（每一个 `cuda*` 调用，都对应 §1.2 调用链里的 ② 层 `libcudart`，运行时再经 `dlopen` 进入 `libcuda`，见 §2.4）：

```cpp
int main() {
  // ① 检查：两个方向的 P2P 可达性，并授权
  cudaDeviceCanAccessPeer(&can07, 0, 7);  cudaDeviceCanAccessPeer(&can70, 7, 0);
  enable_peer(0, 7);  enable_peer(7, 0);   // 内部：cudaSetDevice(dev) + cudaDeviceEnablePeerAccess(peer, 0)

  // ② 准备：GPU0 上分配 16 GiB 并填充；GPU7 上分配接收缓冲并清零
  cudaSetDevice(0);  cudaMalloc(&src, BYTES);  launch_fill(src, N);
  cudaSetDevice(7);  cudaMalloc(&dst, BYTES);  cudaMemset(dst, 0, BYTES);
                     cudaMalloc(&bad, sizeof(*bad));  cudaMemset(bad, 0, sizeof(*bad));

  // ③ 传输：warmup 一次，再用 CUDA 事件计时 3 次
  cudaMemcpyPeer(dst, 7, src, 0, BYTES);  cudaDeviceSynchronize();
  cudaEventRecord(t0);
  for (int i = 0; i < iters; i++) cudaMemcpyPeer(dst, 7, src, 0, BYTES);
  cudaEventRecord(t1);  cudaEventSynchronize(t1);  cudaEventElapsedTime(&ms, t0, t1);

  // ④ 校验：在 GPU7 本地数"不等于模式"的元素，只把 8 字节计数器拷回 host
  launch_check(dst, N, bad);  cudaDeviceSynchronize();
  cudaMemcpy(&h_bad, bad, sizeof(h_bad), cudaMemcpyDeviceToHost);
}
```

**这些代码在本文哪里被用到**：

| 代码 | 本文里的位置 |
|---|---|
| `fill_kernel` 的源码 → SASS | §3.2：逐行对照 `S2R`、`IMAD.WIDE.U32`、`I2F.U64`、`STG.E` 等指令 |
| `check_kernel` 的 `atomicAdd` | §3.2 要点 4：被编译成 `REDG.E.ADD.64.STRONG.GPU` |
| `<<<grid, block>>>` | §3.1 阶段 7（host 编译，被改写成 `cudaLaunchKernel`）；§1.2 的 ① 层 |
| `cudaMalloc`、`cudaMemcpyPeer` 等 Runtime 调用 | §2.3 证明 libcudart 静态链入（`nm` 里 `cudaMalloc` 是 `t`）；§2.4 运行时经 `dlopen` 进入 libcuda |
| 整个源文件的编译 | §3.1 的 8 阶段流水线；§3.3 的 `-arch` 开关；§3.4 的 release / debug / profile 三种构建 |

## TOC

- [样例程序：0_to_7.cu 的关键代码](#sample)
- [1. 全貌：一条命令、一个产物、五层调用链](#1)
- [2. 产物解剖：ELF、静态链接与 dlopen](#2)
- [3. 编译期：流水线、SASS 与两组开关](#3)
- [附录 A. 速查：编译命令与坑](#A)

<a id="1"></a>
## 1. 全貌：一条命令、一个产物、五层调用链

### 1.1 一条编译命令

```
nvcc -O3 -arch=native 0_to_7.cu -o 0_to_7
 │     │       │         │        │
 │     │       │         │        └─ 输出名（给最终链接步骤，§3.1 阶段 8）
 │     │       │         └─ 输入源文件（host/device 混合体）
 │     │       └─ 架构目标（§3.3）
 │     └─ host 代码优化级别（§3.4）
 └─ 编译驱动器——它自己不是编译器（§3.1）
```

产物（**实测**）：6.6 s 编译出一个 **978 KB 的独立 ELF 可执行文件**。它自持程度比看上去还高：CUDA Runtime 静态打进二进制（`ldd` 零 cudart 项），运行时 GPU 侧只靠 `dlopen` 装载驱动 `libcuda.so`。

### 1.2 运行起来的五层调用链

```
① C++ host 层   main() / CHECK_CUDA / cudaEvent 计时       ← 你全权掌控
② API 层        libcudart（本产物中静态链入）
③ 驱动层        libcuda.so（用户态驱动）→ ioctl → 内核态驱动
④ 硬件层        命令写入 GPU 推送缓冲区
⑤ GPU 执行层    fill_kernel / check_kernel 的 sm_103 SASS 在 SM 上跑
```

以 `cudaMalloc` 和 `kernel<<<grid,block>>>(...)` 为例，各层分别是：

| 层 | 是什么 | 与上一层怎么连 | 在二进制里吗 |
|---|---|---|---|
| ① host | 你写的 `main()` 等普通 CPU 代码。`kernel<<<>>>` 不是函数调用，nvcc 把它改写成 `cudaLaunchKernel`。**启动是异步的**：host 交出任务就返回，所以计时要用 `cudaEvent` 而不是 CPU 时钟 | — | 是（你的代码） |
| ② API（libcudart） | `cudaMalloc`、`cudaMemcpy`、`cudaEventRecord` 的实现；管理 fatbin 注册，把 Runtime 调用翻译成 Driver API（`cuMemAlloc`、`cuLaunchKernel`）。**不直接碰硬件** | 普通函数调用 | **是，静态链入** |
| ③ 驱动（libcuda.so） | 用户态驱动：管理 context 与虚拟地址空间，把 kernel 启动编码成 GPU 命令；需要特权的操作（分配显存、建映射）再用 `ioctl` 进入内核态驱动 `nvidia.ko` | **运行期 `dlopen`** | **否，动态加载**，来自宿主机 |
| ④ 硬件 | 命令写入**推送缓冲区**（CPU 与 GPU 都能访问的内存），再写 GPU 寄存器（doorbell）通知它；此后 GPU 自己取命令，CPU 已可去做别的事——异步的物理基础 | MMIO / DMA | — |
| ⑤ GPU 执行 | 命令处理器把 grid 拆成 block 分给空闲 SM；每 32 线程一个 warp 执行 **SASS**（原生机器码）。`sm_103` 表示编的就是 B300 这一代 | 硬件调度 | SASS 在 `.nv_fatbin` 里（§2.1） |

一次 kernel 启动的完整路径：

```
<<<>>> → cudaLaunchKernel（②）→ cuLaunchKernel（③，经 dlopen 得到的函数指针）
       → 命令写入推送缓冲区并通知 GPU（④）→ host 立即返回
       → GPU 在 SM 上执行 SASS（⑤）→ cudaEventSynchronize 让 host 等到完成
```

没有解释器、没有绑定层——这是所有 GPU 软件栈的最底层用户态形态，其余一切（框架、运行时系统）都建在它上面。

### 1.3 "链接"在这张图里的三种含义

这是最容易混淆的地方：

- ② **静态链接**：libcudart 的代码被拷进了二进制。
- ③ **运行期动态加载**：libcuda 不是链接期依赖，而是程序运行时 `dlopen` 进来的。
- ③ → 内核驱动：是 `ioctl` 系统调用，根本不属于链接问题。

§2 逐一用命令证明。

<a id="2"></a>
## 2. 产物解剖：ELF、静态链接与 dlopen（跟敲）

按"文件里有什么 → 依赖怎么分类 → 两个库各自怎么证明 → 版本跟谁走 → 装载全过程"的顺序展开。

### 2.1 文件本身：ELF 头与 fatbin

**ELF**（Executable and Linkable Format）是 Linux 上 `.o` / `.so` / 可执行文件的统一格式，一种格式服务两个阶段：链接期（链接器按 section 拼装）和运行期（内核 loader 按 program header 装载）。

**ELF 头——身份与入口**：

```bash
$ readelf -h 0_to_7 | grep -E 'Class|Type|Machine|Entry'
Class:                             ELF64
Type:                              DYN (Position-Independent Executable file)  ← PIE，基址随机化(ASLR)
Machine:                           Advanced Micro Devices X86-64
Entry point address:               0x9410           ← 内核交出控制权的起点
```

| 字段 | 值 | 含义 |
|---|---|---|
| **Class** | `ELF64` | 64 位程序，地址宽度 64 位 |
| **Type** | `DYN (Position-Independent Executable file)` | **PIE**（位置无关可执行文件）：可被加载到内存任意地址，每次运行系统随机选基址（**ASLR**，地址空间随机化），攻击者难以预测代码位置。显示 `DYN` 而非 `EXEC`，是因为它像动态库一样可重定位；现代发行版的 gcc 默认生成 PIE |
| **Machine** | `Advanced Micro Devices X86-64` | 目标 CPU 架构是 x86-64。名字里的 AMD 只是历史原因（64 位指令集最早由 AMD 设计），Intel CPU 同样用这个标识 |
| **Entry point address** | `0x9410` | **入口地址**：程序加载后内核把控制权交给的第一条指令的位置。这是**偏移量**，实际地址 = 随机基址 + `0x9410`。入口通常是 `_start`，它先做初始化，最后才调用 `main()` |

注意：ELF 头只描述 **host（CPU）侧**的可执行文件，GPU 机器码不体现在这里。

**SASS 藏身处**——§3.1"打包"阶段的落点：

```bash
$ readelf -S 0_to_7 | grep -iE 'nv_|fatbin'
[18] .nv_fatbin        PROGBITS  000000000008fdf0  ...   ← fatbin 容器本体
[29] .nvFatBinSegment  PROGBITS  00000000000ad058  ...   ← 注册桩引用的入口
```

978 KB 里的两个 CUDA 专属 section——SASS 就躺在这里。

### 2.2 动态依赖：NEEDED 与三种"链接"方式

```bash
$ readelf -d 0_to_7 | grep NEEDED          # 链接期声明的直接依赖（纯静态查看）
 Shared library: [libstdc++.so.6]
 Shared library: [libgcc_s.so.1]
 Shared library: [libc.so.6]
 Shared library: [ld-linux-x86-64.so.2]
$ ldd 0_to_7                               # 启动时实际装载的完整闭包——实测与上表一致
```

**没有 libcuda，也没有 libcudart。** 但"没有"有两种完全不同的原因——这正是三种"链接"方式要区分的。

**核心问题**：程序要调用一个函数（`cudaMalloc`、`printf`、`cuLaunchKernel`……），这个函数的**代码从哪来**，又是**什么时候接上**的？

| | 方式 1：静态链接 | 方式 2：链接期动态 | 方式 3：运行期动态 |
|---|---|---|---|
| 一句话 | 编译时把库的二进制**拷进**应用程序 | 启动时由系统按清单装载 | 运行到那一步，程序**自己**去加载 |
| 库的代码在哪 | **在你的可执行文件里** | 独立的 `.so` 文件 | 独立的 `.so` 文件 |
| 谁来接上 | 链接器 | 系统的 `ld.so` | 程序自己调用 `dlopen` / `dlsym` |
| 何时接上 | **编译时** | **进程启动时** | **第一次用到时** |
| 可执行文件里记录什么 | 库的代码本身 | 一条依赖声明（`DT_NEEDED`） | 库名字符串 + 对 `dlopen` 的导入 |
| `ldd` / `readelf -d` 能否看到 | 看不到（已在文件内部） | **能** | **看不到**（运行后只在 `/proc/PID/maps` 里） |
| 库缺失时（**推断**） | 不受影响，自带 | **启动就失败** | 程序能启动，**用到时才失败** |
| 库升级后 | 须重新编译才生效 | 直接生效 | 直接生效 |
| 可执行文件体积 | 大 | 小 | 小 |
| **本程序里的例子** | **libcudart** | `libc`、`libstdc++`；以及用 `-lcuda` 显式链接的 driver API 程序（§2.6） | **libcuda.so**（runtime API 程序，如 `0_to_7`） |

**打个比方**（做菜要用一种酱料）：

- 方式 1：做菜时就把酱料**调好倒进菜里**，端出来的菜自带酱料，吃的时候不需要任何外部东西；
- 方式 2：菜单上写明"配 X 酱料"，**开席前**服务员（`ld.so`）必须把酱料备齐，备不齐就不开席；
- 方式 3：菜单上没写，**吃到某一步**厨师（程序自己）才临时去仓库取酱料，取不到那一刻才出问题。

**放到时间线上**：

```
编译 / 链接 ──────── 启动 ─────────────── 运行中
  │                    │                      │
  方式 1：             方式 2：               方式 3：
  libcudart 的代码     ld.so 按清单装载        第一次 CUDA 调用时，
  被拷进应用程序        libc、libstdc++        cudart 调 dlopen("libcuda.so.1")
```

**本程序的完整接法**：应用程序 `0_to_7` **静态链接 libcudart——把它的二进制代码拷进了自己**（方式 1）；启动时系统按清单装 `libc` 等（方式 2）；运行到第一次 CUDA 调用，这份已经在你文件里的 cudart 再去 `dlopen` 加载 `libcuda.so`（方式 3）。所以 libcudart 与 libcuda 是**同一个程序里两种不同的接法**。

因此，"`ldd` 里没有 libcuda"**不等于**"没有用 libcuda"——它走的是方式 3，而 `ldd` 只能看到方式 2。下面两节分别用命令证明。

### 2.3 证明 libcudart 是静态链入（方式 1）

NEEDED 里没有 libcudart 只是**反面证据**（只能说明不是动态链接；一个根本没用 CUDA 的程序也会如此）。要**正面**证明，需要看到 cudart 的代码本体在文件里：

```bash
$ nm 0_to_7 | grep -w cudaMalloc
000000000004d420 t cudaMalloc          ← 小写 t：本文件内已定义（最有力的一条）
$ nm -D 0_to_7 | grep -c cudart        # 动态导入表里的 cudart 条目数
0
$ strings -a 0_to_7 | grep -m1 libcudart
libcudart_static_955d0e854921d49ef397edbaffbaf12e1afbf272   ← 静态版 cudart 的内部标识
```

| 证据 | 说明了什么 | 强度 |
|---|---|---|
| `nm` 里 `cudaMalloc` 是 `t` | cudart 代码在二进制里 | **正面，最强** |
| `libcudart_static_<哈希>` 字符串 | 静态版 cudart 的特征 | 辅助 |
| NEEDED 里无 `libcudart.so`、`nm -D` 里 cudart 为 0 | 启动时不依赖、也不从外部导入 | 反面，需与上两条配合 |

对照：若是动态链接，`nm` 里 `cudaMalloc` 会是大写 `U`（未定义，要去别的库找），且 NEEDED 里会多出 `libcudart.so.13`。

**拷进去的到底是什么、拷了多少**：

- 拷的是静态库 `libcudart_static.a`（nvcc 默认链接它，见 §3.1 阶段 8 的 `-lcudart_static`）里的**目标代码**，由链接器在编译时放进可执行文件。运行时不再需要镜像里的 cudart 副本（§2.5）。
- **不是整个库原样拷进去。** 实测（2026-10-31）：`libcudart_static.a` 为 1.3 MB，而整个 `0_to_7`（还含你的 host 代码和 `.nv_fatbin`）才约 978 KB，说明没被引用的部分被裁掉了。（`ar t` 显示这个 `.a` 里只有 1 个成员，所以不是"按文件挑选"；具体由链接器的裁剪机制完成——这是**推断**，未用 `--gc-sections` 等选项验证。）
- 对照：动态版 `libcudart.so.13.0.96` 为 688 KB，是独立文件，不会进入你的二进制。

### 2.4 证明 libcuda 是运行期动态加载（方式 3）

**结论：libcuda 是动态的，而且是方式 3——运行期 `dlopen`，不是链接期依赖。** 静态链入的 cudart 在第一次 CUDA 调用时执行 `dlopen("libcuda.so.1")`，再用 `dlsym` 取出 `cuLaunchKernel` 等函数地址，之后经函数指针调用。证据分三步：

**① 运行前：导入表里有 dlopen 的指纹**

```bash
$ readelf --dyn-syms 0_to_7 | grep -E 'dlopen|dlsym'
 77: 0000000000000000 0 FUNC GLOBAL DEFAULT UND dlsym@GLIBC_2.34
108: 0000000000000000 0 FUNC GLOBAL DEFAULT UND dlopen@GLIBC_2.34
```

**`dlopen` 是 libc 里的函数，不是命令**（`nm -D /lib/x86_64-linux-gnu/libc.so.6 | grep -w dlopen` 可见其住址；在 shell 里敲 `dlopen` 只会得到 command not found）。它是动态链接的第二个时机：load-time 的 `DT_NEEDED` 由 `ld.so` 启动时自动装载，run-time 由程序自己 `dlopen` 打开、`dlsym` 按名字取函数地址。`UND`（本文件无定义、需从 libc 导入）的调用者正是**静态链入的 cudart**——不跑起来就能看到 dlopen 踪迹。

**② 运行中：地址空间里现身**

```bash
$ cd /vllm-workspace
$ ./0_to_7 >/dev/null 2>&1 &                # 进程只存活几秒：用轮询抢在它退出前观察
$ while :; do PID=$(pgrep -x 0_to_7); [ -n "$PID" ] && grep -q libcuda /proc/$PID/maps && break; sleep 0.05; done
$ grep libcuda /proc/$PID/maps | head -1
/usr/lib/x86_64-linux-gnu/libcuda.so.590.48.01   ← 依赖列表里没有，地址空间里有
$ wait                                      # 等它自然结束
```

同一次观察中（2026-10-31，实测）：`grep -E 'libcuda|libcudart' /proc/PID/maps` 只有 `libcuda.so.590.48.01`，**没有 libcudart**——cudart 在二进制内部，不会作为独立库出现。进程还持有 13 个 `/dev/nvidia*` 文件描述符，这是 libcuda 通过 `ioctl` 与内核态驱动通信的通道（各描述符的具体用途未细查）。

**③ libcuda 没有静态形态，且不是镜像的一部分**

| | libcudart | libcuda |
|---|---|---|
| 归属 | CUDA toolkit | NVIDIA **驱动**（用户态组件，与内核态驱动配对） |
| 静态形态 | 有：`libcudart_static.a`（nvcc 默认） | **无**：只发布 .so |
| 版本跟着谁走 | toolkit（13.0） | 装机驱动（随节点变化） |
| 容器里怎么来 | 镜像自带（`/usr/local/cuda/targets/x86_64-linux/lib/`） | 宿主机 bind-mount 注入，**镜像里没有** |

```bash
$ ls /usr/lib/x86_64-linux-gnu/libcuda.a       # 静态形态不存在
No such file or directory
$ ls -la /usr/lib/x86_64-linux-gnu/libcuda.so  # 驱动版本命名的符号链
libcuda.so -> libcuda.so.1 -> libcuda.so.590.48.01
$ grep libcuda /proc/mounts | head -1           # 从宿主机只读挂载进容器
/dev/vda1 /usr/lib/x86_64-linux-gnu/libcuda.so.590.48.01 ext4 ro,...
```

**为什么必须这样设计**：libcuda 负责把 API 调用翻译成 `ioctl` 交给内核态驱动，两者**同版本配对**。若可静态链入，每次驱动升级都要重编所有应用；**动态 + dlopen 换来"编译一次，任何 ≥ 最低版本驱动的机器都能跑"**（NVIDIA 驱动向前兼容模型）。

### 2.5 版本跟谁走：三段式供应链

| 组件 | 版本跟谁走 | 什么时候固定 |
|---|---|---|
| SASS（sm_103） | 编译机的 GPU | 编译时（`-arch`，§3.3） |
| libcudart | **容器镜像**（toolkit 13.0 → `libcudart.so.13.0.96`，文件日期 = 镜像构建日） | 镜像构建时；换 toolkit 版本 = 换镜像 |
| libcuda | **宿主机驱动**（bind-mount 注入，见 §2.4 ③） | 节点驱动升级时——与应用、镜像都无关 |

"跟镜像走"的两层含义（实测）：

1. **构建时**：`/usr/local/cuda` 在镜像层里（`grep cudart /proc/mounts` 为空——不是挂载），nvcc 从这里取 `-lcudart_static`；
2. **链入后**：静态链入意味着 cudart 的代码**拷进了二进制**——运行时连镜像里的副本都不再需要（`ldd` 零 cudart 项的根本原因）。镜像的使命在链接完成那一刻就结束了。

唯一的配对约束：镜像里的 toolkit 要求宿主机驱动 ≥ 最低版本（CUDA 13.0 ↔ 580+）。一个跟镜像走、一个跟节点走，向前兼容模型的作用就是让这两条版本线尽量解耦。

### 2.6 对照：driver API 程序 vs runtime API 程序

同一份 libcuda，两条装载路径：

| | driver API 程序（`-lcuda` 显式链接） | runtime API 程序（`0_to_7`） |
|---|---|---|
| 链接期 | `DT_NEEDED` 里**有** `libcuda.so.1` | NEEDED 里**没有** libcuda |
| 装载者 / 时机 | `ld.so`，进程启动时（load-time，方式 2） | cudart，第一次 CUDA 调用时（run-time，dlopen，方式 3） |
| 驱动库缺失时的失败点（**推断**） | 启动即失败（`error while loading shared libraries`） | 第一个 CUDA 调用才失败 |

跟敲——写一个最小的 driver API 程序作对照：

```bash
$ cat > /tmp/drv.c << 'EOF'
#include <cuda.h>
#include <stdio.h>
int main() {
    cuInit(0);
    int n = 0; cuDeviceGetCount(&n);
    printf("driver API sees %d GPUs\n", n);
    return 0;
}
EOF
$ gcc /tmp/drv.c -o /tmp/drv -I/usr/local/cuda/include -lcuda
$ readelf -d /tmp/drv | grep NEEDED            # libcuda.so.1 在列
 Shared library: [libcuda.so.1]
 Shared library: [libc.so.6]
$ /tmp/drv
driver API sees 8 GPUs
$ readelf -d 0_to_7 | grep NEEDED              # 对照：没有 libcuda（→ §2.4 ② 运行中才现身）
```

两条路径殊途同归：进程地址空间里都是 `/usr/lib/.../libcuda.so.<驱动版本>`，且**都是动态的——libcuda 没有任何静态形态**。区别只在装载时机与失败时机：runtime API 程序对驱动版本零感知（dlopen 找到什么用什么），driver API 程序则要求目标机器启动前就存在 soname 为 `libcuda.so.1` 的文件。

### 2.7 装载全链：把上面的切面串成一条时间线

```
execve(0_to_7)
  → 内核解析 ELF header，按 program header mmap 段         ← §2.1
  → ld.so 按 DT_NEEDED 装载 libstdc++/libc               ← §2.2 ldd 看到的就是这层
  → 跳到 entry 0x9410 → main()
  → 第一个 CUDA 调用：cudart（静态链入，已在二进制里）      ← §2.3
      dlopen("libcuda.so.1") + dlsym(cuLaunchKernel 等)   ← §2.4 ① 导入表预告、② 运行中现身
      从 .nv_fatbin section 取 SASS → 提交驱动 → kernel 上 GPU    ← §2.1 fatbin
```

<a id="3"></a>
## 3. 编译期：流水线、SASS 与两组开关

### 3.1 流水线：八个工具的一场接力

nvcc 是"编译驱动器"：真正干活的是它调度的一串工具。跟敲：

```bash
$ nvcc --dryrun -O3 -arch=native 0_to_7.cu -o /tmp/x | grep -E 'gcc|cudafe|cicc|ptxas|fatbinary|nvlink|g\+\+'
```

`--dryrun` 只打印不执行；grep 过滤掉环境行后剩下的 **11 条**就是真正会被执行的命令，对应下表 8 个阶段（每行截短）：

| 阶段 | 实际命令（dryrun 实测） | 干什么 |
|---|---|---|
| 1 host 预处理 | `gcc -E -x c++ ...` | 展开 include/宏，host 视角 |
| 2 host/device 分离 | `cudafe++` | 把 `.cu` 拆成 host 部分和 device 部分 |
| 3 device 预处理 | `gcc -E -D__CUDA_ARCH__=1030 ...` | **第二次预处理**，device 视角 |
| 4 device 前端 | `cicc -arch compute_103 → .ptx` | CUDA C++ → PTX |
| 5 汇编 | `ptxas -arch=sm_103 → .cubin` | PTX → SASS 机器码 |
| 6 打包 | `fatbinary --image3=kind=elf,sm=103` | SASS 装进 fatbin，生成 C 数组 |
| 7 host 编译 | `gcc -c -O3 ... → .o` | host 代码 + fatbin 注册桩 |
| 8 链接 | `g++ -O3 -lcudart_static ... -o 0_to_7` | 出可执行文件 |

两个贯穿性机制：

- **同一份源码被预处理两次**：device 那趟带着 `-D__CUDA_ARCH__=1030` 宏（1030 = CC 10.3）——这是 `#if __CUDA_ARCH__ >= 1000` 类架构条件编译的机制基础；
- **`<<<grid,block>>>` 调用点**被替换为 `cudaLaunchKernel` + fatbin 句柄，SASS 作为数据段嵌入可执行文件（§2.1），运行时从中取出提交给驱动。

中间产物（`.ii` / `.ptx` / `.cubin` / `.o`）都在 `/tmp/tmpxft_*`，编完即删。

**编译时机的谱系**（概念定位）：本路线是 **AOT**（Ahead-Of-Time，开发时编译、运行时只执行）。另两个时机：**driver 侧 JIT**——fatbin 带 PTX 时，驱动在新架构卡上首次加载现编译成 SASS（前向兼容的兜底）；**NVRTC**——运行时把 CUDA C++ 源码编成 PTX 嵌入程序（推理框架常用）。独立可执行文件选 AOT 是自然结果：编译/运行彻底分离，产物可离线分发。

### 3.2 SASS：GPU 的真机器码（跟敲）

stage 5（ptxas）的产物，就存在 `.nv_fatbin` section 里（§2.1）。**SASS 是 GPU 的真实机器指令集，每代架构一套**（sm_103 的 SASS 拿到别的卡上不合法），PTX 则是跨架构的虚拟 ISA——§3.3"native 产物无前向兼容"的根源就在这。

**名字考**：PTX 有官方全称（**P**arallel **T**hread e**X**ecution，PTX ISA 指南开篇定义）；**SASS 没有官方全称**——NVIDIA 文档只作名词使用、从不展开（本机 nvdisasm 的帮助与内嵌字符串实测也搜不到任何展开），社区通行展开为 "Streaming ASSembly"（呼应 **S**treaming **M**ultiprocessor 命名家族，属合理推断，非官方背书）。

跟敲（需要 `nvdisasm`，随 cuda-gdb 安装，附录 A 坑 6）：

```bash
$ cuobjdump -sass 0_to_7 | grep -A60 'Function : _Z11fill_kernel' | grep -E '^ +/\*[0-9a-f]{4}\*/'
```

实测的源码 → 机器码对照（节选）：

| 源码（`0_to_7.cu` 的 `fill_kernel`，见"样例程序"一节） | SASS 指令（实测，节选） |
|---|---|
| `threadIdx.x` / `blockIdx.x` | `S2R R2, SR_TID.X` / `S2UR UR4, SR_CTAID.X` |
| `blockIdx.x * blockDim.x + threadIdx.x` | `IMAD.WIDE.U32 R2, R5, UR4, R2` |
| `i < n` 判断（`size_t`，64 位） | `ISETP.GE.U32.AND P0, PT, R2, UR8, PT` + `ISETP.GE.U32.AND.EX P0, PT, R3, UR9, PT, P0` + `@P0 EXIT` |
| `(float)i` | `I2F.U64 R3, R2` |
| `p[i] = ...` | `STG.E desc[UR6][R6.64], R3`（32 位标量存储） |
| grid-stride 回边（`i += stride`） | `IADD3 R2, P0, PT, R2, R4, RZ` + `@!P0 BRA 0x110`（谓词化循环） |

四个要点：

1. **源码写法决定指令宽度**：`fill_kernel` 的元素是标量 `float`，所以存储是 32 位的 `STG.E`；若改用 `float4` 访问，预期会看到 128 位的宽存储（这一点是**推断**，本程序没有 `float4` 访问，未验证）；
2. **64 位运算被拆成多条 32 位指令**：`size_t` 的比较编译成 `ISETP.GE.U32` + `ISETP.GE.U32.AND.EX` 两条，`(float)` 转换用 `I2F.U64`；
3. **`i < n` 没有变成分支**：编译成谓词执行（`ISETP` 置谓词寄存器 + `@P0 EXIT`），越界线程直接退出；循环回边同样是谓词化的 `@!P0 BRA`——warp 内不发散，这是 grid-stride 写法的隐藏收益；
4. **`check_kernel` 的 `atomicAdd` 被编译成 `REDG.E.ADD.64.STRONG.GPU`**（实测，用 `cuobjdump -sass 0_to_7 | grep -E 'ATOM|RED'` 可见）：返回值没被使用，所以用的是不带回传的归约型原子指令；
5. **kernel 名在 SASS 里是 mangled 的**：输出里看到的是 `_Z11fill_kernelPfm`（C++ 名字修饰：`fill_kernel` 长 11 个字符，参数 `float*`、`size_t` 依次编码为 `Pf`、`m`），`grep` 按子串匹配即可。

### 3.3 架构开关：`-arch` 一族

`-arch=native` 的含义：编译时现场探测**编译机**的 GPU。dryrun 显示它实际展开为三个目标：

```
-D__CUDA_ARCH_LIST__=1030     ← 预处理宏
cicc  -arch compute_103       ← 前端目标：虚拟架构（PTX 的假想 GPU）
ptxas -arch=sm_103            ← 后端目标：真实架构（B300 的 SASS）
```

| 写法 | 效果 |
|---|---|
| `-arch=native` | 探测编译机 → 本机展开为 sm_103。CUDA 12.9+ 可用，编译机=运行机时推荐 |
| `-arch=sm_103` | 显式指定 B300，编译机可以没有 GPU（交叉编译） |
| 不写（默认） | **实测默认 sm_75（Turing）**——SASS 在 B300 上跑不了，靠默认内嵌的 compute_75 PTX 让驱动 JIT 兜底：能跑，但首次加载慢 + 用不上 Blackwell 指令集 |
| `-gencode arch=compute_103,code=compute_103` | 多嵌一份 PTX，换新卡时驱动 JIT 兜底（老 PTX 在新卡可跑，反之不行） |

跟敲验证（三组对照）：

```bash
$ nvcc --dryrun -arch=native 0_to_7.cu -o /tmp/x | grep -m1 -oE 'CUDA_ARCH_LIST__=[0-9]+'
CUDA_ARCH_LIST__=1030                                    ← native 的选择
$ nvcc --dryrun 0_to_7.cu -o /tmp/x | grep -m1 -oE 'CUDA_ARCH_LIST__=[0-9]+'
CUDA_ARCH_LIST__=750                                     ← 不写 -arch 的默认：Turing
$ cuobjdump --list-elf 0_to_7
ELF file    1: 0_to_7.1.sm_103.cubin
ELF file    2: 0_to_7.2.sm_103.cubin                     ← 两份 SASS（kernel + device-link）
$ cuobjdump --list-ptx 0_to_7
cuobjdump info : No PTX file found to extract ...        ← 空列表：native 产物只有 SASS
```

再对照两趟 dryrun 的 `fatbinary` 行（最有说服力的一处）：**默认（sm_75）带 `kind=elf` 与 `kind=ptx` 两类 image**——`--image3=kind=elf,sm=75,...cubin` 和 `--image3=kind=ptx,sm=75,...ptx`（SASS + PTX，所以默认产物能在 B300 上靠驱动 JIT 跑起来）；**native 只有 `kind=elf`**——`--image3=kind=elf,sm=103,...cubin`，没有 `kind=ptx`（两种情况都另有一个 device-link 的 elf image，此处略）。这是下面坑 (2) 的源头。

两个实测坑：

1. B300 是 Blackwell **Ultra**，CC **10.3（sm_103）**，不是 B200 的 10.0；
2. `native` 生成的 fatbin **只有 SASS（`kind=elf`），没有 PTX image**——拿到未来架构（如 sm_11x）的卡上直接跑不了，要前向兼容必须 `-gencode` 补 PTX。

### 3.4 优化与调试信息开关：`-O` / `-g` / `-G` / `-lineinfo`

**`-O3` 只作用于 host 代码**（**实测**）：dryrun 里 `-O3` 出现在 host 侧的 4 条 `gcc` 与最终的 `g++` 命令里，而 `cicc` 与 `ptxas` 的命令行里没有任何 `-O`——device 代码默认就是全优化。想关掉 device 优化只有一个开关：`-G`。

三件调试信息开关与三种构建形态：

| 构建 | 编译命令 | 用途 |
|---|---|---|
| release | `nvcc -O3 -arch=native` | 性能测量 |
| debug | `nvcc -g -G -O0 -arch=native` | 调试（见 [debug.md](debug.md)）：`-g` host 调试信息，`-G` device 调试信息 + **关优化**（kernel 逐行单步的前提，代价是显著变慢——**不要用 debug 版跑性能**） |
| profile | `nvcc -O3 -lineinfo -arch=native` | 性能调优：保留优化 + 行号（能定位源码行，不能单步） |

跟敲——把三种构建都编出来，尺寸差异就是三种形态的直接体现：

```bash
$ nvcc -O3 -arch=native 0_to_7.cu -o 0_to_7
$ nvcc -g -G -O0 -arch=native 0_to_7.cu -o 0_to_7.debug
$ nvcc -O3 -lineinfo -arch=native 0_to_7.cu -o 0_to_7.prof
$ ls -lh 0_to_7 0_to_7.debug 0_to_7.prof | awk '{print $9, $5}'
0_to_7        978K   ← release
0_to_7.debug  1008K  ← debug：-G 关优化 + 调试信息，更大更慢
0_to_7.prof   982K   ← profile：release + 行号表，尺寸几乎不变
```

<a id="A"></a>
## 附录 A. 速查：编译命令与坑

```bash
# 0_to_7——三种构建（§3.4）
nvcc -O3 -arch=native              0_to_7.cu -o 0_to_7         # release：性能测量
nvcc -g -G -O0 -arch=native        0_to_7.cu -o 0_to_7.debug   # debug：调试（无 CLI 参数）
nvcc -O3 -lineinfo -arch=native    0_to_7.cu -o 0_to_7.prof    # profile

# 看编译器在干什么
nvcc --dryrun -O3 -arch=native 0_to_7.cu -o /tmp/x   # 展开 8 阶段真实命令（§3.1）
```

| # | 坑/注意 | 说明 |
|---|---|---|
| 1 | `-arch=native` 需要编译机有 GPU | 交叉编译场景用 `-arch=sm_103` / `-gencode` 显式指定（§3.3） |
| 2 | sm_103 ≠ sm_100 | B300 是 Blackwell **Ultra**（CC 10.3）；拷命令行示例时别用 B200 的 10.0（§3.3） |
| 3 | `native` 产物只含 SASS，无 PTX | 拿到未来架构的卡上直接跑不了；要前向兼容加 `-gencode arch=compute_103,code=compute_103`（§3.3） |
| 4 | `-O3` 不影响 device 代码 | device 默认全优化，`-G` 才是关优化的开关（§3.4） |
| 5 | `-G` 构建不能跑性能 | device 优化被关闭，kernel 显著变慢；性能测量只用 release 构建（§3.4） |
| 6 | 裸镜像缺 `nvdisasm` | `cuobjdump -sass` 报 `Could not find executable file 'nvdisasm'`。用 `n-server1.yaml` 部署时启动脚本会装 cuda-gdb，apt 自动补上；用原版 `n-server.yaml` 则需手动安装（见 [debug.md](debug.md) §1） |
| 7 | 调试器不在原版镜像里 | `n-server.yaml` 部署的 Pod 无 gdb / cuda-gdb；`n-server1.yaml` 已把安装固化进启动脚本。注意 Pod 重建后**拷进去的代码和编译产物会丢**（见 [debug.md](debug.md) §2.3） |

---

*环境：server-0（StatefulSet `examples/n-server.yaml` / 调试用 `n-server1.yaml`，8×B300 SXM6, NV18 全互联, CUDA 13.0, nvcc 13.0）；实验与调试验证日期 2026-10-26/27、2026-10-31；代码：`p2p/0_to_7.cu`，含中文注释。*

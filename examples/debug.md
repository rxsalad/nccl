# CUDA 调试：gdb、cuda-gdb 与 VS Code（Pod 内）

目标：在 server-0 Pod（8×B300，sm_103，CUDA 13.0）里对 `.cu` 程序下断点、单步、看变量，包括停进 GPU kernel。本文覆盖：**准备环境 → 拷贝代码 → 编译 debug 版 → 命令行调试 → VS Code 调试 → 排错**。用到三个调试程序：`0_to_7`（16 GB 定点传输）、`remote_access`（方式 A：kernel 直接读写对端显存，数据量小，更适合调试）、`remote_access_noenable`（故意不 `enable`，用来观察非法访问）。

约定：**实测** = 2026-10-31 在 `n-server1.yaml` 部署的 server-0 上跑过；**未验证** = 未实际跑过；**使用者确认** = 由使用者在自己的环境里操作并反馈。所有命令在 nuc 上执行（`ssh nuc`），`kubectl` 默认在 nuc 上可用。

**TOC**

- [1. 准备环境](#1)
- [2. 向 Pod 拷贝代码](#2)
- [3. 编译 debug 版](#3)
- [4. 命令行调试](#4)
- [5. VS Code 调试](#5)
- [6. 排错与注意事项](#6)

<a id="1"></a>
## 1. 准备环境

### 1.1 部署 Pod

使用 `n-server1.yaml`（基于 `n-server.yaml`，已把调试环境固化进启动脚本）：

```bash
cd /home/ubuntu/work/nccl/examples
kubectl apply -f n-server1.yaml
kubectl logs -f server-0        # 看到末尾 "+ exec sleep infinity" 即安装完成
```

`n-server1.yaml` 与 `n-server.yaml` 的差异：

| 改动 | 作用 |
|---|---|
| `apt install ... gdb` | host 侧调试器 |
| 下载并 `dpkg -i cuda-gdb-13-0_13.0.85-1_amd64.deb`，再 `apt-get install -f` | 设备侧调试器 `cuda-gdb`；`-f`（fix-broken）自动补装依赖 `cuda-nvdisasm-13-0` |
| `capabilities.add: SYS_PTRACE` | 允许 attach 到**非子进程**（`gdb -p PID`）。仅"gdb 启动程序"的方式不需要它 |
| `replicas: 1` | 单机调试，只占 8 张 GPU |

注意：

- **Pod 没有 readiness probe**，一启动就显示 Ready，但启动脚本还要装一两分钟包。以日志末尾的 `exec sleep infinity` 为准。
- StatefulSet 名字仍是 `server`，**不能与 `n-server.yaml` 同时部署**（先 `kubectl delete -f` 另一个）。
- 安装的软件在容器文件系统层，Pod 重建会丢，但启动脚本会自动重装；**拷进去的代码和编译产物会丢**（见 2.3）。

### 1.2 验证环境（实测）

```bash
kubectl exec server-0 -- bash -lc 'which gdb cuda-gdb nvdisasm nvcc; gdb --version | head -1; nvidia-smi -L | wc -l; cat /proc/sys/kernel/yama/ptrace_scope'
```

期望：`/usr/bin/gdb`、`/usr/local/cuda/bin/cuda-gdb`、`/usr/local/cuda/bin/nvdisasm`、nvcc 都有；gdb 12.1；GPU 数 8；`ptrace_scope` 为 0。

### 1.3 软件清单

| 软件 | 版本（实测） | 来源 | 用途 |
|---|---|---|---|
| `gdb` | 12.1-0ubuntu1~22.04.2 | apt | host 侧调试 |
| `cuda-gdb-13-0` | 13.0.85-1 | NVIDIA repo deb | 设备侧调试（kernel 断点、`info cuda ...`） |
| `cuda-nvdisasm-13-0` | 13.0.85-1 | cuda-gdb 的依赖 | SASS 反汇编，`cuobjdump -sass` 需要 |
| `nvcc` | 13.0 | 镜像自带 | 编译 |

镜像 `vllm/vllm-openai:cu130-nightly` **不含任何调试器**，也不含 compute-sanitizer（需要的话另装，**未验证**包名）。

<a id="2"></a>
## 2. 向 Pod 拷贝代码

源码在 nuc 的 `/home/ubuntu/work/nccl/examples/p2p/`：

| 文件 | 说明 |
|---|---|
| `0_to_7.cu` | 16 GB 定点传输实验，无命令行参数 |
| `remote_access.cu` | 方式 A 示例：GPU0 的 kernel 直接读写 GPU7 的显存（64 MiB），无命令行参数 |
| `.vscode/launch.json` | VS Code 调试配置，3 个（见 §5.3） |

### 2.1 用 kubectl cp（实测）

`kubectl cp` 在 Pod 内需要 `tar`（镜像里有）。**注意目的路径要与 launch.json 的 `${workspaceFolder}` 对应**：launch.json 里 `program` 形如 `${workspaceFolder}/0_to_7.debug`，所以代码平铺放在 `/vllm-workspace/`：

```bash
cd /home/ubuntu/work/nccl/examples/p2p
kubectl cp 0_to_7.cu        server-0:/vllm-workspace/0_to_7.cu
kubectl cp remote_access.cu server-0:/vllm-workspace/remote_access.cu
kubectl cp .vscode          server-0:/vllm-workspace/.vscode
```

验证：

```bash
kubectl exec server-0 -- ls -a /vllm-workspace /vllm-workspace/.vscode
```

**注意**：`kubectl cp .vscode` 会**覆盖** Pod 里的 `launch.json`。如果你在 Pod 里（比如通过 VS Code）改过它，先按 §2.2 拷回 nuc，再做 2.1，否则改动会丢。

### 2.2 反向：把 Pod 里的修改拷回 nuc

在 Pod 里直接改了代码后，别忘了拷回，否则 Pod 重建就丢：

```bash
kubectl cp server-0:/vllm-workspace/0_to_7.cu /home/ubuntu/work/nccl/examples/p2p/0_to_7.cu
kubectl cp server-0:/vllm-workspace/.vscode/launch.json /home/ubuntu/work/nccl/examples/p2p/.vscode/launch.json   # 在 Pod/VS Code 里改过配置时
```

### 2.3 持久化（可选，未做）

`/vllm-workspace` 在容器层，Pod 重建即丢。要持久化，可在 yaml 里加一个 `hostPath` volume 挂到工作目录，或每次重建后重新执行 2.1。

### 2.4 备选：在 VS Code 里直接编辑

连上 Pod 后（§5.1），在 VS Code 里直接编辑 `/vllm-workspace/*.cu` 即可，无需 `kubectl cp`；但仍要记得 2.2 把结果拷回 nuc。

<a id="3"></a>
## 3. 编译 debug 版（实测）

在 Pod 里编译（用 `kubectl exec -it server-0 -- bash -l` 进去，或下面的一行式）：

```bash
kubectl exec server-0 -- bash -lc 'cd /vllm-workspace \
 && nvcc -g -G -O0 -arch=native            0_to_7.cu       -o 0_to_7.debug \
 && nvcc -g -G -O0 -arch=native            remote_access.cu -o remote_access.debug \
 && nvcc -g -G -O0 -arch=native -DNO_ENABLE remote_access.cu -o remote_access_noenable.debug'
```

**三个调试程序**：

| 程序 | 源码 | 说明 |
|---|---|---|
| `0_to_7.debug` | `0_to_7.cu` | 16 GB 定点传输（方式 B：`cudaMemcpyPeer`）；两块 GPU 各占 16 GiB 显存 |
| `remote_access.debug` | `remote_access.cu` | 方式 A：GPU0 的 kernel 直接读写 GPU7 的显存；数据量 64 MiB，轻 |
| `remote_access_noenable.debug` | `remote_access.cu` + `-DNO_ENABLE` | **同一份源码**，只是编译时去掉了 `cudaDeviceEnablePeerAccess` 这一段，其余完全相同（两个二进制大小只差 80 字节）；用来观察"忘了 enable"的后果 |


三个 flag 的职责：

| flag | 作用 |
|---|---|
| `-g` | host 源码行 + 变量 |
| `-G` | **device** 源码行 + 变量，**同时关闭 device 优化**（kernel 会明显变慢，**不要用 debug 版测性能**） |
| `-O0` | host 侧关优化，避免行号跳跃 |
| `-arch=native` | 编译为当前 GPU 的架构（B300 是 sm_103）；与调试无关，只是编对架构 |

调试目标与所需 flag：

| 目标 | 需要 |
|---|---|
| 停进 kernel、按源码单步、看变量 | `-g -G` |
| 只调 host 逻辑 | 只要 `-g` |
| 只确认 kernel 执行了没有 | release 版也能断（"盲断点"，没有源码行和变量） |

**注意**：`0_to_7.debug` 固定传 16 GB（2^32 个元素），没有参数可以缩小数据量；`fill_kernel` 启动 4096 个 block × 256 线程，而且 `-G` 关闭了 device 优化，断点命中后继续执行会明显变慢，属正常。

<a id="4"></a>
## 4. 命令行调试

### 4.1 启动（实测）

```bash
kubectl exec -it server-0 -- bash -l
cd /vllm-workspace
cuda-gdb --args ./0_to_7.debug
```

批处理一行式（适合快速验证）：

```bash
kubectl exec server-0 -- bash -lc 'cd /vllm-workspace && cuda-gdb -batch \
  -ex "break fill_kernel" -ex run -ex next -ex "print i" --args ./0_to_7.debug'
```

实测输出（节选）：

```
Breakpoint 1 at 0x9fb2: file /vllm-workspace/0_to_7.cu, line 44.
CUDA thread hit Breakpoint 1.2, fill_kernel<<<(4096,1,1),(256,1,1)>>> (p=0x7ff960000000, n=4294967296) at 0_to_7.cu:45
45	  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程 id
46	  size_t stride = (size_t)gridDim.x * blockDim.x;            // grid 总步长
$1 = 0
```

**断点停在该行执行之前**：停下来立刻 `print i` 会得到 `<optimized out>` 或未赋值的值，先 `next` 一步再看。

### 4.2 常用命令

| 命令 | 作用 |
|---|---|
| `break fill_kernel` / `break 0_to_7.cu:45` | 函数断点 / 行断点（kernel 里同样有效） |
| `run` / `continue` / `next` / `step` | 运行 / 继续 / 单步 |
| `print threadIdx` / `print blockIdx` | 查看当前 GPU 线程坐标 |
| `info cuda kernels` | 活跃 kernel 与 grid/block 配置 |
| `cuda block (3,0,0)` | 焦点切到 block 3 |
| `cuda thread (255,0,0)` | 焦点切到 thread 255 |
| `set cuda break_on_launch application` | 每个 kernel 启动即停，不用先设断点 |

### 4.3 attach 已运行进程（**未验证**）

`n-server1.yaml` 已加 `SYS_PTRACE`，理论上可以：

```bash
cuda-gdb -p <PID>
```

当前 Pod 的 `ptrace_scope=0`，同 UID 进程本来就能 attach；`SYS_PTRACE` 是兜底。`SYS_PTRACE` 能读同一 PID 命名空间里其他进程的内存，只在调试用的 Pod 上保留，**不要带到生产 yaml**。

### 4.4 调试 remote_access：远程读写与 NO_ENABLE（实测）

`remote_access.cu` 里值得下断点的位置：

| 行 | 内容 |
|---|---|
| 57 | `remote_read` 里的 `local[i] = remote[i] + 1`（远程读） |
| 68 | `remote_write` 里的 `remote[i] = ...`（远程写） |
| 143 | `cudaDeviceEnablePeerAccess(7, 0)` |
| 158 / 176 | host 侧启动 `remote_read` / `remote_write` |

```bash
cuda-gdb -batch -ex "break remote_read" -ex run -ex "info cuda kernels" -ex "print n" -ex "print remote" --args ./remote_access.debug
```

```
CUDA thread hit Breakpoint 1.3, remote_read<<<(4096,1,1),(256,1,1)>>>
   (remote=0x7ffd9c000000, local=0x7ffd6a000000, n=16777216) at remote_access.cu:54
$1 = 16777216
```

`remote`（GPU7 的显存）和 `local`（GPU0 的）在地址上**看起来都是普通 UVA 指针**，从数值上分不出哪个在对端。

**NO_ENABLE：看非法访问是怎么被抓到的**——对 `remote_access_noenable.debug` 直接 `run`，不需要设断点：

```
CUDA Exception: Warp Illegal Address
The exception was triggered at PC ...  remote_read(...)  (remote_access.cu:57)
[Switching focus to CUDA kernel 0, grid 1, block (6,0,0), thread (192,0,0), device 0, ...]
```

要点：

1. 它精确定位到 `remote[i]` 那一行；
2. 停下来的 block / thread 是**第一个被捕获到出错的那个**，每次运行可能不同，不要当成固定位置；
3. 不用调试器直接运行时，**启动本身不报错**（`cudaGetLastError()` 通过），错误要到后面的 `cudaDeviceSynchronize()` 才暴露为 `an illegal memory access was encountered`——所以"启动后查错"之外，同步处也要查。

<a id="5"></a>
## 5. VS Code 调试

### 5.1 连接到 Pod

`n-server1.yaml` 启动脚本已装好 VS Code CLI（`/usr/local/bin/code`），可以用 **code tunnel** 连接。**使用者确认**：已通过隧道连上 Pod 并完成调试；我们在 Pod 里实测到 `code tunnel --accept-server-license-terms --name 001` 与 VS Code server 在运行。GitHub 授权由使用者在浏览器完成，本文档作者未亲自执行：

```bash
kubectl exec -it server-0 -- bash -l
code tunnel user login --provider github          # 按提示在浏览器完成 GitHub 授权
nohup code tunnel --accept-server-license-terms --name 001 &> /tmp/tunnel.log &
```

然后在本地 VS Code 里：命令面板 → **Remote-Tunnels: Connect to Tunnel…** → 选 `001` → 打开文件夹 `/vllm-workspace`。

### 5.2 需要安装的插件

在已连接 Pod 的窗口里装，按钮会显示 **"Install in Tunnel/SSH/Remote"**，点它（装到 Pod 侧）。装完在扩展面板**下半区**（远端分区）确认出现。

| 扩展 | 必要性 | 作用 |
|---|---|---|
| `ms-vscode.cpptools`（C/C++，Microsoft） | **必须** | 提供 `cppdbg` 调试引擎，launch.json 的 `"type": "cppdbg"` 由它实现，它通过 MI 协议驱动 cuda-gdb。不装则 F5 直接报错 |
| `nvidia.nsight-vscode-edition`（Nsight VS Code Edition，NVIDIA） | 推荐 | `.cu` 语法高亮（不装则是纯文本）+ kernel 停下后的 block/thread 图形化切换 + CUDA 内存视图。不装也能调试，焦点切换要手敲 `-exec` |

**实测**：Pod 侧 `/root/.vscode-server/extensions/` 下已装 `ms-vscode.cpptools-1.34.4` 与 `nvidia.nsight-vscode-edition-2025.1.36067579`。（本文早先版本把 Nsight 扩展写成 `theycallsway.nsight-vsce`，那个 ID 是错的，已更正。）

**不需要装**：gdb / cuda-gdb（已在 Pod 里）、CMake Tools（命令行 nvcc 构建，无 CMake 工程）、CodeLLDB（我们用 gdb 系）。

### 5.3 launch.json

位于 `/vllm-workspace/.vscode/launch.json`（§2.1 已拷入），**三个配置**，F5 启动**下拉框当前选中的那个**（见 §5.6）：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "remote_access (cuda-gdb)",
      "type": "cppdbg",
      "request": "launch",
      "program": "${workspaceFolder}/remote_access.debug",
      "args": [],
      "cwd": "${workspaceFolder}",
      "MIMode": "gdb",
      "miDebuggerPath": "/usr/local/cuda/bin/cuda-gdb"
    },
    {
      "name": "0_to_7 (cuda-gdb)",
      "type": "cppdbg",
      "request": "launch",
      "program": "${workspaceFolder}/0_to_7.debug",
      "args": [],
      "cwd": "${workspaceFolder}",
      "MIMode": "gdb",
      "miDebuggerPath": "/usr/local/cuda/bin/cuda-gdb"
    },
    {
      "name": "remote_access NO_ENABLE (cuda-gdb)",
      "type": "cppdbg",
      "request": "launch",
      "program": "${workspaceFolder}/remote_access_noenable.debug",
      "args": [],
      "cwd": "${workspaceFolder}",
      "MIMode": "gdb",
      "miDebuggerPath": "/usr/local/cuda/bin/cuda-gdb"
    }
  ]
}
```

关键一行是 **`miDebuggerPath` 指向 cuda-gdb**：对 cpptools 来说它就是一个 gdb，断点、单步、变量查看开箱即用，`.cu` 里 kernel 函数的断点直接生效。

**逐项核对结果（实测）**：JSON 合法、名称无重复；三个 `program` 都存在且可执行；`miDebuggerPath` 存在；三个 `.debug` 都含 `.debug_info`（`-g` 生效）；每个配置只有 8 个标准键，没有多余或拼错的字段。配置在文件里的**顺序不影响运行结果**（见 §5.6）。没有 `preLaunchTask`：改了源码后要手动重编（§3）。

### 5.4 验证清单（四项全过再 F5）

1. 扩展面板远端分区能看到 **C/C++**；
2. 打开 `/vllm-workspace/0_to_7.cu` 有语法高亮（Nsight 扩展生效）；
3. 断点是**红色实心圆**——例如 `0_to_7.cu` 第 45 行（`fill_kernel` 函数体首行）或 `remote_access.cu` 第 57 行；灰色空心圆 = 扩展未生效或工作区没打开对；
4. **"运行和调试"面板顶部下拉框里显示的配置名，就是你要调试的那个**（见 §5.6）。

F5 前必须已完成 §3（`*.debug` 要存在）。

### 5.5 在调试控制台里控制 GPU 焦点

用 `-exec` 前缀：

| 命令 | 作用 |
|---|---|
| `-exec info cuda kernels` | 活跃 kernel 与 grid/block |
| `-exec cuda block (3,0,0)` | 切 block |
| `-exec cuda thread (255,0,0)` | 切 thread |
| `-exec set cuda break_on_launch application` | 每个 kernel 启动即停 |

### 5.6 选择配置，并确认实际启动的是哪个程序（经验）

**现象**：不管怎么按 F5，总是进入 `0_to_7.cu`。

**原因**：VS Code 会**记住你上一次选中的配置**（按名字记，存在**本地** VS Code 的工作区状态里），而不是每次选列表里第一个。所以即使把别的配置在 `launch.json` 里挪到第一位，已选定的配置不会变。

**做法**：

1. `Ctrl+Shift+D` 打开"运行和调试"面板；
2. 点顶部绿色三角**右边的下拉框**，明确选择想调试的配置；
3. **确认下拉框里显示的就是它**，再按 F5。选定后会被记住。

**怎么确认实际启动了哪个程序**（实测）——在 Pod 里看进程，不用靠猜：

```bash
kubectl exec server-0 -- bash -lc 'ps -eo pid,ppid,etimes,args | grep -E "cuda-gdb|\.debug" | grep -v grep | cut -c1-200'
```

VS Code 的 C/C++ 扩展通过 `cuda-gdb --interpreter=mi` 驱动调试，其子进程就是被调试的 `.debug` 程序，例如：

```
/usr/local/cuda/bin/cuda-gdb-python3.12-tui --interpreter=mi --tty=/dev/pts/1
    └─ /vllm-workspace/0_to_7.debug
```

会话很短的话要在 F5 之后立刻看。这次排查正是靠它发现启动的是 `0_to_7.debug`。

### 5.7 Pod 里的 tasks.json 是什么

`/vllm-workspace/.vscode/tasks.json` 是 VS Code 的 C/C++ 扩展**自动生成**的（文件里写着 `"detail": "Task generated by Debugger."`），**不是我们创建的**。内容是一个构建任务：

```json
{ "label": "C/C++: gcc build active file", "command": "/usr/bin/gcc",
  "args": ["-fdiagnostics-color=always", "-g", "${file}", "-o", "${fileDirname}/${fileBasenameNoExtension}"],
  "group": { "kind": "build", "isDefault": true } }
```

即"用 `gcc -g` 把当前文件编译成同名可执行文件"。**对 CUDA 程序不适用**：`gcc` 不认识 `__global__` 与 `<<<>>>`，会编译失败。

| 触发方式 | 是否会用到它 |
|---|---|
| F5（本文的 `launch.json` **没有** `preLaunchTask`） | 不会 |
| 编辑器右上角 ▷"运行/调试 C/C++ 文件" | 会（它的来源） |
| `Ctrl+Shift+B`（默认构建任务） | 会 |

所以**不要用 ▷ 按钮和 `Ctrl+Shift+B`**，调试一律用"运行和调试"面板里选定配置后按 F5。可选改进（**未做**）：把它换成 `nvcc -g -G -O0` 的任务，并在 `launch.json` 里加 `preLaunchTask`，让 F5 前自动重编。

<a id="6"></a>
## 6. 排错与注意事项

| 现象 | 原因 / 处理 |
|---|---|
| F5 报错找不到调试类型 | 没装 `ms-vscode.cpptools`，或装在了本地而不是 Pod 侧 |
| 断点是灰色空心圆 | 扩展未在远端生效，或打开的工作区不是 `/vllm-workspace` |
| F5 提示 program 不存在 | 没执行 §3 编译，或代码没放在 `/vllm-workspace` 平铺 |
| 停在 kernel 里但没有源码行、`print i` 报 `No symbol` | 用的是 release 版，没带 `-g -G` |
| `print i` 得到 `<optimized out>`/垃圾值 | 断点在行执行前，先 `next` 一步 |
| `cuobjdump -sass` 报 `Could not find executable file 'nvdisasm'` | 没装 cuda-gdb 的依赖 `cuda-nvdisasm-13-0`；`n-server1.yaml` 已自动装 |
| `gdb -p` attach 被拒 | 确认 `SYS_PTRACE` 已生效：`grep CapEff /proc/1/status`（第 19 位）；`ptrace_scope` 是否为 0 |
| Pod 重建后工具/代码都没了 | 工具会由启动脚本重装；**代码要重新 `kubectl cp`**（§2.1），修改过的要先拷回（§2.2） |
| debug 版跑得特别慢 | `-G` 关了 device 优化，正常；性能测试用 release 版（`nvcc -O3 -arch=native`） |
| 每次 F5 都进入 `0_to_7.cu` | 下拉框记住了上次选中的配置，与 `launch.json` 顺序无关；明确选择配置，并用进程确认（§5.6） |
| 点 ▷ 或 `Ctrl+Shift+B` 报 gcc 错误 | `tasks.json` 是自动生成的 `gcc` 任务，不适用 CUDA；改用"运行和调试"面板（§5.7） |
| 改了源码后断点位置对不上 | `.debug` 没有重编；重新执行 §3 的 `nvcc -g -G -O0 ...`（`launch.json` 没有 `preLaunchTask`） |
| 不 `enable` 时程序报 `an illegal memory access was encountered` | 启动不报错，错误在 `cudaDeviceSynchronize()` 才暴露；在 `cuda-gdb` 下会直接停在 `CUDA Exception: Warp Illegal Address`（§4.4） |
| 想同时调试时 GPU 显存被占满 | `0_to_7.debug` 占两块 16 GiB；`remote_access` 只占 64 MiB 量级，优先用它 |

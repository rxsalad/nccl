// ============================================================================
// 0_to_7.cu —— GPU0 通过 NVLink 向 GPU7 传递 16 GB 数据
//
// 四步流程（与输出一一对应）：
//   ① 检查：GPU0↔GPU7 的 P2P 可达性（NVLink 直连的证据，can 查询）
//   ② 准备：GPU0 上分配 16 GB 源缓冲，填入已知模式 p[i] = (float)i
//   ③ 传输：cudaMemcpyPeer GPU0→GPU7（可 P2P 则自动走 NVLink），事件计时
//   ④ 校验：在 GPU7 上用 kernel 数出"不等于 (float)i"的元素个数——
//      校验在 GPU 上做（不用把 16 GB 拉回 host），两边独立算同一模式，
//      拷坏任何字节都会被数出来
//
// 想确认流量真的走了 NVLink（而非绕行 host）：
//   nvidia-smi nvlink -gt d   前后各取一次 GPU0/GPU7 的计数差即可
//
// 编译：  nvcc -O3 -arch=native 0_to_7.cu -o 0_to_7
// 运行：  ./0_to_7
// ============================================================================

#include <cuda_runtime.h>
#include <cstdio>

// 16 GB = 2^34 字节 = 2^32 个 float（4 字节）
#define BYTES ((size_t)16 << 30)
#define N     (BYTES / sizeof(float))

// ----------------------------------------------------------------------------
// 错误检查宏：裸 C 没有 try/catch，每个 CUDA 调用的返回值都必须看。
// ----------------------------------------------------------------------------
#define CHECK_CUDA(call)                                                    \
  do {                                                                      \
    cudaError_t e = (call);                                                 \
    if (e != cudaSuccess) {                                                 \
      fprintf(stderr, "FATAL %s:%d  %s: %s\n", __FILE__, __LINE__, #call,  \
              cudaGetErrorString(e));                                       \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

// ----------------------------------------------------------------------------
// 填充 kernel：p[i] = (float)i，每个元素的值 = 自己的下标
// 这是校验链的起点——内容已知，后面才有"拷坏必现"的检出力。
// ----------------------------------------------------------------------------

__global__ void fill_kernel(float* __restrict__ p, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程 id
  size_t stride = (size_t)gridDim.x * blockDim.x;            // grid 总步长
  for (; i < n; i += stride) p[i] = (float)i;                 // 填下标模式
}

// ----------------------------------------------------------------------------
// 校验 kernel：数出"值 ≠ (float)i"的元素个数（在 GPU7 本地读，不占 NVLink）
// 注意两边用的是同一套取整规则（(float)i 对大数有精度损失，但两边一致），
// 所以只要拷贝没坏，bad 必然 = 0。
//
// 局限（已用 float32 实际计算验证）：(float)i 在 i >= 2^24 之后，相邻下标会共用
//   同一个 float 值——从 2^28 起连续 1024 个下标只有 33 个不同值，从 2^31 起只有
//   5 个（约每 256 个下标共用一个值）。数组共 2^32 个元素，后一半都是这种情况。
//   因此："值被改坏"一定会被数出来；但"数据错位 / 某一小块放错了位置"，只要错位后
//   的值落在同一片相同值里，就会漏检。
//   若想让错位也能检出：把模式改成 32 位无符号整数 p[i] = (unsigned)i
//   （2^32 个下标恰好都能被 32 位无符号数精确表示，每个值唯一），校验代价不变。
// ----------------------------------------------------------------------------

__global__ void check_kernel(const float* __restrict__ p, size_t n,
                            unsigned long long* bad) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  unsigned long long local = 0;               // 本线程数到的不匹配数
  for (; i < n; i += stride)
    if (p[i] != (float)i) local++;
  if (local) atomicAdd(bad, local);            // 各线程计数原子累加到全局
}

// 启动填充 kernel：block 封顶 4096，余量交给 grid-stride
static void launch_fill(float* p, size_t n) {
  int block = 256;
  size_t grid = (n + block - 1) / block;
  if (grid > 4096) grid = 4096;
  fill_kernel<<<(int)grid, block>>>(p, n);
  CHECK_CUDA(cudaGetLastError());  // <<<>>> 不返回错误码，启动错误只能这样查
}

// 启动校验 kernel：同款封顶策略
static void launch_check(const float* p, size_t n, unsigned long long* bad) {
  int block = 256;
  size_t grid = (n + block - 1) / block;
  if (grid > 4096) grid = 4096;
  check_kernel<<<(int)grid, block>>>(p, n, bad);
  CHECK_CUDA(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// peer access 开关：授予"当前设备（先 cudaSetDevice 切过去）"访问 peer 的权限。
// cudaMemcpyPeer 其实不依赖它（不能 P2P 会自动走 host staging）；这里显式
// 打开是为了语义完整——若把传输改成 kernel 远程解引用，enable 就是硬前提。
// "已开启/未开启"的返回码视为成功（幂等）。
// ----------------------------------------------------------------------------

static void enable_peer(int dev, int peer) {
  CHECK_CUDA(cudaSetDevice(dev));
  cudaError_t e = cudaDeviceEnablePeerAccess(peer, 0);  // flags 必须为 0
  if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
    fprintf(stderr, "FATAL enable_peer(%d,%d): %s\n", dev, peer,
            cudaGetErrorString(e));
    exit(1);
  }
}

// ----------------------------------------------------------------------------
// main：全程只有两张卡参与——GPU0 是源端，GPU7 是目的端。
//
// 先记住 CUDA 的一条核心规则——"当前设备"（cudaSetDevice 设定的那张卡）：
//   cudaMalloc / cudaMemset / kernel 启动 / cudaEventCreate / cudaDeviceSynchronize
//   都作用在【调用时的当前设备】上。所以下面会反复出现 cudaSetDevice(0) 与
//   cudaSetDevice(7)：每次切换，都是在告诉后面的调用"现在对哪张卡操作"。
//
// 时间线（谁在哪张卡上干什么）：
//   ① 检查   查询 0↔7 可达性；enable_peer 内部会切换当前设备
//   ② 准备   GPU0：分配 src 并填充；GPU7：分配 dst、清零、分配计数器 bad
//   ③ 传输   当前设备 = 7；cudaMemcpyPeer（dst@GPU7 ← src@GPU0），事件计时
//   ④ 校验   当前设备仍是 7；check_kernel 在 GPU7 本地读 dst
//
// 显存占用：src 16 GiB（GPU0）+ dst 16 GiB（GPU7）+ 8 字节计数器（GPU7）。
// ----------------------------------------------------------------------------
int main() {
  printf("GPU0 --NVLink--> GPU7, %.1f GB per message\n\n", BYTES / 1e9);

  // ---- ① 检查：P2P 可达性（NVLink 直连的证据）--------------------------
  // can 查询只回答"拓扑上允许直连吗"，不改任何状态；它按方向问，所以 0→7 与
  // 7→0 各查一次（NVSwitch 全互联下两个方向恒相等，但代码不假设这一点）。
  int ndev = 0;
  CHECK_CUDA(cudaGetDeviceCount(&ndev));          // 查卡数
  if (ndev < 8) {                                 // 本实验需要 0 号和 7 号卡
    fprintf(stderr, "need 8 GPUs, found %d\n", ndev);
    return 1;
  }

  int can07 = 0, can70 = 0;
  CHECK_CUDA(cudaDeviceCanAccessPeer(&can07, 0, 7));  // GPU0 的 kernel 能读 GPU7？
  CHECK_CUDA(cudaDeviceCanAccessPeer(&can70, 7, 0));  // GPU7 的 kernel 能读 GPU0？
  printf("[1] P2P: canAccessPeer(0->7)=%d, canAccessPeer(7->0)=%d  ",
         can07, can70);
  printf(can07 && can70 ? "(NVLink 直连)\n" : "(不可直连, 将走 host staging)\n");

  // enable 才是真正"授权"：在对应方向的页表里登记对端显存的映射。
  // 对 cudaMemcpyPeer 来说它不是必须的（见 enable_peer 上方的说明），这里照常
  // 打开，一是语义完整，二是让快路径（直连）确定可用，三是便于以后改成 kernel
  // 远程解引用时直接复用。两个方向都开，且重复调用是安全的（幂等）。
  enable_peer(0, 7);  // 打开 GPU0 → GPU7 方向（幂等）
  enable_peer(7, 0);  // 打开 GPU7 → GPU0 方向

  // ---- ② 准备：GPU0 上分配并填充源缓冲 ----------------------------------
  // 注意：launch_fill 里的 fill_kernel 启动是【异步】的（<<<>>> 提交后立刻返回）。
  // 本程序在拷贝前没有对 GPU0 做显式 cudaDeviceSynchronize，依赖的是拷贝与默认流
  // 之间的隐式顺序。本机实测 PASS，说明顺序在这里成立；但这个隐式依赖没有单独
  // 验证过——想要写得严格，可在 launch_fill 之后加 cudaDeviceSynchronize()。
  CHECK_CUDA(cudaSetDevice(0));                   // 后续分配作用在 GPU0
  float* src;
  CHECK_CUDA(cudaMalloc(&src, BYTES));             // 16 GB（内容未定义）
  launch_fill(src, N);                             // 填 (float)i 模式

  CHECK_CUDA(cudaSetDevice(7));                    // 切到接收端
  float* dst;
  CHECK_CUDA(cudaMalloc(&dst, BYTES));             // GPU7 上的接收缓冲
  CHECK_CUDA(cudaMemset(dst, 0, BYTES));           // 先清零：没拷到的位置必然 ≠ 模式
                                                   // （下标 0 例外：(float)0 本来就是 0）

  unsigned long long* bad;
  CHECK_CUDA(cudaMalloc(&bad, sizeof(*bad)));      // 校验计数器（GPU7 本地）
  CHECK_CUDA(cudaMemset(bad, 0, sizeof(*bad)));    // 清零计数器

  // ---- ③ 传输：cudaMemcpyPeer + 事件计时（三段式）-----------------------
  // 为什么用 CUDA 事件，而不用 CPU 时钟（chrono）？
  //   cudaMemcpyPeer 对 host 是【异步】的：实测对 4 GiB 的拷贝，调用本身约 12 µs
  //   就返回，传输约 5.5 ms 后才完成。用 chrono 包住它只会量到"提交命令"的时间。
  //   事件是 GPU 侧的时间戳，排在拷贝命令的前后，量到的才是真实传输时间。
  //
  // 三段式 = ① warmup（不计成绩）→ ② 起点事件 + 被测循环 + 终点事件 → ③ 同步取时间。
  CHECK_CUDA(cudaSetDevice(7));                    // 事件建在目标设备上
  // warmup：第一次拷贝带有一次性开销（建立连接/映射等，推断），不能计入成绩。
  CHECK_CUDA(cudaMemcpyPeer(dst, 7, src, 0, BYTES));  // warmup：触发建路，不计成绩
  CHECK_CUDA(cudaDeviceSynchronize());             // 等 warmup 完成再开始计时

  const int iters = 3;                             // 被测轮数
  cudaEvent_t t0, t1;
  CHECK_CUDA(cudaEventCreate(&t0));                // GPU 侧时间戳一对
  CHECK_CUDA(cudaEventCreate(&t1));
  CHECK_CUDA(cudaEventRecord(t0));                 // 起点入队（此刻只是"排队"，不是"已发生"）
  for (int i = 0; i < iters; i++)
    CHECK_CUDA(cudaMemcpyPeer(dst, 7, src, 0, BYTES));  // 被测传输：连续 3 次，同一对缓冲
  CHECK_CUDA(cudaEventRecord(t1));                 // 终点入队
  // 这一句同时承担两个职责：
  //   (a) 让 host 等到 t1 真正到达，才能读取 ElapsedTime；
  //   (b) 保证上面 3 次拷贝都已完成——后面的 check_kernel 读 dst 时数据才是完整的。
  CHECK_CUDA(cudaEventSynchronize(t1));             // 阻塞至终点真正到达
  float ms;
  CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));   // 间隔（毫秒），覆盖 3 次传输
  CHECK_CUDA(cudaEventDestroy(t0));
  CHECK_CUDA(cudaEventDestroy(t1));

  // 带宽 = 总字节数 / 总时间。ms*1e6 把"毫秒"换成"纳秒"，字节/纳秒 = GB/s（十进制）。
  // 例：3 × 17.2 GB / 66 ms ≈ 779 GB/s，与 B300 单向 NVLink 理论值（~900 GB/s）同量级。
  double gb_s = BYTES * (double)iters / (ms * 1e6);  // 带宽 GB/s（十进制）
  printf("[2] transfer: %d x %.1f GB in %.2f ms -> %.1f GB/s\n",
         iters, BYTES / 1e9, ms, gb_s);

  // ---- ④ 校验：在 GPU7 上数"不等于模式"的元素 ----------------------------
  // 当前设备仍是 GPU7，所以 check_kernel 在 GPU7 上运行、读的是 GPU7 本地的 dst
  // ——这一步不占 NVLink。它排在拷贝之后是安全的，因为上面 cudaEventSynchronize
  // 已经等所有拷贝完成。校验只证明"数据对不对"；"走的是不是 NVLink"它证明不了
  // （经主机中转的降级路径同样能通过），要看 nvidia-smi nvlink -gt d 的计数器。
  launch_check(dst, N, bad);                        // GPU 本地读，不占 NVLink
  CHECK_CUDA(cudaDeviceSynchronize());             // 等校验 kernel 完成
  unsigned long long h_bad = 0;
  CHECK_CUDA(cudaMemcpy(&h_bad, bad, sizeof(h_bad), cudaMemcpyDeviceToHost));  // 只回传 8 字节
  printf("[3] verify: %llu mismatched elements -> %s\n", h_bad,
         h_bad == 0 ? "PASS" : "FAIL");

  // ---- 清理：释放全部资源 -----------------------------------------------
  // 与分配对应：计数器、dst 在 GPU7；src 在 GPU0，所以先切回 GPU0 再释放。
  CHECK_CUDA(cudaFree(bad));                       // 归还校验计数器
  CHECK_CUDA(cudaFree(dst));                       // 归还 GPU7 接收缓冲
  CHECK_CUDA(cudaSetDevice(0));                    // 切回 GPU0 释放源缓冲
  CHECK_CUDA(cudaFree(src));
  printf("\ndone.\n");
  return 0;
}

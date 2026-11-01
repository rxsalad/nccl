// ============================================================================
// remote_access.cu —— 方式 A：kernel 里直接读写对端 GPU 的显存
//
// 对照 c-single.md §1.4：不调用 cudaMemcpyPeer，而是让 GPU0 上的 kernel 直接
// 拿 GPU7 显存的指针做 load / store。
//
// 演示两件事：
//   [A1] 远程读：GPU0 的 kernel 读 GPU7 的 buffer，结果写进 GPU0 本地
//   [A2] 远程写：GPU0 的 kernel 把数据直接写进 GPU7 的 buffer
//
// 关键点：
//   * 指针是普通的 UVA 指针——kernel 里写法与访问本地显存完全一样
//   * 前提是 cudaDeviceEnablePeerAccess：kernel 所在的卡（GPU0）要"开"对 GPU7 的访问
//   * 不 enable 会怎样？编译时加 -DNO_ENABLE 即可观察
//
// 编译：  nvcc -O3 -arch=native remote_access.cu -o remote_access
//         nvcc -O3 -arch=native -DNO_ENABLE remote_access.cu -o remote_access_noenable
// 运行：  ./remote_access
// ============================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define N ((size_t)1 << 24)          // 16M 个 unsigned = 64 MiB，足够演示又很快
#define BYTES (N * sizeof(unsigned))

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t e = (call);                                                  \
    if (e != cudaSuccess) {                                                  \
      fprintf(stderr, "FATAL %s:%d  %s: %s\n", __FILE__, __LINE__, #call,    \
              cudaGetErrorString(e));                                        \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

// ---------------------------------------------------------------------------
// 用 unsigned 作模式：2^24 个下标都能被精确表示，每个值唯一（见 0_to_7.cu 的注释）
// ---------------------------------------------------------------------------
__global__ void init_kernel(unsigned* p, size_t n) {          // 在本地写本地
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) p[i] = (unsigned)i;
}

// ---------------------------------------------------------------------------
// 远程读：remote 指向【另一张卡】的显存，local 在【本卡】
// 这个 kernel 在 GPU0 上运行；remote[i] 这一句会被硬件翻译成一次跨 NVLink 的读：
//   读请求 → 对端读 HBM → 读响应带回数据（见 §1.4"读与写不同"）
// ---------------------------------------------------------------------------
__global__ void remote_read(const unsigned* __restrict__ remote,
                            unsigned* __restrict__ local, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride)
    local[i] = remote[i] + 1;        // 读对端、写本地；+1 是为了让结果与源数据可区分
}

// ---------------------------------------------------------------------------
// 远程写：remote 指向另一张卡。remote[i] = ... 会变成一次跨 NVLink 的写：
//   写请求（带地址和数据）→ 对端写 HBM → 写确认
// ---------------------------------------------------------------------------
__global__ void remote_write(unsigned* __restrict__ remote, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride)
    remote[i] = (unsigned)i * 2u;    // 直接写进对端显存
}

// 启动配置：与 0_to_7.cu 相同的封顶策略（grid 封顶 4096，余量交给 grid-stride）
static void launch_dims(size_t n, int* grid, int* block) {
  *block = 256;
  size_t g = (n + *block - 1) / *block;
  if (g > 4096) g = 4096;
  *grid = (int)g;
}

// ----------------------------------------------------------------------------
// main：GPU0 是"访问者"（kernel 在它上面运行），GPU7 是"被访问者"（数据放在它上面）。
//
// 先分清两个问题，初学最容易混：
//   (1) 指针指向哪张卡的显存？ —— 由【cudaMalloc 时的当前设备】决定。
//   (2) kernel 在哪张卡上运行？ —— 由【<<<>>> 启动时的当前设备】决定。
// 两者互相独立。本程序正是故意让它们不一致：数据在 GPU7，kernel 却在 GPU0 上跑，
// 靠 cudaDeviceEnablePeerAccess 授权后，GPU0 的 kernel 才能直接解引用 GPU7 的指针。
//
// "当前设备"规则：cudaSetDevice 之后，cudaMalloc / kernel 启动 / cudaDeviceSynchronize
// 都作用在当前设备上，所以下面反复出现 cudaSetDevice(0) 与 cudaSetDevice(7)。
//
// 时间线（谁在哪张卡上干什么）：
//   1. 查询   GPU0 能否直接访问 GPU7
//   2. 准备   当前设备=7：分配 remote_buf 并本地填充；当前设备=0：分配 local_buf
//   3. 授权   当前设备=0：cudaDeviceEnablePeerAccess(7)  ← 方向：GPU0 访问 GPU7
//   A1 远程读  当前设备=0：remote_read（读 GPU7，写 GPU0 本地）
//   A2 远程写  当前设备=0：remote_write（写进 GPU7）；然后切到 7 取回数据核对
//
// 显存占用：两块各 64 MiB（N = 2^24 个 unsigned）。
// ----------------------------------------------------------------------------
int main() {
  // 本程序固定使用 0 号和 7 号卡（与 0_to_7.cu 一致），所以先确认机器上至少有 8 张。
  int ndev = 0;
  CHECK(cudaGetDeviceCount(&ndev));
  if (ndev < 8) { fprintf(stderr, "need 8 GPUs, found %d\n", ndev); return 1; }

  // ---- 1. 查询：GPU0 的 kernel 能直接访问 GPU7 的显存吗？ -----------------
  // can 只回答"拓扑上允许直连吗"，不改任何状态，返回 1 也不代表已经授权（见第 3 步）。
  // 它按方向问：这里问的是 0→7（GPU0 的 kernel 访问 GPU7）。
  int can = 0;
  CHECK(cudaDeviceCanAccessPeer(&can, 0, 7));
  printf("canAccessPeer(0->7) = %d\n", can);
  if (!can) { fprintf(stderr, "no direct path 0->7\n"); return 1; }

  // ---- 2. 准备：GPU7 上放"对端 buffer"，GPU0 上放"本地 buffer" ------------
  // 三个 kernel 共用同一组启动配置：launch_dims 算出 grid（封顶 4096）与 block（256），
  // 元素多于线程时由 kernel 里的 grid-stride 循环补上。
  int grid, block;
  launch_dims(N, &grid, &block);

  // 当前设备切到 7：接下来的 cudaMalloc 和 init_kernel 都作用在 GPU7 上。
  // 因此 remote_buf 这个指针"属于"GPU7——指针值本身是个普通的 64 位整数，
  // "属于哪张卡"记在驱动的分配记录里（cudaPointerGetAttributes 可查到 device=7）。
  CHECK(cudaSetDevice(7));
  unsigned* remote_buf;                       // 在 GPU7 上
  CHECK(cudaMalloc(&remote_buf, BYTES));
  init_kernel<<<grid, block>>>(remote_buf, N);   // GPU7 本地填 remote_buf[i] = i
  CHECK(cudaGetLastError());
  // 这次同步不能省：kernel 启动是异步的，而 GPU0 上稍后启动的 remote_read 并不会自动
  // 等待 GPU7 上的工作（跨设备的隐式顺序没有保证，推断）。显式同步后，GPU0 去读时 GPU7
  // 上的数据一定已经填好。（对比 0_to_7.cu：那里 fill_kernel 与拷贝之间没有这样的显式同步。）
  CHECK(cudaDeviceSynchronize());                // 等 GPU7 填完，再让 GPU0 去读

  // 当前设备切回 0：local_buf 属于 GPU0，也是后面所有 kernel 的"运行设备"。
  CHECK(cudaSetDevice(0));
  unsigned* local_buf;                        // 在 GPU0 上
  CHECK(cudaMalloc(&local_buf, BYTES));

  // ---- 3. 授权：当前设备是 GPU0，所以这句是"GPU0 访问 GPU7"这个方向 --------
  // 这是 kernel 远程解引用的硬前提；映射挂在本进程的 CUDA context 上。
  // 授权是单向的：这里只开了 GPU0→GPU7，没有开 GPU7→GPU0（本程序也不需要）。
  // 返回值有三种：成功；"已开启"（重复调用，幂等，视同成功）；其他（真失败）。
#ifndef NO_ENABLE
  cudaError_t e = cudaDeviceEnablePeerAccess(7, 0);   // flags 必须为 0
  if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
    fprintf(stderr, "enable failed: %s\n", cudaGetErrorString(e));
    return 1;
  }
  printf("peer access 0->7 enabled\n");
#else
  printf("!! 故意不调用 cudaDeviceEnablePeerAccess（-DNO_ENABLE）\n");
#endif

  // ---- [A1] 远程读：GPU0 的 kernel 读 GPU7 的 remote_buf -------------------
  // 注意 kernel 在【当前设备 GPU0】上启动，remote_buf 是 GPU7 的指针，
  // 写法上与本地指针没有任何区别。
  // 指针是按值传给 kernel 的：传过去的只是那个 64 位地址，真正的跨卡访问发生在 kernel
  // 里的 remote[i]——GPU0 的 MMU 查页表，命中第 3 步登记的映射，请求经 NVLink 到 GPU7。
  remote_read<<<grid, block>>>(remote_buf, local_buf, N);
  CHECK(cudaGetLastError());                     // <<<>>> 不返回错误码，启动后必须查
  CHECK(cudaDeviceSynchronize());                // 访问错误通常在这里才暴露
  // 对照 -DNO_ENABLE：没有第 3 步的授权时，上面的 cudaGetLastError 仍然通过（启动本身
  // 没问题），报错要到这一句同步才出现：an illegal memory access was encountered。

  // 校验在 host 上做：把 GPU0 的 local_buf 拷回来，逐个核对 local[i] == i + 1。
  // （kernel 里 +1 是为了让结果与源数据 remote[i] == i 可区分，避免"根本没读"也碰巧通过。）
  unsigned* h = (unsigned*)malloc(BYTES);
  CHECK(cudaMemcpy(h, local_buf, BYTES, cudaMemcpyDeviceToHost));
  size_t bad1 = 0;
  for (size_t i = 0; i < N; i++) if (h[i] != (unsigned)i + 1u) bad1++;
  printf("[A1] 远程读 : GPU0 kernel 读 GPU7 -> 不匹配 %zu 个 -> %s\n",
         bad1, bad1 == 0 ? "PASS" : "FAIL");

  // ---- [A2] 远程写：GPU0 的 kernel 把数据写进 GPU7 的 remote_buf ------------
  // 方向与 A1 相反：数据从 GPU0 的 SM 发往 GPU7（写请求带着数据，对端回写确认）。
  // 写的值是 i*2，与 A1 里 remote_buf 原来的内容（i）不同，所以"写没生效"一定能被发现。
  remote_write<<<grid, block>>>(remote_buf, N);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());                // 等 GPU0 的写全部完成

  // 切到 GPU7 再取回 remote_buf，只是为了让"这块内存属于谁"一目了然；在 UVA 下 cudaMemcpy
  // 是否必须切换设备，本程序未验证。上面的同步已保证 GPU0 的写全部完成。
  CHECK(cudaSetDevice(7));                       // 切到 GPU7 把 remote_buf 取回来看
  CHECK(cudaMemcpy(h, remote_buf, BYTES, cudaMemcpyDeviceToHost));
  size_t bad2 = 0;
  for (size_t i = 0; i < N; i++) if (h[i] != (unsigned)i * 2u) bad2++;
  printf("[A2] 远程写 : GPU0 kernel 写 GPU7 -> 不匹配 %zu 个 -> %s\n",
         bad2, bad2 == 0 ? "PASS" : "FAIL");

  // ---- 清理：每块内存回到它所属的设备上释放 ------------------------------
  // 当前设备此刻是 7，所以先释放 remote_buf；再切回 0 释放 local_buf。
  free(h);
  CHECK(cudaFree(remote_buf));
  CHECK(cudaSetDevice(0));
  CHECK(cudaFree(local_buf));
  // 退出码：0 = 两项都通过；1 = 有不匹配（NO_ENABLE 时更早就因非法访问退出了）。
  return (bad1 || bad2) ? 1 : 0;
}

// ============================================================================
// reg_bench.cu —— 探查"把 CPU 内存 / GPU 内存注册给网卡"的真实行为（单机，不需要对端）
//
// 对应文档 c-multi-node.md（§1 原理、§5.8 复现）。只做注册与注销，不收发数据，所以不需要 QP、不需要 OOB、不需要第二台机器。
//
// 子命令：
//   ./reg_bench lat   [设备名] [模式片段]   注册 / 注销耗时随大小的变化：CPU 未触摸、CPU 已触摸、GPU(nvidia_peermem)、GPU(dmabuf)
//   ./reg_bench bar1  [设备名]   GPU 注册前后 BAR1 的占用变化
//   ./reg_bench vmm   [设备名]   用 cuMemCreate（VMM）分配的显存，能否经 peermem / dmabuf 注册
//   ./reg_bench life  [设备名]   注册之后 cudaFree 会怎样（生命周期）
//   ./reg_bench attr  [设备名]   打印与 GPUDirect RDMA / dmabuf 相关的设备属性
//
// 编译： nvcc -O3 -arch=native reg_bench.cu -o reg_bench -libverbs -lcuda
// 默认设备 mlx5_0、GPU0（server-0 上 GPU0 就近的网卡）
// ============================================================================

#include <cuda_runtime.h>
#include <cuda.h>
#include <infiniband/verbs.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define DIE(...) do { fprintf(stderr, "FATAL %s:%d  ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); \
                      fprintf(stderr, "\n"); exit(1); } while (0)
#define CHECK_CUDA(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) DIE("%s: %s", #call, cudaGetErrorString(e_)); } while (0)
#define CHECK_CU(call) do { CUresult r_ = (call); if (r_ != CUDA_SUCCESS) { const char* s_ = nullptr; \
    cuGetErrorString(r_, &s_); DIE("%s: %s", #call, s_ ? s_ : "?"); } } while (0)

static const int ACC = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
static const size_t MiB = 1 << 20;

static double now_us() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec * 1e-3; }

struct Dev { ibv_context* ctx; ibv_pd* pd; };

static Dev open_dev(const char* name) {
  int n = 0; ibv_device** l = ibv_get_device_list(&n);
  if (!l) DIE("ibv_get_device_list");
  ibv_device* d = nullptr;
  for (int i = 0; i < n; i++) if (!strcmp(ibv_get_device_name(l[i]), name)) d = l[i];
  if (!d) DIE("找不到设备 %s", name);
  Dev r; r.ctx = ibv_open_device(d); ibv_free_device_list(l);
  if (!r.ctx) DIE("打开 %s 失败", name);
  r.pd = ibv_alloc_pd(r.ctx);
  if (!r.pd) DIE("ibv_alloc_pd");
  return r;
}

// 一次注册 + 注销的结果
struct Res { bool ok; int err; double reg_us; double dereg_us; double export_us; };

// ---- CPU 内存：posix_memalign；touch=true 表示注册前先把每页写一遍（页已经在内存里）
static Res reg_cpu(Dev& d, size_t sz, bool touch) {
  void* p = nullptr;
  if (posix_memalign(&p, 4096, sz)) DIE("posix_memalign");
  if (touch) memset(p, 0xA5, sz);
  Res r{}; double t0 = now_us();
  ibv_mr* mr = ibv_reg_mr(d.pd, p, sz, ACC);
  double t1 = now_us();
  r.ok = mr != nullptr; r.err = r.ok ? 0 : errno; r.reg_us = t1 - t0;
  if (mr) { double t2 = now_us(); ibv_dereg_mr(mr); r.dereg_us = now_us() - t2; }
  free(p);
  return r;
}

// ---- GPU 显存，经 nvidia_peermem：ibv_reg_mr 直接收 cudaMalloc 的指针
static Res reg_gpu_peermem(Dev& d, size_t sz) {
  void* p = nullptr; CHECK_CUDA(cudaMalloc(&p, sz));
  Res r{}; double t0 = now_us();
  ibv_mr* mr = ibv_reg_mr(d.pd, p, sz, ACC);
  double t1 = now_us();
  r.ok = mr != nullptr; r.err = r.ok ? 0 : errno; r.reg_us = t1 - t0;
  if (mr) { double t2 = now_us(); ibv_dereg_mr(mr); r.dereg_us = now_us() - t2; }
  CHECK_CUDA(cudaFree(p));
  return r;
}

// ---- GPU 显存，经 dmabuf：先让 CUDA 把这段显存导出成一个 dmabuf 文件描述符，再 ibv_reg_dmabuf_mr
static Res reg_dmabuf_ptr(Dev& d, CUdeviceptr ptr, size_t sz) {
  Res r{};
  int fd = -1; double t0 = now_us();
  CUresult cr = cuMemGetHandleForAddressRange(&fd, ptr, sz, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
  double t1 = now_us(); r.export_us = t1 - t0;
  if (cr != CUDA_SUCCESS) { r.ok = false; r.err = -(int)cr; return r; }       // err<0 表示 CUDA 导出失败
  ibv_mr* mr = ibv_reg_dmabuf_mr(d.pd, 0, sz, (uint64_t)ptr, fd, ACC);
  double t2 = now_us();
  r.ok = mr != nullptr; r.err = r.ok ? 0 : errno; r.reg_us = t2 - t1;
  if (mr) { double t3 = now_us(); ibv_dereg_mr(mr); r.dereg_us = now_us() - t3; }
  close(fd);
  return r;
}
static Res reg_gpu_dmabuf(Dev& d, size_t sz) {
  void* p = nullptr; CHECK_CUDA(cudaMalloc(&p, sz));
  Res r = reg_dmabuf_ptr(d, (CUdeviceptr)p, sz);
  CHECK_CUDA(cudaFree(p));
  return r;
}

static const char* errstr(const Res& r) {
  static char b[96];
  if (r.ok) return "成功";
  if (r.err < 0) { const char* s = nullptr; cuGetErrorString((CUresult)(-r.err), &s); snprintf(b, sizeof b, "CUDA 导出失败：%s", s ? s : "?"); }
  else snprintf(b, sizeof b, "失败：errno=%d (%s)", r.err, strerror(r.err));
  return b;
}

// ---------------------------------------------------------------------------
static void cmd_lat(Dev& d, const char* only) {
  const size_t sizes[] = {1 * MiB, 64 * MiB, 1024 * MiB, 4096 * MiB};
  const int REPS = 5;
  printf("%-26s %8s | %12s %12s | %12s | %12s\n", "方式", "大小", "注册(中位 ms)", "注册(最小 ms)", "注销(中位 ms)", "导出(中位 ms)");
  struct M { const char* name; int id; };
  const M modes[] = {{"CPU（未触摸的页）", 0}, {"CPU（已触摸的页）", 1}, {"GPU / nvidia_peermem", 2}, {"GPU / dmabuf", 3}};
  reg_gpu_peermem(d, 1 * MiB); reg_cpu(d, 1 * MiB, true); reg_gpu_dmabuf(d, 1 * MiB);        // 预热
  for (auto& m : modes) for (size_t sz : sizes) {
    if (only && !strstr(m.name, only)) continue;                       // 按名字片段过滤，例如 "dmabuf"
    std::vector<double> reg, dereg, exp; bool ok = true; Res last{};
    for (int i = 0; i < REPS; i++) {
      Res r = m.id == 0 ? reg_cpu(d, sz, false) : m.id == 1 ? reg_cpu(d, sz, true)
            : m.id == 2 ? reg_gpu_peermem(d, sz) : reg_gpu_dmabuf(d, sz);
      last = r; if (!r.ok) { ok = false; break; }
      reg.push_back(r.reg_us); dereg.push_back(r.dereg_us); exp.push_back(r.export_us);
    }
    if (!ok) { printf("%-26s %5zu MiB | %s\n", m.name, sz / MiB, errstr(last)); continue; }
    std::sort(reg.begin(), reg.end()); std::sort(dereg.begin(), dereg.end()); std::sort(exp.begin(), exp.end());
    if (m.id == 3) printf("%-26s %5zu MiB | %12.3f %12.3f | %12.3f | %12.3f\n", m.name, sz / MiB, reg[REPS / 2] / 1e3, reg[0] / 1e3, dereg[REPS / 2] / 1e3, exp[REPS / 2] / 1e3);
    else           printf("%-26s %5zu MiB | %12.3f %12.3f | %12.3f | %12s\n", m.name, sz / MiB, reg[REPS / 2] / 1e3, reg[0] / 1e3, dereg[REPS / 2] / 1e3, "-");
  }
}

static void bar1(int gpu, char* out, size_t n) {
  // 这版 nvidia-smi 的 --query-gpu 不支持 bar1.* 字段，所以解析 -q -d MEMORY 里的 "BAR1 Memory Usage"
  char cmd[256]; snprintf(cmd, sizeof cmd,
    "nvidia-smi -i %d -q -d MEMORY | awk '/BAR1 Memory Usage/{f=1} f&&/Total/{t=$3} f&&/Used/{u=$3; print t\",\"u; exit}'", gpu);
  FILE* f = popen(cmd, "r"); out[0] = 0; if (f) { if (!fgets(out, n, f)) out[0] = 0; pclose(f); }
  size_t l = strlen(out); if (l && out[l - 1] == '\n') out[l - 1] = 0;
}
static void cmd_bar1(Dev& d, int gpu) {
  char b[128]; const size_t sz = 4096 * MiB;
  bar1(gpu, b, sizeof b);  printf("[基线]                         BAR1 total, used (MiB) = %s\n", b);
  void* p = nullptr; CHECK_CUDA(cudaMalloc(&p, sz));
  bar1(gpu, b, sizeof b);  printf("[cudaMalloc 4 GiB，未注册]      BAR1 total, used (MiB) = %s\n", b);
  ibv_mr* mr = ibv_reg_mr(d.pd, p, sz, ACC);
  if (!mr) DIE("peermem 注册失败");
  bar1(gpu, b, sizeof b);  printf("[ibv_reg_mr 4 GiB (peermem)]   BAR1 total, used (MiB) = %s\n", b);
  ibv_dereg_mr(mr);
  bar1(gpu, b, sizeof b);  printf("[注销之后]                      BAR1 total, used (MiB) = %s\n", b);
  int fd = -1; CHECK_CU(cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)p, sz, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
  mr = ibv_reg_dmabuf_mr(d.pd, 0, sz, (uint64_t)p, fd, ACC);
  if (!mr) DIE("dmabuf 注册失败");
  bar1(gpu, b, sizeof b);  printf("[ibv_reg_dmabuf_mr 4 GiB]      BAR1 total, used (MiB) = %s\n", b);
  ibv_dereg_mr(mr); close(fd);
  bar1(gpu, b, sizeof b);  printf("[注销之后]                      BAR1 total, used (MiB) = %s\n", b);
  CHECK_CUDA(cudaFree(p));
}

static void cmd_attr(int gpu) {
  CUdevice dev; CHECK_CU(cuDeviceGet(&dev, gpu));
  struct A { const char* name; CUdevice_attribute a; };
  const A as[] = {
    {"GPU_DIRECT_RDMA_SUPPORTED（支持 GPUDirect RDMA）",                       CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED},
    {"GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED（VMM 显存也支持）",              CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED},
    {"DMA_BUF_SUPPORTED（支持导出 dmabuf）",                                   CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED},
    {"VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED（支持 VMM）",                        CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED},
    {"GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS（RDMA 写的 flush 选项位掩码）",     CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS},
    {"GPU_DIRECT_RDMA_WRITES_ORDERING（RDMA 写的有序性级别）",                 CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING},
  };
  for (auto& x : as) { int v = -1; CUresult r = cuDeviceGetAttribute(&v, x.a, dev); printf("  %-62s = %d%s\n", x.name, v, r ? "（查询失败）" : ""); }
}

static void cmd_vmm(Dev& d, int gpu) {
  CUdevice dev; CHECK_CU(cuDeviceGet(&dev, gpu));
  for (int rdma_cap = 0; rdma_cap <= 1; rdma_cap++) {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = gpu;
    prop.allocFlags.gpuDirectRDMACapable = rdma_cap;
    size_t gran = 0; CHECK_CU(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    size_t sz = ((256 * MiB + gran - 1) / gran) * gran;
    CUmemGenericAllocationHandle h; CUdeviceptr ptr;
    CUresult cr = cuMemCreate(&h, sz, &prop, 0);
    if (cr != CUDA_SUCCESS) { const char* s = nullptr; cuGetErrorString(cr, &s); printf("[gpuDirectRDMACapable=%d] cuMemCreate 失败：%s\n", rdma_cap, s ? s : "?"); continue; }
    CHECK_CU(cuMemAddressReserve(&ptr, sz, 0, 0, 0));
    CHECK_CU(cuMemMap(ptr, sz, 0, h, 0));
    CUmemAccessDesc acc{}; acc.location = prop.location; acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CU(cuMemSetAccess(ptr, sz, &acc, 1));
    printf("[VMM 分配 %zu MiB，粒度 %zu KiB，gpuDirectRDMACapable=%d]\n", sz / MiB, gran / 1024, rdma_cap);
    ibv_mr* mr = ibv_reg_mr(d.pd, (void*)ptr, sz, ACC);
    int e = mr ? 0 : errno;
    printf("    经 nvidia_peermem（ibv_reg_mr）：%s\n", mr ? "成功" : strerror(e));
    if (mr) ibv_dereg_mr(mr);
    Res r = reg_dmabuf_ptr(d, ptr, sz);
    printf("    经 dmabuf（ibv_reg_dmabuf_mr）：%s\n", errstr(r));
    cuMemUnmap(ptr, sz); cuMemAddressFree(ptr, sz); cuMemRelease(h);
  }
}

static void cmd_life(Dev& d) {
  for (int mode = 0; mode < 2; mode++) {
    void* p = nullptr; const size_t sz = 256 * MiB; CHECK_CUDA(cudaMalloc(&p, sz));
    int fd = -1; ibv_mr* mr = nullptr;
    if (mode == 0) mr = ibv_reg_mr(d.pd, p, sz, ACC);
    else { CHECK_CU(cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)p, sz, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
           mr = ibv_reg_dmabuf_mr(d.pd, 0, sz, (uint64_t)p, fd, ACC); }
    if (!mr) { printf("[%s] 注册失败\n", mode ? "dmabuf" : "peermem"); continue; }
    printf("[%s] 已注册 %zu MiB；现在 cudaFree 这块显存（MR 仍未注销）\n", mode ? "dmabuf" : "peermem", sz / MiB);
    cudaError_t e = cudaFree(p);
    printf("    cudaFree 返回：%s\n", cudaGetErrorString(e));
    int rc = ibv_dereg_mr(mr);
    printf("    随后 ibv_dereg_mr 返回：%d%s\n", rc, rc ? "（失败）" : "（成功）");
    if (fd >= 0) close(fd);
    cudaGetLastError();
  }
  printf("    进程仍然存活，没有崩溃\n");
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "用法: %s lat|bar1|vmm|life|attr [设备名]\n", argv[0]); return 2; }
  const char* devname = argc > 2 ? argv[2] : "mlx5_0";
  CHECK_CUDA(cudaSetDevice(0)); CHECK_CUDA(cudaFree(0));
  Dev d = open_dev(devname);
  if (!strcmp(argv[1], "lat")) cmd_lat(d, argc > 3 ? argv[3] : nullptr);
  else if (!strcmp(argv[1], "bar1")) cmd_bar1(d, 0);
  else if (!strcmp(argv[1], "vmm")) cmd_vmm(d, 0);
  else if (!strcmp(argv[1], "life")) cmd_life(d);
  else if (!strcmp(argv[1], "attr")) cmd_attr(0);
  else { fprintf(stderr, "未知子命令 %s\n", argv[1]); return 2; }
  return 0;
}

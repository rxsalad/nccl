// ============================================================================
// server-gpu.cu —— 跨机 GPU↔GPU 的 RDMA 实验：服务端
//              运行在 server-1，使用 GPU7；与运行在 server-0（GPU0）的 client-gpu.cu 配对
//
// 两条通道，别混：
//   * OOB（带外）：eth0 上的 TCP。只传【控制信息】——QP 号、GID、内存地址与 rkey，
//     以及"写完了/校验结果"这类同步信号。数据不走它。
//   * 数据：RDMA（RoCE v2）。client 的网卡经 RDMA WRITE / READ 直接读写本机 GPU7 的显存；
//     本机 CPU 不参与数据搬运（单边操作），GPU 显存由网卡经 GPUDirect RDMA 直接访问。
//
// 本程序是"被动方"：它准备好显存、注册、把地址与 rkey 告诉 client，然后只负责校验。
//
// 显存布局（GPU7 上一块 2×region 字节的 buffer，每个区域 n 个 unsigned）：
//   [0, n)  ：留给 client【写入】（RDMA WRITE 的目标），初始为 0；  期望值 A[i] = i*3 + 1
//   [n, 2n) ：留给 client【读取】（RDMA READ 的来源），初始为 B；    B[i] = i*5 + 2
//
// 与 §3.4 的 ipc_tcp.cu 对照：那里是"同一台机器上的两个进程经 CUDA IPC 共享显存"；
// 这里是"两台机器"，CUDA IPC 不能跨机，所以改用 RDMA。
//
// 编译： nvcc -O3 -arch=native server-gpu.cu -o server-gpu -libverbs -lcuda
// 运行： ./server-gpu [-p 端口] [-d RDMA设备] [-g GPU号] [-x GID下标] [-s 区域MiB] [-t 流量类别] [-i OOB网口] [-m peermem|dmabuf]
//        默认：-p 5555 -d mlx5_14 -g 7 -s 64 -t 0 -i eth0 -m peermem，GID 自动选择
//        -m peermem|dmabuf：把 GPU 显存注册给网卡的方式，默认 peermem（nvidia_peermem 模块）；
//           dmabuf 不依赖该模块（CUDA 导出 dmabuf fd + ibv_reg_dmabuf_mr）。两端可以各自选择，互不影响。
// 调试：nvcc -g -G -O0 server-gpu.cu -o server-gpu.debug -libverbs -lcuda，用 cuda-gdb（既能看主机代码也能进 kernel）。
//   常用断点：ib_setup（注册显存）、ib_connect（QP 到 RTS）、count_bad（校验）。
//   server 会停在 accept() 等 client；断点停太久时 client 的 OOB 同步会一直等。
//   两端的 -t 流量类别要一致（本集群上推荐 -t 104）。
// ============================================================================

#include <cuda_runtime.h>
#include <cuda.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>

// ---------------------------------------------------------------------------
// 错误处理：裸 C 没有异常，每个调用的返回值都要看
// ---------------------------------------------------------------------------
#define DIE(...) do { fprintf(stderr, "[server] FATAL %s:%d  ", __FILE__, __LINE__); \
                      fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } while (0)
#define CHECK_CUDA(call) do { cudaError_t e_ = (call); \
    if (e_ != cudaSuccess) DIE("%s: %s", #call, cudaGetErrorString(e_)); } while (0)
#define CHECK_CU(call) do { CUresult r_ = (call); \
    if (r_ != CUDA_SUCCESS) { const char* s_ = nullptr; cuGetErrorString(r_, &s_); \
      DIE("%s: %s", #call, s_ ? s_ : "?"); } } while (0)

// ---------------------------------------------------------------------------
// 校验用的数据模式：p[i] = i * mul + add（unsigned，每个下标的值唯一，错位也能被发现）
// ---------------------------------------------------------------------------
__global__ void fill_pattern(unsigned* p, size_t n, unsigned mul, unsigned add) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) p[i] = (unsigned)i * mul + add;
}

static void launch_fill(unsigned* p, size_t n, unsigned mul, unsigned add) {
  int block = 256;
  size_t grid = (n + block - 1) / block;
  if (grid > 4096) grid = 4096;
  fill_pattern<<<(int)grid, block>>>(p, n, mul, add);
  CHECK_CUDA(cudaGetLastError());
}

// 把 GPU 上的 n 个 unsigned 拷回 host，数出与期望模式不一致的个数
static size_t count_bad(const unsigned* d_ptr, size_t n, unsigned mul, unsigned add) {
  unsigned* h = (unsigned*)malloc(n * sizeof(unsigned));
  if (!h) DIE("malloc 失败");
  CHECK_CUDA(cudaMemcpy(h, d_ptr, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (size_t i = 0; i < n; i++) if (h[i] != (unsigned)i * mul + add) bad++;
  free(h);
  return bad;
}

// ---------------------------------------------------------------------------
// OOB：TCP。流式协议，一次 send/recv 不保证收发完整，必须循环到凑够字节数
// ---------------------------------------------------------------------------
static int send_all(int fd, const void* buf, size_t len) {
  const char* p = (const char*)buf;
  while (len) { ssize_t k = send(fd, p, len, MSG_NOSIGNAL); if (k <= 0) return -1; p += k; len -= k; }
  return 0;
}
static int recv_all(int fd, void* buf, size_t len) {
  char* p = (char*)buf;
  while (len) { ssize_t k = recv(fd, p, len, 0); if (k <= 0) return -1; p += k; len -= k; }
  return 0;
}

// 取指定网口（默认 eth0）的 IPv4 地址：让 OOB 明确走 eth0，而不是"碰巧哪个口"
static bool get_if_ipv4(const char* ifname, in_addr* out) {
  ifaddrs* list = nullptr;
  if (getifaddrs(&list)) return false;
  bool found = false;
  for (ifaddrs* a = list; a; a = a->ifa_next) {
    if (a->ifa_addr && a->ifa_addr->sa_family == AF_INET && !strcmp(a->ifa_name, ifname)) {
      *out = ((sockaddr_in*)a->ifa_addr)->sin_addr; found = true; break;
    }
  }
  freeifaddrs(list);
  return found;
}

// ---------------------------------------------------------------------------
// 经 OOB 交换的连接信息：建立 RC QP 和做单边读写所需的全部"对端信息"
// 两端都是 x86，字节序与结构体布局一致，所以直接按原始字节收发
// ---------------------------------------------------------------------------
struct ConnInfo {
  uint32_t qpn;          // QP 号
  uint32_t psn;          // 起始包序号
  ibv_gid  gid;          // RoCE 用 GID 寻址（相当于 IP 地址）
  uint64_t addr;         // 已注册显存的起始地址（GPU 的 UVA 地址）
  uint32_t rkey;         // 远端访问凭证：client 做 RDMA 读写时必须带上
  uint64_t len;          // 已注册的字节数
};

// ---------------------------------------------------------------------------
// ibverbs 资源
// ---------------------------------------------------------------------------
struct Ib {
  ibv_context*  ctx = nullptr;   // 设备上下文：打开网卡（ibv_open_device）得到，后面所有资源都从它创建
  ibv_pd*       pd  = nullptr;   // 保护域 Protection Domain：QP 与 MR 必须属于同一个 PD 才能配合使用
  ibv_cq*       cq  = nullptr;   // 完成队列 Completion Queue：网卡把"某个操作做完了"的完成项放在这里，程序用 ibv_poll_cq 取
  ibv_qp*       qp  = nullptr;   // 队列对 Queue Pair：RDMA 通信的端点；RC 类型 = 可靠连接，一对一
  ibv_mr*       mr  = nullptr;   // 内存区域 Memory Region：已注册给网卡的那块【显存】，含 lkey（本端用）与 rkey（给对端用）
  ibv_port_attr pa{};            // 端口属性：状态、链路层、active_mtu、GID 表长度等
  int           gid_idx = -1;    // 选中的 GID 下标：RoCE 用 GID 当"IP 地址"来寻址
  ibv_gid       gid{};           // 本端 GID 的内容（16 字节，发给对端）
  uint32_t      psn = 0;         // 本端 QP 的起始包序号 Packet Sequence Number，发给对端
  int           dmabuf_fd = -1;  // dmabuf 方式下导出的文件描述符（peermem 方式为 -1）；注销 MR 之后要自己 close
  double        reg_ms = 0;      // 注册耗时（毫秒）；dmabuf 方式含"导出 + 注册"
};

// 自动选 GID：RoCE v2 且不是链路本地（fe80::）的那一条。
// 本机实测：gid[1] 是 fe80 的 RoCE v2（链路本地，不能跨机路由），gid[3] 才是 fabric 的 ULA 地址。
static int pick_gid(ibv_context* ctx, int port, int tbl_len, int want) {
  if (want >= 0) return want;
  for (int i = 0; i < tbl_len; i++) {
    ibv_gid_entry e;
    if (ibv_query_gid_ex(ctx, port, i, &e, 0)) continue;
    if (e.gid_type != IBV_GID_TYPE_ROCE_V2) continue;
    if (e.gid.raw[0] == 0xfe && e.gid.raw[1] == 0x80) continue;          // 链路本地
    static const unsigned char zero[16] = {0};
    if (!memcmp(e.gid.raw, zero, 16)) continue;                           // 空表项
    return i;
  }
  return -1;
}

static void ib_setup(Ib& ib, const char* devname, int gid_arg, void* gpu_buf, size_t bytes, bool use_dmabuf) {
  // 找到并打开网卡。ibv_get_device_list 列出本 Pod 能看到的全部 RDMA 设备（mlx5_0、mlx5_2 ...），
  // 按名字挑出 devname，再 ibv_open_device 打开。打开后设备列表就可以释放了。
  int n = 0;
  ibv_device** list = ibv_get_device_list(&n);
  if (!list) DIE("ibv_get_device_list 失败");
  ibv_device* dev = nullptr;
  for (int i = 0; i < n; i++) if (!strcmp(ibv_get_device_name(list[i]), devname)) dev = list[i];
  if (!dev) DIE("找不到 RDMA 设备 %s（本 Pod 只有偶数号网卡可用：mlx5_0/2/.../14）", devname);
  ib.ctx = ibv_open_device(dev);
  if (!ib.ctx) DIE("打开 %s 失败（是否缺少 /dev/infiniband/uverbsN？）", devname);
  ibv_free_device_list(list);

  // 查询端口 1 的属性并检查：端口 ACTIVE、链路层是以太网（即 RoCE，而不是 InfiniBand）。
  // 网卡只有一个端口，端口号固定写 1。
  if (ibv_query_port(ib.ctx, 1, &ib.pa)) DIE("ibv_query_port 失败");
  if (ib.pa.state != IBV_PORT_ACTIVE) DIE("%s 的端口不是 ACTIVE", devname);
  if (ib.pa.link_layer != IBV_LINK_LAYER_ETHERNET) DIE("%s 不是 Ethernet（RoCE）链路层", devname);

  // 确定 GID。RoCE v2 没有 InfiniBand 的 LID，对端靠 GID（形如 IPv6 地址）找到本端。
  ib.gid_idx = pick_gid(ib.ctx, 1, ib.pa.gid_tbl_len, gid_arg);
  if (ib.gid_idx < 0) DIE("没有可用的 RoCE v2 GID");
  if (ibv_query_gid(ib.ctx, 1, ib.gid_idx, &ib.gid)) DIE("ibv_query_gid 失败");

  ib.pd = ibv_alloc_pd(ib.ctx);                                           // 保护域：QP、MR 都挂在它下面
  if (!ib.pd) DIE("ibv_alloc_pd 失败");

  // 把【GPU 显存】注册给网卡，有两种方式（-m 选择，见下）。peermem 方式能成功是因为宿主机加载了 nvidia_peermem（GPUDirect RDMA）：
  // 它让 ibverbs 认得 GPU 指针，并把显存页的物理地址交给网卡做 DMA。
  // 访问权限：本端写（LOCAL_WRITE）、允许对端 RDMA 读/写（REMOTE_READ/WRITE）。
  const int acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  timespec r0, r1;  clock_gettime(CLOCK_MONOTONIC, &r0);                  // 注册耗时（dmabuf 方式含"导出 + 注册"）
  if (!use_dmabuf) {
    // 方式一（默认）：nvidia_peermem。ibv_reg_mr 直接收 GPU 指针。
    ib.mr = ibv_reg_mr(ib.pd, gpu_buf, bytes, acc);
    if (!ib.mr) DIE("ibv_reg_mr(GPU 显存) 失败：宿主机是否加载了 nvidia_peermem？");
  } else {
    // 方式二：dmabuf。① 让 CUDA 把这段显存导出成一个 dmabuf 文件描述符；② 用 ibv_reg_dmabuf_mr 注册这个 fd。
    // 不依赖 nvidia_peermem，走 Linux 内核通用的 dmabuf 框架（需要 NVIDIA 开放内核模块等条件）。
    // 参数：offset = 0（从 fd 对应缓冲的起点）；iova = 显存指针本身——这样对端用的远端地址与 peermem 方式一致（仍是这个 UVA 地址）。
    CHECK_CU(cuMemGetHandleForAddressRange(&ib.dmabuf_fd, (CUdeviceptr)gpu_buf, bytes,
                                           CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
    ib.mr = ibv_reg_dmabuf_mr(ib.pd, 0, bytes, (uint64_t)gpu_buf, ib.dmabuf_fd, acc);
    if (!ib.mr) DIE("ibv_reg_dmabuf_mr 失败：errno=%d (%s)", errno, strerror(errno));
  }
  clock_gettime(CLOCK_MONOTONIC, &r1);
  ib.reg_ms = ((r1.tv_sec - r0.tv_sec) + (r1.tv_nsec - r0.tv_nsec) * 1e-9) * 1e3;

  // 完成队列。深度 256 要不小于 QP 的 max_send_wr（256），否则 CQ 可能溢出。
  // 参数 nullptr, nullptr, 0：不用完成事件通道，由程序自己忙轮询（ibv_poll_cq）。
  ib.cq = ibv_create_cq(ib.ctx, 256, nullptr, nullptr, 0);                // 完成队列：本端发出的操作做完后在这里出现
  if (!ib.cq) DIE("ibv_create_cq 失败");

  // 创建 QP（初始在 RESET 状态）。
  ibv_qp_init_attr qa{};
  qa.send_cq = ib.cq; qa.recv_cq = ib.cq;
  qa.qp_type = IBV_QPT_RC;                                                // 可靠连接：支持单边 READ/WRITE
  // 容量：发送队列最多 256 个未完成的请求（并发深度 -w 不能超过它）；本程序只做单边操作，
  // 不会 post 接收请求，所以接收队列只要 1 个占位；每个请求只带 1 个 SGE（一段连续内存）。
  qa.cap.max_send_wr = 256; qa.cap.max_recv_wr = 1;
  qa.cap.max_send_sge = 1;  qa.cap.max_recv_sge = 1;
  ib.qp = ibv_create_qp(ib.pd, &qa);
  if (!ib.qp) DIE("ibv_create_qp 失败");

  // QP 状态机第 1 步：RESET → INIT
  // 先说明：新建的 QP 在 RESET 状态；切到 INIT 时要给出 pkey、端口，
  // 以及【允许对端对本 QP 做哪些远端操作】——不开 REMOTE_READ/WRITE，对端的 RDMA 读写会被拒绝。
  ibv_qp_attr a{};
  a.qp_state = IBV_QPS_INIT; a.pkey_index = 0; a.port_num = 1;
  a.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
  if (ibv_modify_qp(ib.qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
    DIE("QP → INIT 失败");

  ib.psn = (uint32_t)(getpid() * 2654435761u) & 0xffffff;                 // 24 位随机起始序号
}

// QP 状态机第 2、3 步：INIT → RTR（准备接收）→ RTS（准备发送）。需要对端信息，所以要等 OOB 交换完
static void ib_connect(Ib& ib, const ConnInfo& r, int tc) {
  ibv_qp_attr a{};
  a.qp_state = IBV_QPS_RTR;
  a.path_mtu = ib.pa.active_mtu;                                          // 本机实测 4096
  a.dest_qp_num = r.qpn; a.rq_psn = r.psn;
  // max_dest_rd_atomic：本端作为被读方，最多同时处理多少个对端发来的 RDMA READ；
  // min_rnr_timer：对端接收没准备好（RNR）时的重试间隔，本程序不用接收，仅需给个合法值。
  a.max_dest_rd_atomic = 16; a.min_rnr_timer = 12;
  a.ah_attr.is_global = 1;                                                // RoCE 必须带 GRH，用 GID 寻址
  a.ah_attr.grh.dgid = r.gid;
  a.ah_attr.grh.sgid_index = ib.gid_idx;
  a.ah_attr.grh.hop_limit = 64;                                           // 两端 fabric 在不同子网，要经三层转发
  a.ah_attr.grh.traffic_class = tc;                                       // 流量类别（DSCP<<2），交换机据此分队列
  a.ah_attr.sl = 0; a.ah_attr.port_num = 1;
  if (ibv_modify_qp(ib.qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                    IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
    DIE("QP → RTR 失败");

  ibv_qp_attr b{};
  b.qp_state = IBV_QPS_RTS;
  // RTS 参数——可靠连接的重传策略：
  //   timeout=14   ：等 ACK 的超时 = 4.096 µs × 2^14 ≈ 67 ms；
  //   retry_cnt=7  ：超时后最多重传 7 次，再失败就报 IBV_WC_RETRY_EXC_ERR（对端不可达/路径不通时会看到）；
  //   rnr_retry=7  ：7 表示对端 RNR 时无限重试；
  //   sq_psn       ：本端发送的起始包序号，要与对端 RTR 里的 rq_psn 一致（OOB 里交换过）。
  b.timeout = 14; b.retry_cnt = 7; b.rnr_retry = 7;
  b.sq_psn = ib.psn; b.max_rd_atomic = 16;
  if (ibv_modify_qp(ib.qp, &b, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                    IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
    DIE("QP → RTS 失败");
}

static void print_gid(const char* tag, const ibv_gid& g) {
  char s[INET6_ADDRSTRLEN];
  inet_ntop(AF_INET6, g.raw, s, sizeof s);
  printf("[server] %s%s\n", tag, s);
}

int main(int argc, char** argv) {
  int port = 5555, gpu = 7, gid_arg = -1, tc = 0;
  size_t region_mib = 64;
  const char* dev = "mlx5_14";   // GPU7 就近的网卡：nvidia-smi topo -m 里 GPU7 与 NIC14/NIC15 为 PXB
  const char* oob_if = "eth0";
  bool use_dmabuf = false;       // -m：false = peermem（默认），true = dmabuf
  // 命令行参数。-s 两端必须一致（否则交换连接信息后退出）；-t 要与 client 相同；
  // -g 选 GPU，-d 选网卡（默认是与该 GPU 同一 PCIe 交换机下的就近网卡）；
  // -i 指定 OOB 用哪个网口，默认 eth0，目的是让控制信息明确走以太网、不与 RDMA 数据混在一起。
  int opt;
  while ((opt = getopt(argc, argv, "p:d:g:x:s:t:i:m:")) != -1) {
    switch (opt) {
      case 'p': port = atoi(optarg); break;
      case 'd': dev = optarg; break;
      case 'g': gpu = atoi(optarg); break;
      case 'x': gid_arg = atoi(optarg); break;
      case 's': region_mib = (size_t)atoll(optarg); break;
      case 't': tc = atoi(optarg); break;
      case 'm':
        if (!strcmp(optarg, "dmabuf")) use_dmabuf = true;
        else if (strcmp(optarg, "peermem")) { fprintf(stderr, "-m 只能是 peermem 或 dmabuf\n"); return 2; }
        break;
      case 'i': oob_if = optarg; break;
      default: fprintf(stderr, "用法: %s [-p 端口] [-d RDMA设备] [-g GPU号] [-x GID下标] [-s 区域MiB] [-t 流量类别] [-i OOB网口] [-m peermem|dmabuf]\n", argv[0]); return 2;
    }
  }
  const size_t region = region_mib << 20;                    // 每个区域的字节数
  const size_t n = region / sizeof(unsigned);                // 每个区域的元素个数
  const size_t total = 2 * region;                           // 注册的总字节数

  // ---- 1. GPU：分配并准备数据 -------------------------------------------
  // buf 的布局（total = 2 × region 字节，整块注册成一个 MR）：
  //
  //   buf                    buf + n（元素）= buf + region（字节）          buf + 2n
  //   |<----- 区域 [0, n) ---->|<--------- 区域 [n, 2n) --------->|
  //
  // buf 是 unsigned*，所以 buf + n 是"跳过 n 个 unsigned"（= region 字节），不是跳过 n 字节。
  // buf 是 cudaMalloc 返回的【设备指针】：在统一虚拟地址（UVA）下它同时也是网卡能认的显存地址，
  // 下面的 ibv_reg_mr 直接收这个指针；CPU 不能直接解引用它（要读必须 cudaMemcpy，见 count_bad）。
  CHECK_CUDA(cudaSetDevice(gpu));
  unsigned* buf = nullptr;
  CHECK_CUDA(cudaMalloc(&buf, total));
  // GPUDirect RDMA 的惯例：把这块显存标成"同步内存操作"，保证 CUDA 的内存操作在 RDMA 访问前完成
  unsigned one = 1;
  // 注意第一个参数传的是 &one（属性值），最后一个才是目标显存地址；SYNC_MEMOPS=1 表示对这块显存的
  // 同步类内存操作（如 cudaMemcpy、cudaMemset）在返回前对 GPUDirect RDMA 都已可见。
  CHECK_CU(cuPointerSetAttribute(&one, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)buf));
  CHECK_CUDA(cudaMemset(buf, 0, region));                    // [0, n)：留给 client 写入，先清零
  launch_fill(buf + n, n, 5u, 2u);                           // [n, 2n)：给 client 读取，B[i] = i*5 + 2
  CHECK_CUDA(cudaDeviceSynchronize());                       // 数据必须先就绪，再把地址交给对端
  printf("[server] GPU%d 上 buf = %p，总 %zu MiB（每区域 %zu MiB）\n", gpu, (void*)buf, total >> 20, region_mib);

  // ---- 2. RDMA：打开网卡、注册显存、创建 QP -----------------------------
  Ib ib;
  ib_setup(ib, dev, gid_arg, buf, total, use_dmabuf);
  printf("[server] RDMA 设备 %s：active_mtu=%d，max_msg_sz=%u，使用 gid[%d]\n", dev,
         (int)(128 << ib.pa.active_mtu), ib.pa.max_msg_sz, ib.gid_idx);
  print_gid("本端 GID = ", ib.gid);
  printf("[server] 已注册 GPU 显存（%s）：addr=%p  rkey=0x%x  len=%zu；注册耗时 %.3f ms\n",
         use_dmabuf ? "dmabuf" : "peermem", (void*)buf, ib.mr->rkey, total, ib.reg_ms);

  // ---- 3. OOB：在 eth0 上监听，等 client 连接 ---------------------------
  in_addr ip;
  if (!get_if_ipv4(oob_if, &ip)) DIE("找不到网口 %s 的 IPv4 地址", oob_if);
  // 监听套接字。SO_REUSEADDR：程序刚退出时端口还在 TIME_WAIT，不加这一项马上重启会 bind 失败。
  // 只绑定 eth0 的 IPv4 地址（而不是 0.0.0.0），保证 OOB 一定走 eth0。
  int ls = socket(AF_INET, SOCK_STREAM, 0);
  int yes = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes);
  sockaddr_in addr; memset(&addr, 0, sizeof addr);
  addr.sin_family = AF_INET; addr.sin_addr = ip; addr.sin_port = htons(port);   // 只绑 eth0 的地址
  if (bind(ls, (sockaddr*)&addr, sizeof addr) || listen(ls, 1)) DIE("bind/listen 失败");
  char ipstr[INET_ADDRSTRLEN]; inet_ntop(AF_INET, &ip, ipstr, sizeof ipstr);
  printf("[server] OOB 监听 %s:%d（%s），等待 client ...\n", ipstr, port, oob_if);
  fflush(stdout);                                            // 重定向到文件时是全缓冲，不刷新外面看不到"就绪"
  // accept 会一直阻塞到 client 连上来：此时 MR 已经注册好，QP 在 INIT，只差对端信息。
  int cs = accept(ls, nullptr, nullptr);
  if (cs < 0) DIE("accept 失败");

  // ---- 4. 交换连接信息，QP 切到 RTS -------------------------------------
  ConnInfo mine{}, theirs{};
  mine.qpn = ib.qp->qp_num; mine.psn = ib.psn; mine.gid = ib.gid;
  mine.addr = (uint64_t)(uintptr_t)buf; mine.rkey = ib.mr->rkey; mine.len = total;
  // 交换顺序：server 先收后发，client 先发后收，两边正好对上；如果两端都先收，就会互相等待（死锁）。
  // 交换的内容就是 ConnInfo：本端 QP 号 / PSN / GID，以及已注册显存的地址、rkey、长度。
  if (recv_all(cs, &theirs, sizeof theirs) || send_all(cs, &mine, sizeof mine)) DIE("交换连接信息失败");
  if (theirs.len != total) DIE("两端区域大小不一致：本端 %zu，对端 %lu（两端的 -s 要相同）", total, (unsigned long)theirs.len);
  print_gid("对端 GID = ", theirs.gid);
  ib_connect(ib, theirs, tc);
  printf("[server] QP 已就绪（RTS），等待 client 的读写与校验请求\n");
  fflush(stdout);

  // ---- 5. 服务循环：RDMA 是单边操作，server 的 CPU 不参与数据搬运；只响应校验请求 ---
  // OOB 命令协议（client → server，每条 1 字节）：
  //   'W' ：RDMA WRITE 做完了，请校验 [0, n)           → 回复 'K'（全部正确）或 'X'（有不匹配）
  //   'B' ：带宽测试做完了，请再校验一次 [0, n)         → 同上
  //   'Q' ：全部结束，退出循环
  // 单边操作对 server 是"无声"的：网卡直接把数据写进显存，server 的 CPU 完全不知道，
  // 所以必须靠这条 OOB 消息告诉它"可以看了"。循环里 recv_all 是阻塞的，等待期间 CPU 几乎不占用。
  int rc = 0;
  for (;;) {
    char cmd = 0;
    if (recv_all(cs, &cmd, 1)) { fprintf(stderr, "[server] client 提前断开\n"); rc = 1; break; }
    if (cmd == 'Q') break;
    if (cmd == 'W' || cmd == 'B') {
      // client 说"RDMA WRITE 已完成"：校验 [0, n) 是否等于 A[i] = i*3 + 1
      // 此时数据是网卡直接写进 GPU 显存的，server 的 CPU 全程没有碰过它
      size_t bad = count_bad(buf, n, 3u, 1u);
      printf("[server] 校验 client 写进来的数据（%s）：不匹配 %zu 个 -> %s\n",
             cmd == 'W' ? "RDMA WRITE 之后" : "带宽测试之后", bad, bad == 0 ? "PASS" : "FAIL");
      fflush(stdout);
      char ack = bad == 0 ? 'K' : 'X';
      if (bad) rc = 1;
      if (send_all(cs, &ack, 1)) { rc = 1; break; }
    }
  }

  // ---- 6. 清理 ----------------------------------------------------------
  close(cs); close(ls);
  // 销毁顺序：QP → CQ → MR → PD → 设备，最后才释放显存；MR 必须在 cudaFree 之前注销，
  // PD 必须在它下面的 QP/MR 都销毁之后才能释放。
  ibv_destroy_qp(ib.qp); ibv_destroy_cq(ib.cq); ibv_dereg_mr(ib.mr);
  if (ib.dmabuf_fd >= 0) close(ib.dmabuf_fd);                              // dmabuf 方式：注销 MR 之后再关 fd
  ibv_dealloc_pd(ib.pd); ibv_close_device(ib.ctx);
  CHECK_CUDA(cudaFree(buf));
  printf("[server] 结束，退出码 %d\n", rc);
  return rc;
}

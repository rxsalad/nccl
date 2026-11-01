/* ============================================================================
 * client-cpu.c —— 跨机 RDMA 实验（主机 CPU 内存版）：客户端（主动方）
 *                 运行在 server-0，与运行在 server-1 的 server-cpu.c 配对
 *
 * 本程序演示 RDMA 单边操作（WRITE / READ），读写的是【普通进程内存】（主机内存）。
 * 纯 C、不依赖 CUDA，只用 libibverbs。
 *
 * 两条通道，别混：
 *   * OOB（带外）：eth0 上的 TCP。只传【控制信息】——QP 号、GID、对端内存的地址与 rkey，
 *     以及"写完了/校验结果"这类同步信号。数据不走它。
 *   * 数据：RDMA（RoCE v2）。本机网卡经 RDMA WRITE / READ 直接读写【对端的内存】，
 *     对端 CPU 不参与（单边操作）；本机一侧同样由网卡直接访问本机内存。
 *
 * 本程序是"主动方"：连接 server、建 QP、发起 RDMA 读写、校验、测带宽。
 *
 * 内存布局（一块 2×region 字节的 buffer，每个区域 n 个 unsigned）：
 *   [0, n)  ：本端要【写给对端】的数据（RDMA WRITE 的来源），A[i] = i*3 + 1
 *   [n, 2n) ：从对端【读回来】的落点（RDMA READ 的目标），初始为 0；期望读到 B[i] = i*5 + 2
 *
 * 实验步骤：
 *   1. RDMA WRITE：本端 [0,n) → 对端 [0,n)；通知 server 校验
 *   2. RDMA READ ：对端 [n,2n) → 本端 [n,2n)；本端校验
 *   3. 带宽：连续多次 WRITE、READ，测 GB/s
 *
 * 内存的来源与注册：
 *   posix_memalign 分配（4096 字节对齐），【先把每一页写一遍再注册】；ibv_reg_mr 走内核的标准钉页路径。
 *   为什么先写：没写过的页还没有物理内存，注册时内核得先替你分配，这部分成本会被算进注册耗时；
 *   不先写也能注册成功，只是更慢、计时不干净（这是机制上的说法，未做对照测量）。
 *
 * 编译： gcc -O2 -Wall -Wextra client-cpu.c -o client-cpu -libverbs
 * 运行： ./client-cpu -a <server 的 eth0 IP> [-p 端口] [-d RDMA设备] [-x GID下标]
 *                     [-s 区域MiB] [-n 迭代数] [-w 并发深度] [-t 流量类别] [-o 顺序] [-W 窗口]
 *        默认：-p 5555 -d mlx5_0 -s 64 -n 20 -w 8 -t 0，GID 自动选择
 *        注意：-s 必须与 server 一致
 *        -o RW 或 -o WR：在正式步骤之前，先【逐次计时】连续 6 次 READ（R）与 6 次 WRITE（W），
 *           按给定顺序执行，用来观察"冷启动"（第一次 READ 为什么慢）。加了 -o 之后，
 *           后面正式步骤里的"首次 WRITE/READ 耗时"就不再是真正的第一次了。
 *        -W 窗口：带宽测试时，每完成"窗口"次操作就记一个时间点，结束后打印【逐窗口的带宽】，
 *           用来观察带宽随时间的变化（例如开头有没有一段预热期）。0 表示不打印（默认）。
 * 调试：gcc -g -O0 client-cpu.c -o client-cpu.debug -libverbs，用普通 gdb。
 *   常用断点：post_op（发起操作）、wait_cq（等完成）、run_bw（带宽循环）。
 *   在 gdb 里：p theirs（对端信息）；p/x theirs.rkey；x/8u buf+n 看读回来的数据（应是 2 7 12 17 ...）。
 *   client 在 wait_cq / run_bw 里是忙轮询（CPU 约 98%），Ctrl-C 可以打断。
 *   两端的 -t 流量类别要一致（本集群上推荐 -t 104）：实测 -t 0 时主机内存的 READ 首次要几十毫秒且带宽不稳。
 *   带宽测试的 -n 要足够大：并发深度 ≥ 2 时 WRITE 有一段爬升期，-n 100 会严重低估（见 c-multi-node-cpu.md §5.4）。
 * ========================================================================== */

#define _GNU_SOURCE
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 错误处理：C 没有异常，每个调用的返回值都要看 */
#define DIE(...) do { fprintf(stderr, "[client-cpu] FATAL %s:%d  ", __FILE__, __LINE__); \
                      fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } while (0)

static double now_sec(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------------------------------------------------------------------------
 * 校验用的数据模式：p[i] = i * mul + add（unsigned，每个下标的值唯一，错位也能被发现）
 * 直接在主机上用循环填充 / 校验：被检查的内存就是主机内存，不需要拷贝
 * ------------------------------------------------------------------------- */
static void fill_pattern(unsigned* p, size_t n, unsigned mul, unsigned add) {
  for (size_t i = 0; i < n; i++) p[i] = (unsigned)i * mul + add;
}
static size_t count_bad(const unsigned* p, size_t n, unsigned mul, unsigned add) {
  size_t bad = 0;
  for (size_t i = 0; i < n; i++) if (p[i] != (unsigned)i * mul + add) bad++;
  return bad;
}

/* ---------------------------------------------------------------------------
 * OOB：TCP。流式协议，一次 send/recv 不保证收发完整，必须循环到凑够字节数
 * ------------------------------------------------------------------------- */
static int send_all(int fd, const void* buf, size_t len) {
  const char* p = (const char*)buf;
  while (len) { ssize_t k = send(fd, p, len, MSG_NOSIGNAL); if (k <= 0) return -1; p += k; len -= (size_t)k; }
  return 0;
}
static int recv_all(int fd, void* buf, size_t len) {
  char* p = (char*)buf;
  while (len) { ssize_t k = recv(fd, p, len, 0); if (k <= 0) return -1; p += k; len -= (size_t)k; }
  return 0;
}

/* ---------------------------------------------------------------------------
 * 经 OOB 交换的连接信息：建立 RC QP 和做单边读写所需的全部"对端信息"
 * 两端都是 x86，字节序与结构体布局一致，所以直接按原始字节收发
 * ------------------------------------------------------------------------- */
struct conn_info {
  uint32_t qpn;          /* QP 号 */
  uint32_t psn;          /* 起始包序号 */
  union ibv_gid gid;     /* RoCE 用 GID 寻址（相当于 IP 地址） */
  uint64_t addr;         /* 已注册内存的起始地址（主机虚拟地址） */
  uint32_t rkey;         /* 远端访问凭证：做 RDMA 读写时必须带上 */
  uint64_t len;          /* 已注册的字节数 */
};

/* ---------------------------------------------------------------------------
 * ibverbs 资源
 * ------------------------------------------------------------------------- */
struct ib {
  struct ibv_context*  ctx;      /* 设备上下文：打开网卡（ibv_open_device）得到，后面所有资源都从它创建 */
  struct ibv_pd*       pd;       /* 保护域 Protection Domain：QP 与 MR 必须属于同一个 PD 才能配合使用 */
  struct ibv_cq*       cq;       /* 完成队列 Completion Queue：网卡把"某个操作做完了"的完成项放在这里，程序用 ibv_poll_cq 取 */
  struct ibv_qp*       qp;       /* 队列对 Queue Pair：RDMA 通信的端点；RC 类型 = 可靠连接，一对一 */
  struct ibv_mr*       mr;       /* 内存区域 Memory Region：已注册给网卡的那块内存，含 lkey（本端用）与 rkey（给对端用） */
  struct ibv_port_attr pa;       /* 端口属性：状态、链路层、active_mtu、GID 表长度等 */
  int                  gid_idx;  /* 选中的 GID 下标：RoCE 用 GID 当"IP 地址"来寻址 */
  union ibv_gid        gid;      /* 本端 GID 的内容（16 字节，发给对端） */
  uint32_t             psn;      /* 本端 QP 的起始包序号 Packet Sequence Number，发给对端 */
  double               reg_ms;      /* ibv_reg_mr 的耗时（毫秒），启动时打印 */
};

/* 自动选 GID：RoCE v2 且不是链路本地（fe80::）的那一条。
 * 本机实测：gid[1] 是 fe80 的 RoCE v2（链路本地，不能跨机路由），gid[3] 才是 fabric 的 ULA 地址。 */
static int pick_gid(struct ibv_context* ctx, int port, int tbl_len, int want) {
  if (want >= 0) return want;
  static const unsigned char zero[16] = {0};
  for (int i = 0; i < tbl_len; i++) {
    struct ibv_gid_entry e;
    if (ibv_query_gid_ex(ctx, (uint32_t)port, (uint32_t)i, &e, 0)) continue;
    if (e.gid_type != IBV_GID_TYPE_ROCE_V2) continue;
    if (e.gid.raw[0] == 0xfe && e.gid.raw[1] == 0x80) continue;      /* 链路本地 */
    if (!memcmp(e.gid.raw, zero, 16)) continue;                       /* 空表项 */
    return i;
  }
  return -1;
}

static void ib_setup(struct ib* ib, const char* devname, int gid_arg, void* buf, size_t bytes) {
  /* 找到并打开网卡。ibv_get_device_list 列出本 Pod 能看到的全部 RDMA 设备（mlx5_0、mlx5_2 ...），
   * 按名字挑出 devname，再 ibv_open_device 打开。打开后设备列表就可以释放了。 */
  int n = 0;
  struct ibv_device** list = ibv_get_device_list(&n);
  if (!list) DIE("ibv_get_device_list 失败");
  struct ibv_device* dev = NULL;
  for (int i = 0; i < n; i++) if (!strcmp(ibv_get_device_name(list[i]), devname)) dev = list[i];
  if (!dev) DIE("找不到 RDMA 设备 %s（本 Pod 只有偶数号网卡可用：mlx5_0/2/.../14）", devname);
  ib->ctx = ibv_open_device(dev);
  if (!ib->ctx) DIE("打开 %s 失败（是否缺少 /dev/infiniband/uverbsN？）", devname);
  ibv_free_device_list(list);

  /* 查询端口 1 的属性并做三项检查：端口 ACTIVE、链路层是以太网（即 RoCE，而不是 InfiniBand）。
   * 网卡只有一个端口，端口号固定写 1。 */
  if (ibv_query_port(ib->ctx, 1, &ib->pa)) DIE("ibv_query_port 失败");
  if (ib->pa.state != IBV_PORT_ACTIVE) DIE("%s 的端口不是 ACTIVE", devname);
  if (ib->pa.link_layer != IBV_LINK_LAYER_ETHERNET) DIE("%s 不是 Ethernet（RoCE）链路层", devname);

  /* 确定 GID。RoCE v2 没有 InfiniBand 的 LID，对端靠 GID（形如 IPv6 地址）找到本端。 */
  ib->gid_idx = pick_gid(ib->ctx, 1, ib->pa.gid_tbl_len, gid_arg);
  if (ib->gid_idx < 0) DIE("没有可用的 RoCE v2 GID");
  if (ibv_query_gid(ib->ctx, 1, ib->gid_idx, &ib->gid)) DIE("ibv_query_gid 失败");

  ib->pd = ibv_alloc_pd(ib->ctx);                                     /* 保护域：QP、MR 都挂在它下面 */
  if (!ib->pd) DIE("ibv_alloc_pd 失败");

  /* 把【主机内存】注册给网卡：内核钉住这些页、把各页的总线地址写进网卡的翻译表（MTT），并发给我们 lkey/rkey。
   * 这是注册的"标准情形"，不需要额外的内核模块。
   * LOCAL_WRITE：RDMA READ 的数据要写进本端内存，必须有这项权限。
   * 耗时被单独计下来：数据已经事先写过一遍（页都在物理内存里），所以这里主要是钉页与建表的成本。 */
  double t0 = now_sec();
  ib->mr = ibv_reg_mr(ib->pd, buf, bytes,
                      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
  ib->reg_ms = (now_sec() - t0) * 1e3;
  if (!ib->mr) DIE("ibv_reg_mr(主机内存) 失败：errno=%d (%s)", errno, strerror(errno));

  /* 完成队列。深度 256 要不小于 QP 的 max_send_wr（256），否则 CQ 可能溢出。
   * 参数 NULL, NULL, 0：不用完成事件通道，由程序自己忙轮询（ibv_poll_cq）。 */
  ib->cq = ibv_create_cq(ib->ctx, 256, NULL, NULL, 0);                /* 完成队列：本端发出的操作做完后在这里出现 */
  if (!ib->cq) DIE("ibv_create_cq 失败");

  /* 创建 QP（RESET 状态）。 */
  struct ibv_qp_init_attr qa;
  memset(&qa, 0, sizeof qa);
  qa.send_cq = ib->cq; qa.recv_cq = ib->cq;
  qa.qp_type = IBV_QPT_RC;                                            /* 可靠连接：支持单边 READ/WRITE */
  /* 容量：发送队列最多 256 个未完成的请求（并发深度 -w 不能超过它）；本程序只做单边操作，
   * 不会 post 接收请求，所以接收队列只要 1 个占位；每个请求只带 1 个 SGE（一段连续内存）。 */
  qa.cap.max_send_wr = 256; qa.cap.max_recv_wr = 1;
  qa.cap.max_send_sge = 1;  qa.cap.max_recv_sge = 1;
  ib->qp = ibv_create_qp(ib->pd, &qa);
  if (!ib->qp) DIE("ibv_create_qp 失败");

  /* QP 状态机第 1 步：RESET → INIT */
  /* 先说明：新建的 QP 在 RESET 状态；切到 INIT 时要给出 pkey、端口，
   * 以及【允许对端对本 QP 做哪些远端操作】——不开 REMOTE_READ/WRITE，对端的 RDMA 读写会被拒绝。 */
  struct ibv_qp_attr a;
  memset(&a, 0, sizeof a);
  a.qp_state = IBV_QPS_INIT; a.pkey_index = 0; a.port_num = 1;
  a.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
  if (ibv_modify_qp(ib->qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
    DIE("QP → INIT 失败");

  ib->psn = ((uint32_t)getpid() * 2654435761u) & 0xffffff;            /* 24 位随机起始序号 */
}

/* QP 状态机第 2、3 步：INIT → RTR（准备接收）→ RTS（准备发送）。需要对端信息，所以要等 OOB 交换完 */
static void ib_connect(struct ib* ib, const struct conn_info* r, int tc) {
  struct ibv_qp_attr a;
  memset(&a, 0, sizeof a);
  a.qp_state = IBV_QPS_RTR;
  a.path_mtu = ib->pa.active_mtu;                                     /* 本机实测 4096 */
  a.dest_qp_num = r->qpn; a.rq_psn = r->psn;
  /* max_dest_rd_atomic：本端作为被读方，最多同时处理多少个对端发来的 RDMA READ；
   * min_rnr_timer：对端接收没准备好（RNR）时的重试间隔，本程序不用接收，仅需给个合法值。 */
  a.max_dest_rd_atomic = 16; a.min_rnr_timer = 12;
  a.ah_attr.is_global = 1;                                            /* RoCE 必须带 GRH，用 GID 寻址 */
  a.ah_attr.grh.dgid = r->gid;
  a.ah_attr.grh.sgid_index = (uint8_t)ib->gid_idx;
  a.ah_attr.grh.hop_limit = 64;                                       /* 两端 fabric 在不同子网，要经三层转发 */
  a.ah_attr.grh.traffic_class = (uint8_t)tc;                          /* 流量类别（DSCP<<2） */
  a.ah_attr.sl = 0; a.ah_attr.port_num = 1;
  if (ibv_modify_qp(ib->qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                    IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
    DIE("QP → RTR 失败");

  struct ibv_qp_attr b;
  memset(&b, 0, sizeof b);
  b.qp_state = IBV_QPS_RTS;
  /* RTS 参数——可靠连接的重传策略：
   *   timeout=14   ：等 ACK 的超时 = 4.096 µs × 2^14 ≈ 67 ms；
   *   retry_cnt=7  ：超时后最多重传 7 次，再失败就报 IBV_WC_RETRY_EXC_ERR（对端不可达/路径不通时会看到）；
   *   rnr_retry=7  ：7 表示对端 RNR 时无限重试；
   *   sq_psn       ：本端发送的起始包序号，要与对端 RTR 里的 rq_psn 一致（OOB 里交换过）。 */
  b.timeout = 14; b.retry_cnt = 7; b.rnr_retry = 7;
  b.sq_psn = ib->psn; b.max_rd_atomic = 16;                           /* max_rd_atomic：同时在飞的 RDMA READ 上限 */
  if (ibv_modify_qp(ib->qp, &b, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                    IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
    DIE("QP → RTS 失败");
}

/* ---------------------------------------------------------------------------
 * 发起一次单边操作（RDMA WRITE 或 READ）：
 *   WRITE：把本地 [laddr, laddr+len) 写到远端 [raddr, raddr+len)
 *   READ ：把远端 [raddr, raddr+len) 读到本地 [laddr, laddr+len)
 * 远端的 CPU 不参与。sge.length 是 32 位，单次最多 4 GiB（本实验远小于此）。
 * ------------------------------------------------------------------------- */
static void post_op(struct ib* ib, enum ibv_wr_opcode op, uint64_t laddr, uint64_t raddr, uint32_t rkey,
                    size_t len, uint64_t wr_id) {
  struct ibv_sge sge;
  memset(&sge, 0, sizeof sge);
  sge.addr = laddr; sge.length = (uint32_t)len; sge.lkey = ib->mr->lkey;
  struct ibv_send_wr wr, *bad = NULL;
  memset(&wr, 0, sizeof wr);
  wr.wr_id = wr_id; wr.sg_list = &sge; wr.num_sge = 1;
  wr.opcode = op;
  wr.send_flags = IBV_SEND_SIGNALED;                                  /* 做完后在 CQ 里产生一条完成项 */
  wr.wr.rdma.remote_addr = raddr; wr.wr.rdma.rkey = rkey;
  if (ibv_post_send(ib->qp, &wr, &bad)) DIE("ibv_post_send 失败");
}

/* 从 CQ 里取走 want 条完成项；任何一条状态不是 SUCCESS 都视为失败（如对端不可达会报重试耗尽） */
static void wait_cq(struct ib* ib, int want) {
  struct ibv_wc wc[16];
  int got = 0;
  time_t t0 = time(NULL);
  while (got < want) {
    int k = ibv_poll_cq(ib->cq, 16, wc);
    if (k < 0) DIE("ibv_poll_cq 失败");
    for (int i = 0; i < k; i++) {
      if (wc[i].status != IBV_WC_SUCCESS)
        DIE("完成项状态异常：%s（wr_id=%lu）", ibv_wc_status_str(wc[i].status), (unsigned long)wc[i].wr_id);
      got++;
    }
    if (time(NULL) - t0 > 30) DIE("等待完成项超时（30 秒）");
  }
}

/* 带宽测试：连续 iters 次操作，始终保持最多 depth 个在飞；返回总耗时（秒）。
 * window > 0 时，每完成 window 次操作记一个时间点，结束后（不干扰测量）打印逐窗口的带宽。 */
static double run_bw(struct ib* ib, enum ibv_wr_opcode op, uint64_t laddr, uint64_t raddr, uint32_t rkey,
                     size_t len, int iters, int depth, int window) {
  double* marks = NULL;
  if (window > 0) { marks = (double*)calloc((size_t)(iters / window + 1), sizeof(double)); if (!marks) DIE("calloc 失败"); }
  double t0 = now_sec();
  int posted = 0, done = 0;
  struct ibv_wc wc[16];
  while (done < iters) {
    while (posted < iters && posted - done < depth) { post_op(ib, op, laddr, raddr, rkey, len, (uint64_t)posted); posted++; }
    int k = ibv_poll_cq(ib->cq, 16, wc);
    if (k < 0) DIE("ibv_poll_cq 失败");
    for (int i = 0; i < k; i++) {
      if (wc[i].status != IBV_WC_SUCCESS) DIE("完成项状态异常：%s", ibv_wc_status_str(wc[i].status));
      done++;
      if (marks && done % window == 0) marks[done / window - 1] = now_sec();
    }
  }
  double total = now_sec() - t0;
  if (marks) {
    printf("[client-cpu]   逐窗口带宽（每 %d 次一个窗口，GB/s）：", window);
    double prev = t0;
    for (int w = 0; w < iters / window; w++) {
      printf(" %.1f", (double)len * window / (marks[w] - prev) / 1e9);
      prev = marks[w];
    }
    printf("\n");
    free(marks);
  }
  return total;
}

static void print_gid(const char* tag, const union ibv_gid* g) {
  char s[INET6_ADDRSTRLEN];
  inet_ntop(AF_INET6, g->raw, s, sizeof s);
  printf("[client-cpu] %s%s\n", tag, s);
}

static void usage(const char* prog) {
  fprintf(stderr, "用法: %s -a <server 的 eth0 IP> [-p 端口] [-d RDMA设备] [-x GID下标] "
                  "[-s 区域MiB] [-n 迭代数] [-w 并发深度] [-t 流量类别] [-o RW|WR] [-W 窗口]\n", prog);
  exit(2);
}

int main(int argc, char** argv) {
  int port = 5555, gid_arg = -1, tc = 0, iters = 20, depth = 8;
  size_t region_mib = 64;
  const char* dev = "mlx5_0";    /* client 用 mlx5_0，server 用 mlx5_14：各自一块可用的偶数号网卡 */
  const char* server_ip = NULL;
  const char* probe_order = NULL;   /* -o：逐次计时探查的顺序，NULL 表示不做 */
  int window = 0;                   /* -W：逐窗口带宽的窗口大小，0 表示不打印 */
  /* 命令行参数。-a 必填（server 的 eth0 IP）；-s 必须与 server 一致；-t 要与 server 相同；
   * -n 迭代数、-w 并发深度、-o、-W 是测量与诊断选项，见文件头的说明。 */
  int opt;
  while ((opt = getopt(argc, argv, "a:p:d:x:s:n:w:t:o:W:")) != -1) {
    switch (opt) {
      case 'a': server_ip = optarg; break;
      case 'p': port = atoi(optarg); break;
      case 'd': dev = optarg; break;
      case 'x': gid_arg = atoi(optarg); break;
      case 's': region_mib = (size_t)atoll(optarg); break;
      case 'n': iters = atoi(optarg); break;
      case 'w': depth = atoi(optarg); break;
      case 't': tc = atoi(optarg); break;
      case 'o': probe_order = optarg; break;
      case 'W': window = atoi(optarg); break;
      default: usage(argv[0]);
    }
  }
  if (!server_ip) usage(argv[0]);
  const size_t region = region_mib << 20;                    /* 每个区域的字节数 */
  const size_t n = region / sizeof(unsigned);                /* 每个区域的元素个数 */
  const size_t total = 2 * region;                           /* 注册的总字节数 */

  /* ---- 1. 主机内存：分配并准备数据 -------------------------------------
   * buf 的布局（total = 2 × region 字节，整块注册成一个 MR）：
   *
   *   buf                       buf + n（元素）= buf + region（字节）      buf + 2n
   *   |<------ 区域 [0, n) ----->|<-------- 区域 [n, 2n) -------->|
   *
   * posix_memalign 按 4096 字节（一页）对齐：注册是按页做的，对齐后首尾不会多占半个页。
   * 注意 buf 的类型是 unsigned*，所以 buf + n 是"跳过 n 个 unsigned"（= region 字节），不是跳过 n 字节。
   * 注册前必须把每一页都写一遍（见下面的 memset / fill_pattern）：没写过的页还没有物理内存，
   * 注册时内核得先替你分配，这部分成本会被算进注册耗时（机制说法，未做对照测量）。 */
  unsigned* buf = NULL;
  if (posix_memalign((void**)&buf, 4096, total)) DIE("posix_memalign 失败");
  fill_pattern(buf, n, 3u, 1u);                              /* [0, n)：要写给对端的数据 A[i] = i*3 + 1 */
  memset(buf + n, 0, region);                                /* [n, 2n)：读回来的落点，先清零（同时触摸了这些页） */
  printf("[client-cpu] 主机内存 buf = %p，总 %zu MiB（每区域 %zu MiB）；注册前所有页都已写过一遍\n",
         (void*)buf, total >> 20, region_mib);

  /* ---- 2. RDMA：打开网卡、注册内存、创建 QP ---------------------------- */
  struct ib ib;
  memset(&ib, 0, sizeof ib);
  ib_setup(&ib, dev, gid_arg, buf, total);
  printf("[client-cpu] RDMA 设备 %s：active_mtu=%d，max_msg_sz=%u，使用 gid[%d]；ibv_reg_mr 耗时 %.3f ms\n", dev,
         128 << ib.pa.active_mtu, ib.pa.max_msg_sz, ib.gid_idx, ib.reg_ms);
  print_gid("本端 GID = ", &ib.gid);

  /* ---- 3. OOB：经 eth0 连接 server（server 没起来就重试）--------------- */
  /* 重试循环：server 可能还没 listen，connect 失败就关掉套接字、等 100 ms 再来，最多 300 次（约 30 秒）。
   * 每次重试要重新 socket()，因为失败的套接字不能再次 connect。 */
  int s = -1;
  for (int tries = 0; tries < 300; tries++) {
    s = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, server_ip, &a.sin_addr) != 1) DIE("非法的 server IP：%s", server_ip);
    if (connect(s, (struct sockaddr*)&a, sizeof a) == 0) break;
    close(s); s = -1; usleep(100 * 1000);                    /* 100 ms 后重试，最多约 30 秒 */
  }
  if (s < 0) DIE("连不上 %s:%d", server_ip, port);
  printf("[client-cpu] OOB 已连接 %s:%d\n", server_ip, port);

  /* ---- 4. 交换连接信息，QP 切到 RTS ------------------------------------ */
  struct conn_info mine, theirs;
  memset(&mine, 0, sizeof mine); memset(&theirs, 0, sizeof theirs);
  mine.qpn = ib.qp->qp_num; mine.psn = ib.psn; mine.gid = ib.gid;
  mine.addr = (uint64_t)(uintptr_t)buf; mine.rkey = ib.mr->rkey; mine.len = total;
  /* 交换顺序：client 先发后收，server 先收后发（见 server-cpu.c），避免两端同时等对方而死锁。 */
  if (send_all(s, &mine, sizeof mine) || recv_all(s, &theirs, sizeof theirs)) DIE("交换连接信息失败");
  if (theirs.len != total)
    DIE("两端区域大小不一致：本端 %zu，对端 %lu（两端的 -s 要相同）", total, (unsigned long)theirs.len);
  print_gid("对端 GID = ", &theirs.gid);
  printf("[client-cpu] 对端内存：addr=0x%lx  rkey=0x%x  len=%lu\n",
         (unsigned long)theirs.addr, theirs.rkey, (unsigned long)theirs.len);
  ib_connect(&ib, &theirs, tc);
  printf("[client-cpu] QP 已就绪（RTS）\n");

  /* 地址约定：RDMA 的"本地地址"和"远端地址"都是各自进程里的【虚拟地址】（整数形式）。
   * 对端的 addr 与 rkey 是对端注册内存时拿到的、经 OOB 告诉我们的；两端的 buf 起始地址不同是正常的。
   * 之后每次 RDMA 操作 = (本地地址, 远端地址, rkey, 长度)，其中 rkey 必须是对端的那一个。 */
  const uint64_t l_base = (uint64_t)(uintptr_t)buf;          /* 本端起始地址 */
  const uint64_t r_base = theirs.addr;                       /* 对端起始地址 */
  /* rc：整个程序的退出码（任何一步校验失败就置 1）；ack：从 server 收到的校验回复，'K' = 通过，'X' = 有不匹配。 */
  int rc = 0;
  char ack = 0;

  /* ---- 4.5（可选）逐次计时探查：-o RW / -o WR ---------------------------
   * 每次操作单独 post、单独等完成、单独计时，一共 6 次，看"慢"是只有第一次还是每次。
   * 这一步不经 OOB 与 server 交互：WRITE 写的是和正式步骤相同的数据 A，READ 读的是 B，
   * 所以不会影响后面的校验。 */
  if (probe_order) {
    for (const char* c = probe_order; *c; c++) {
      int is_read = (*c == 'R');
      printf("[client-cpu] 探查：连续 6 次 %s（每次 %zu MiB，单独计时）：", is_read ? "READ " : "WRITE", region_mib);
      for (int k = 0; k < 6; k++) {
        double q0 = now_sec();
        if (is_read) post_op(&ib, IBV_WR_RDMA_READ, l_base + region, r_base + region, theirs.rkey, region, (uint64_t)(100 + k));
        else         post_op(&ib, IBV_WR_RDMA_WRITE, l_base, r_base, theirs.rkey, region, (uint64_t)(100 + k));
        wait_cq(&ib, 1);
        printf(" %.2f", (now_sec() - q0) * 1e3);
      }
      printf(" ms\n");
    }
  }

  /* ---- 5. RDMA WRITE：本端 [0,n) → 对端 [0,n) -------------------------- */
  /* 对端的 CPU 没有参与：数据由两端的网卡直接在两块内存之间搬运 */
  /* 计时只覆盖"post 到收到完成项"这一段，不含后面的 OOB 校验往返。
   * wr_id（最后一个参数）只是个标签，会原样出现在完成项里；这里用 1 表示 WRITE、2 表示 READ，方便出错时辨认。
   * 对 RC 的 WRITE，"完成项出现"表示数据已被对端网卡确认收到（可靠连接的 ACK）。 */
  double t0 = now_sec();
  post_op(&ib, IBV_WR_RDMA_WRITE, l_base, r_base, theirs.rkey, region, 1);
  wait_cq(&ib, 1);
  double t1 = now_sec();
  /* RDMA WRITE 对对端是"无声"的：对端不知道数据已经到了。所以经 OOB 告诉它去校验 */
  if (send_all(s, "W", 1) || recv_all(s, &ack, 1)) DIE("OOB 同步失败");
  printf("[client-cpu] RDMA WRITE  本端 → 对端  %zu MiB：耗时 %.2f ms；对端校验 %s\n",
         region_mib, (t1 - t0) * 1e3, ack == 'K' ? "PASS" : "FAIL");
  if (ack != 'K') rc = 1;

  /* ---- 6. RDMA READ：对端 [n,2n) → 本端 [n,2n) ------------------------- */
  t0 = now_sec();
  post_op(&ib, IBV_WR_RDMA_READ, l_base + region, r_base + region, theirs.rkey, region, 2);
  wait_cq(&ib, 1);
  t1 = now_sec();
  size_t bad = count_bad(buf + n, n, 5u, 2u);                /* 读回来的数据应等于对端预置的 B[i] = i*5 + 2；直接读本机内存即可 */
  printf("[client-cpu] RDMA READ   对端 → 本端  %zu MiB：耗时 %.2f ms；本端校验：不匹配 %zu 个 -> %s\n",
         region_mib, (t1 - t0) * 1e3, bad, bad == 0 ? "PASS" : "FAIL");
  if (bad) rc = 1;

  /* ---- 7. 带宽 --------------------------------------------------------- */
  /* 注意：单 QP、单条消息 = 整个区域；上面的 WRITE/READ 已经做过一次，等于预热 */
  /* run_bw 的做法：循环里先把在飞的操作补到 depth 个，再 poll 一次完成队列；
   * 总字节数 = region × iters，带宽 = 总字节数 / 总耗时。深度 1 = 一条做完才发下一条；
   * 深度 8 = 同时有 8 条 64 MiB 的操作在飞。注意先测 WRITE 再测 READ，两者互不重叠。 */
  double ts = run_bw(&ib, IBV_WR_RDMA_WRITE, l_base, r_base, theirs.rkey, region, iters, depth, window);
  double gbs = (double)region * iters / ts / 1e9;
  printf("[client-cpu] 带宽 WRITE：%d × %zu MiB，并发深度 %d：%.2f s -> %.1f GB/s（%.0f Gb/s）\n",
         iters, region_mib, depth, ts, gbs, gbs * 8);
  if (send_all(s, "B", 1) || recv_all(s, &ack, 1)) DIE("OOB 同步失败");   /* 带宽测试后让对端再校验一次 */
  if (ack != 'K') rc = 1;

  ts = run_bw(&ib, IBV_WR_RDMA_READ, l_base + region, r_base + region, theirs.rkey, region, iters, depth, window);
  gbs = (double)region * iters / ts / 1e9;
  printf("[client-cpu] 带宽 READ ：%d × %zu MiB，并发深度 %d：%.2f s -> %.1f GB/s（%.0f Gb/s）\n",
         iters, region_mib, depth, ts, gbs, gbs * 8);
  bad = count_bad(buf + n, n, 5u, 2u);
  printf("[client-cpu] 带宽测试之后本端再校验：不匹配 %zu 个 -> %s\n", bad, bad == 0 ? "PASS" : "FAIL");
  if (bad) rc = 1;

  /* ---- 8. 收尾：先注销 MR，再释放内存 ---------------------------------- */
  send_all(s, "Q", 1);
  close(s);
  /* 销毁顺序：QP → CQ → MR → PD → 设备；MR 必须在内存 free 之前注销，PD 必须在它下面的 QP/MR 都销毁之后才能释放。 */
  ibv_destroy_qp(ib.qp); ibv_destroy_cq(ib.cq); ibv_dereg_mr(ib.mr);
  ibv_dealloc_pd(ib.pd); ibv_close_device(ib.ctx);
  free(buf);
  printf("[client-cpu] 结束，退出码 %d\n", rc);
  return rc;
}

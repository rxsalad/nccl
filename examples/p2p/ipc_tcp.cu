// ============================================================================
// ipc_tcp.cu —— server（GPU7，先启动）与 client（GPU0）：用 TCP 交换"控制信息"，
//               数据经 NVLink 直接读写
//
// 先弄清两条通道，别混：
//   * TCP 只传【控制信息】：IPC 句柄（64 字节）、数据个数 N、完成信号（1 字节）。
//   * 真正的数据【不走 TCP】：client 的 kernel 经 NVLink 直接读写 server 在 GPU7
//     上的显存（方式 A），或者 cudaMemcpy 也可以。
//
// 重要限制：CUDA IPC 句柄只在【同一台机器】（同一个 OS / 容器的 IPC 命名空间）里有效。
//   TCP 在这里只是"传句柄的信道"。如果 client 在另一台机器上，句柄拿过去打不开，
//   那就不是 CUDA IPC 能解决的问题了（要用 NCCL / MPI / RDMA 之类，见文末说明）。
//
// 协议（一次连接，极简）：
//   server → client : Msg { n, handle }          数据已在 GPU7 上就绪
//   client → server : 1 字节 'K'（成功）或 'X'    client 已读写完并关闭映射
//
// 时间线（左 server，右 client；箭头只表示 TCP 上的控制信息）：
//
//   准备 GPU7 数据，取句柄
//   listen  ──────────────┐
//                         │      connect（server 没起来就重试）
//   accept  ◄─────────────┘
//   send {n, 句柄}  ────────────►  recv
//                                  cudaIpcOpenMemHandle ← 在 client 进程里映射 server 的显存
//                                  kernel 经 NVLink 读写（数据不走 TCP）
//                                  cudaDeviceSynchronize → cudaIpcCloseMemHandle
//   recv 'K'/'X'  ◄─────────────  send 1 字节
//   校验数据 → cudaFree
//
// 读代码顺序建议：main → run_server → run_client；send_all/recv_all 是 TCP 的小工具。
//
// 数据布局（server 在 GPU7 上分配 2n 个 unsigned）：
//   [0, n)  ：server 给 client 的数据，值 = i          （server→client）
//   [n, 2n) ：留给 client 写入，client 写 i*3           （client→server）
//
// 编译： nvcc -O3 -arch=native ipc_tcp.cu -o ipc_tcp
// 运行： ./ipc_tcp server 5555                 # 终端 1，先启动
//        ./ipc_tcp client 127.0.0.1 5555       # 终端 2
// ============================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define N_ELEMS ((size_t)1 << 22)            // 每个区域 4M 个 unsigned = 16 MiB

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t e = (call);                                                  \
    if (e != cudaSuccess) {                                                  \
      fprintf(stderr, "[pid %d] FATAL %s:%d  %s: %s\n", getpid(), __FILE__,  \
              __LINE__, #call, cudaGetErrorString(e));                       \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

// ---- TCP 上传的消息：server → client ---------------------------------------
// 整个结构体按原始字节直接收发。能这样做，是因为 server 与 client 在【同一台机器】上，
// 字节序、结构体对齐都一致——而 CUDA IPC 句柄本来也只在同一台机器内有效，所以这里没有
// 做序列化。（若真要跨机器传结构体，必须先约定字节序与格式，不能直接发原始内存。）
struct Msg {
  unsigned long long n;                      // 每个区域的元素个数（让 client 知道大小）
  cudaIpcMemHandle_t handle;                 // 不透明的 64 字节句柄：内容我们不解释，原样搬运
};

// TCP 是【字节流】，不是"消息"：一次 send 可能只发出一部分，一次 recv 也可能只收到一部分
// （消息边界不被保留）。所以固定长度的消息必须循环收发，直到凑够 len 个字节。
// 返回值 k：>0 本次实际收发的字节数；==0 对方已关闭连接（recv 时）；<0 出错。
// 两种异常都当作失败（k <= 0）。注意：向已被对端关闭的连接 send，默认会触发 SIGPIPE
// 终止进程；本程序没有处理（可用 MSG_NOSIGNAL 避免），属于简化。
static int send_all(int fd, const void* buf, size_t len) {
  const char* p = (const char*)buf;
  while (len) { ssize_t k = send(fd, p, len, 0); if (k <= 0) return -1; p += k; len -= k; }
  return 0;
}
static int recv_all(int fd, void* buf, size_t len) {
  char* p = (char*)buf;
  while (len) { ssize_t k = recv(fd, p, len, 0); if (k <= 0) return -1; p += k; len -= k; }
  return 0;
}

// ---- kernel ----------------------------------------------------------------
__global__ void init_kernel(unsigned* p, size_t n) {          // server：前半 = i，后半 = 0
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) { p[i] = (unsigned)i; p[n + i] = 0u; }
}
__global__ void remote_read(const unsigned* __restrict__ remote,
                            unsigned* __restrict__ local, size_t n) {   // client：读对端
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) local[i] = remote[i] + 1;
}
__global__ void remote_write(unsigned* __restrict__ remote, size_t n) { // client：写对端后半区
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) remote[n + i] = (unsigned)i * 3u;
}
static void dims(size_t n, int* grid, int* block) {
  *block = 256;
  size_t g = (n + *block - 1) / *block;
  *grid = (int)(g > 4096 ? 4096 : g);
}

// ============================ server：GPU7 ==================================
static int run_server(int port) {
  // 1. 先把 GPU 上的数据准备好——"listening"之后连进来的 client 立刻就有东西可读
  CHECK(cudaSetDevice(7));
  unsigned* buf;
  CHECK(cudaMalloc(&buf, 2 * N_ELEMS * sizeof(unsigned)));
  int grid, block; dims(N_ELEMS, &grid, &block);
  init_kernel<<<grid, block>>>(buf, N_ELEMS);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());            // 数据必须先就绪，再把句柄交出去
  printf("[server pid %d] GPU7 上 buf = %p，数据已就绪\n", getpid(), (void*)buf);

  Msg msg; msg.n = N_ELEMS;
  CHECK(cudaIpcGetMemHandle(&msg.handle, buf));

  // 2. 建立 TCP 监听（server 先运行 = 先 listen）
  // 标准的 TCP 服务端五步：socket → setsockopt → bind → listen → accept。
  //   socket(AF_INET, SOCK_STREAM, 0)：创建 IPv4 的 TCP 套接字。
  //   SO_REUSEADDR：server 退出后，端口可能短暂处于 TIME_WAIT，不加这个选项重启会
  //     "Address already in use"。
  //   INADDR_ANY（0.0.0.0）：监听本机所有网卡。要注意安全：句柄一旦被别人拿到、
  //     又在同一台机器上，就能映射这块显存，所以真实环境应只绑 127.0.0.1 或加认证。
  //   htons / htonl：把端口、地址转成"网络字节序"（大端），这是套接字 API 的要求。
  //   listen(ls, 1)：开始监听；第二个参数是等待队列长度，这里只有一个 client。
  //     client 在 server 还没 accept 时就 connect，会先排在这个队列里。
  int ls = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);   // 重启后端口可立即复用
  sockaddr_in addr; memset(&addr, 0, sizeof addr);
  addr.sin_family = AF_INET; addr.sin_addr.s_addr = htonl(INADDR_ANY); addr.sin_port = htons(port);
  if (bind(ls, (sockaddr*)&addr, sizeof addr) || listen(ls, 1)) { perror("bind/listen"); return 1; }
  // stdout 重定向到文件时是全缓冲的，不 fflush 的话这条日志要到进程退出才出现，
  // 外面就没法判断"server 是否已经就绪"。
  printf("[server pid %d] 监听 0.0.0.0:%d，等待 client ...\n", getpid(), port);
  fflush(stdout);

  // 两个套接字别混：ls 是"监听套接字"，只负责接受新连接；accept 返回的 cs 才是
  // 和这个 client 通信的"连接套接字"。后面的 send/recv 都用 cs。
  int cs = accept(ls, nullptr, nullptr);     // 阻塞，直到 client 连上
  if (cs < 0) { perror("accept"); return 1; }

  // 3. 通过 TCP 发"控制信息"：个数 + 句柄（数据本身不走 TCP）
  if (send_all(cs, &msg, sizeof msg)) { perror("send"); return 1; }
  printf("[server pid %d] 已发送句柄，等待 client 完成 ...\n", getpid());

  // 4. 等 client 的完成信号（1 字节）
  // 为什么必须等：server 的 buf 还被 client 映射着。client 没关映射之前，server 不能
  // cudaFree；client 没写完之前，server 也不能校验。这 1 个字节就是跨进程的"同步点"——
  // 两个进程的 GPU 工作互不可见，只能靠主机侧的这类信号来协调。
  // 若 client 中途崩溃，这里 recv 会返回 0（连接被关闭），server 走"提前断开"分支退出。
  char done = 0;
  if (recv_all(cs, &done, 1)) { fprintf(stderr, "[server] client 提前断开\n"); return 1; }

  // 5. 校验：后半区应为 i*3（client 写进来的）；前半区应保持 i（client 只读不写）
  // 这里数据量只有 2×16 MiB，直接拷回 host 逐个比较最简单；数据很大时应像 0_to_7.cu
  // 那样在 GPU 上做校验。client 的 'K' 只代表"它读到的数据对"，server 仍要独立验证
  // "client 写进来的数据对"——两个方向各验一次。
  size_t n = N_ELEMS;
  unsigned* h = (unsigned*)malloc(2 * n * sizeof(unsigned));
  CHECK(cudaMemcpy(h, buf, 2 * n * sizeof(unsigned), cudaMemcpyDeviceToHost));
  size_t bad_w = 0, bad_keep = 0;
  for (size_t i = 0; i < n; i++) {
    if (h[n + i] != (unsigned)i * 3u) bad_w++;
    if (h[i] != (unsigned)i) bad_keep++;
  }
  printf("[server pid %d] client→server（client 写进来的数据）：不匹配 %zu 个 -> %s；原数据被改动 %zu 处\n",
         getpid(), bad_w, bad_w == 0 ? "PASS" : "FAIL", bad_keep);
  free(h);
  close(cs); close(ls);                      // 数据已校验完毕，关闭连接和监听套接字
  CHECK(cudaFree(buf));                      // client 已关闭映射，这里才能安全释放
  return (done == 'K' && bad_w == 0 && bad_keep == 0) ? 0 : 1;
}

// ============================ client：GPU0 ==================================
static int run_client(const char* host, int port) {
  // 1. 连 server；server 可能还没起来，所以重试一会儿
  // connect 成功 = 完成了 TCP 三次握手。server 没在监听时会立刻失败（连接被拒绝），
  // 所以用"每次新建套接字、失败就等 100 ms 再试"的方式重试：一个 connect 失败过的
  // 套接字不应复用。server 启动要先做 CUDA 初始化和数据准备，需要一点时间，重试正是为此。
  int s = -1;
  for (int tries = 0; tries < 100; tries++) {
    s = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr; memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET; addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) { fprintf(stderr, "bad host %s\n", host); return 1; }
    if (connect(s, (sockaddr*)&addr, sizeof addr) == 0) break;
    close(s); s = -1; usleep(100 * 1000);    // 100 ms 后重试，最多 10 秒
  }
  if (s < 0) { fprintf(stderr, "[client] 连不上 %s:%d\n", host, port); return 1; }

  // 2. 通过 TCP 收到"控制信息"
  // recv_all 会一直阻塞到收满 sizeof(Msg) 个字节：此时 server 的数据已经就绪
  // （server 是"先准备数据、再 listen、再发句柄"，见 run_server）。
  Msg msg;
  if (recv_all(s, &msg, sizeof msg)) { perror("recv"); return 1; }
  size_t n = (size_t)msg.n;
  printf("[client pid %d] 收到句柄，n = %zu\n", getpid(), n);

  // 3. 选 GPU0，把 server 的显存映射进【本进程】——这一步之前，client 没有任何办法碰那块显存
  // 要点：句柄只是"凭证"，真正建立映射的是 cudaIpcOpenMemHandle，它在【本进程】的
  // CUDA context 里登记这块显存。返回的 remote 是本进程里的有效指针，之后可以像
  // 本地指针一样传给 kernel。按 CUDA 文档，这个调用要在【另一个进程】里对别人分配的
  // 显存使用，不能打开本进程自己的句柄（文档说法，本程序未测）。
  CHECK(cudaSetDevice(0));
  void* remote = nullptr;
  CHECK(cudaIpcOpenMemHandle(&remote, msg.handle, cudaIpcMemLazyEnablePeerAccess));
  printf("[client pid %d] 经 IPC 映射得到 remote = %p\n", getpid(), remote);

  // 4. client 的 kernel 在 GPU0 上运行，经 NVLink 直接读写 server 在 GPU7 的显存
  int grid, block; dims(n, &grid, &block);
  unsigned* local; CHECK(cudaMalloc(&local, n * sizeof(unsigned)));   // GPU0 本地
  remote_read <<<grid, block>>>((const unsigned*)remote, local, n);
  CHECK(cudaGetLastError());
  remote_write<<<grid, block>>>((unsigned*)remote, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());            // 错误通常在这里才暴露；也保证写已完成

  // 5. 校验 server→client：local[i] 应为 i + 1
  unsigned* h = (unsigned*)malloc(n * sizeof(unsigned));
  CHECK(cudaMemcpy(h, local, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (size_t i = 0; i < n; i++) if (h[i] != (unsigned)i + 1u) bad++;
  printf("[client pid %d] server→client（client 读 server 的数据）：不匹配 %zu 个 -> %s\n",
         getpid(), bad, bad == 0 ? "PASS" : "FAIL");
  free(h);
  CHECK(cudaFree(local));

  // 6. 先关映射，再通过 TCP 告诉 server"我完事了"
  // 顺序不能反：如果先通知 server，server 可能在 client 还映射着这块显存时就 cudaFree。
  // done 的含义：'K' = client 读到的数据全部正确；'X' = 有不匹配。
  CHECK(cudaIpcCloseMemHandle(remote));
  char done = (bad == 0) ? 'K' : 'X';
  send_all(s, &done, 1);
  close(s);
  return bad == 0 ? 0 : 1;
}

// 命令行：同一个可执行文件，用第一个参数选择角色。
//   ./ipc_tcp server <port>            监听 port（先启动）
//   ./ipc_tcp client <host> <port>     连接 host:port（后启动；host 用点分十进制 IP）
int main(int argc, char** argv) {
  if (argc >= 3 && !strcmp(argv[1], "server")) return run_server(atoi(argv[2]));
  if (argc >= 4 && !strcmp(argv[1], "client")) return run_client(argv[2], atoi(argv[3]));
  fprintf(stderr, "用法：\n  %s server <port>\n  %s client <host> <port>\n", argv[0], argv[0]);
  return 2;
}

#!/bin/bash
# 多节点 NCCL 基准启动器（dev 机执行；规模自适应：2/4/… 节点）。
# 用法: ./run.sh <nnodes> <op> [bench 参数…，如 --gb 16 --iters 5]
# 例:  ./run.sh 2 all_reduce --gb 16
#      ./run.sh 4 all_reduce --gb 8 --iters 3
set -u
N=${1:?usage: run.sh <nnodes> <op> [args...]}; OP=${2:?}
shift 2
PORT=${PORT:-29500}
PODS=()
for r in $(seq 0 $((N - 1))); do PODS+=("8-gpu-8-fabric-pod-$r"); done
MASTER=$(kubectl get pod "${PODS[0]}" -o jsonpath='{.status.podIP}')

# 标准环境（见 basic-test.md / nccl-2n.md 方案讨论；§1 分析⑥调优实测后 CROSS_NIC 由 0 改 1）：
# CROSS_NIC=1 = 两端各用自己的就近 NIC（发送 a 走 rail a、接收 b 走 rail b，GPU 号不同即 fabric 内
# cross-rail，basic-test §6 的跨 rail 路由被真正用上），broadcast 实测 51 -> 43 ms（+15%）；
# CROSS_NIC=0（NCCL rail-optimized 默认）= 两端同用接收端 GPU 的 rail，发送侧跨 PCIe。
ENV="NCCL_SOCKET_IFNAME=eth0 NCCL_IB_TC=104 NCCL_IB_GID_INDEX=3 NCCL_CROSS_NIC=1 NCCL_NET_DISABLE_INTRA=1 NCCL_PXN_DISABLE=1"
# 覆盖式开关：NCCL_DEBUG=INFO 等直接前置 run.sh 即可（如 NCCL_DEBUG=INFO ./run.sh 2 all_reduce）
export NCCL_DEBUG=${NCCL_DEBUG:-}

LOG=/tmp/mn
mkdir -p "$LOG"
TAG="$OP-$(date +%s)"

# 自动分发 bench.py 到各参与 Pod（幂等；Pod 重建后无需手动 cp）
for p in "${PODS[@]}"; do
  kubectl exec "$p" -c server -- mkdir -p /root/mn 2>/dev/null
  kubectl cp bench.py "$p:/root/mn/" -c server >/dev/null
done

for r in "${!PODS[@]}"; do
  ( kubectl exec "${PODS[$r]}" -c server -- bash -c \
      "cd /root/mn && $ENV NCCL_DEBUG='$NCCL_DEBUG' torchrun --nnodes=$N --nproc_per_node=8 \
        --node-rank=$r --master-addr=$MASTER --master-port=$PORT bench.py --op $OP $*" \
      > "$LOG/$TAG-node$r.log" 2>&1 ) &
done
wait
echo "=== logs: $LOG/$TAG-node*.log ==="
grep -h '^RESULT' "$LOG/$TAG"-node*.log | sort -t'"' -k4 -n
grep -h '^RESULT' "$LOG/$TAG"-node*.log >/dev/null || { echo "NO-RESULT, 末尾日志:"; tail -5 "$LOG/$TAG"-node*.log; }

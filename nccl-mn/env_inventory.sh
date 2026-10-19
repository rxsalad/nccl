#!/bin/bash
# 环境基线清点（Pod 内执行）: GPU / RDMA 设备与网卡 / VRF 对应 / fabric 寻址
FABRICS="fabric0 fabric2 fabric4 fabric6 fabric8 fabric10 fabric12 fabric14"

echo "HOST $(hostname)"
echo "--- GPU ---"
nvidia-smi -L | sed "s/ (UUID.*//"
echo "GPU_COUNT $(nvidia-smi -L | wc -l)"

echo "--- RDMA 设备 (port state/rate) ---"
for d in $(ibv_devinfo -l 2>/dev/null | grep mlx5 | sort); do
  st=$(ibstat "$d" 2>/dev/null | awk '/Port 1/{f=1} f && /State:/{print $2; exit}')
  rate=$(ibstat "$d" 2>/dev/null | awk '/Rate:/{print $2; exit}')
  echo "$d ${st:-?} ${rate:-?}"
done

echo "--- rdma link: 设备 <-> 网卡 ---"
rdma link 2>/dev/null

echo "--- fabric 网卡 IPv6 ---"
for f in $FABRICS; do
  ip -6 addr show "$f" 2>/dev/null | grep "scope global" | awk -v f="$f" '{print f, $2}'
done

echo "--- VRF 列表 ---"
ip -d link show type vrf 2>/dev/null | grep -E "^[0-9]+:"

echo "--- fabric -> VRF 对应 ---"
for f in $FABRICS; do
  m=$(ip -d link show "$f" 2>/dev/null | grep -o "master [a-z0-9]*" | head -1)
  echo "$f -> ${m:-NONE}"
done

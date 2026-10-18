#!/bin/bash
# 通道数扫描（Pod 内执行）: 6 操作 x NCCL_MAX_NCHANNELS {1,2,4,8,16,32(默认)}
set -u
cd /tmp/sn-codes
for f in reduce all_reduce all_gather reduce_scatter all_to_all rs_ag; do
  echo "##### $f"
  for NC in 1 2 4 8 16 32; do
    if [ "$NC" = 32 ]; then
      unset NCCL_MAX_NCHANNELS
    else
      export NCCL_MAX_NCHANNELS=$NC
    fi
    R=$(torchrun --nproc_per_node=8 "$f.py" 2>/dev/null | grep '^\[耗时\]')
    echo "ch=$NC $R"
  done
  unset NCCL_MAX_NCHANNELS
done
echo SCAN-DONE

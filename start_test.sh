#!/bin/bash

rm /workspace/output/* -fr

# 把完整指令存入变量
cmd=(
python /workspace/segment_anything/scripts/amg.py \
    --checkpoint /workspace/models/sam_vit_b_01ec64.pth \
    --model-type vit_b \
    --input /workspace/datasets/sa_8382.jpg \
    --output /workspace/output \
    --use-ort \
    --ort-engine /workspace/models/sam_vit_b_decoder.onnx
)

# 先打印命令
echo "====================================="
echo "即将执行命令："
echo "${cmd[@]}"
echo "====================================="

# 执行命令
"${cmd[@]}"
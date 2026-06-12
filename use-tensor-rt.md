
  # Step 1: 导出 ONNX decoder
  python scripts/export_onnx_model.py \
      --checkpoint models/sam_vit_b_01ec64.pth \
      --model-type vit_b \
      --output sam_vit_b_decoder.onnx

  # Step 2: 用 trtexec 构建 TensorRT engine
  trtexec \
      --onnx=sam_vit_b_decoder.onnx \
      --fp16 \
      --save-engine=sam_vit_b_decoder.engine \
      --workspace=4096 \
      --min-shapes=point_coords:1x1x2,point_labels:1x1,mask_input:1x1x256x256,has_mask_input:1x1,orig_im_size:1x2 \
      --opt-shapes=point_coords:1x64x2,point_labels:1x64,mask_input:1x1x256x256,has_mask_input:1x1,orig_im_size:1x2 \
      --max-shapes=point_coords:1x256x2,point_labels:1x256,mask_input:1x1x256x256,has_mask_input:1x1,orig_im_size:1x2

  # Step 3: 用 amg.py 运行
  python scripts/amg.py \
      --checkpoint models/sam_vit_b_01ec64.pth \
      --model-type vit_b \
      --input datasets/sa_8382.jpg \
      --output output \
      --use-trt \
      --trt-engine sam_vit_b_decoder.engine \
      --points-per-side 8 \
      --points-per-batch 8 \
      --crop-n-layers 0

  ▎ 注意：image_encoder（encoder 部分）仍然走 PyTorch，只有 prompt_encoder + mask_decoder 用 TensorRT 加速。ONNX export 导出的就是这两部分。


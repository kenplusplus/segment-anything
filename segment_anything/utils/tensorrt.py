# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
TensorRT engine runner for SAM's prompt encoder + mask decoder.

Usage:
    1. Export decoder to ONNX:
       python scripts/export_onnx_model.py \\
           --checkpoint models/sam_vit_b_01ec64.pth \\
           --model-type vit_b \\
           --output sam_vit_b_decoder.onnx

    2. Build TensorRT engine (in TRT docker):
       trtexec \\
           --onnx=sam_vit_b_decoder.onnx \\
           --fp16 \\
           --save-engine=sam_vit_b_decoder.engine \\
           --workspace=4096 \\
           --min-shapes=point_coords:1x1x2,point_labels:1x1,mask_input:1x1x256x256,has_mask_input:1x1,orig_im_size:1x2 \\
           --opt-shapes=point_coords:1x64x2,point_labels:1x64,mask_input:1x1x256x256,has_mask_input:1x1,orig_im_size:1x2 \\
           --max-shapes=point_coords:1x256x2,point_labels:1x256,mask_input:1x1x256x256,has_mask_input:1x1,orig_im_size:1x2

    3. Use in amg.py with --use-trt --trt-engine sam_vit_b_decoder.engine
"""

import numpy as np
import torch


class SamTensorRT:
    """Runs SAM's decoder (prompt encoder + mask decoder) via a TensorRT engine."""

    def __init__(self, engine_path: str, max_points: int = 256):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit

        self.cuda = cuda
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Discover I/O tensor names
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(i)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)

        self.max_points = max_points

    def infer(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
        orig_im_size: torch.Tensor,
    ):
        """
        Run TensorRT inference for the decoder.

        Args:
            image_embeddings: (1, embed_dim, H, W) tensor from image_encoder
            point_coords: (1, N, 2) tensor of point coordinates in [0, 1024]
            point_labels: (1, N) tensor of point labels
            mask_input: (1, 1, 256, 256) tensor
            has_mask_input: (1,) tensor (0.0 or 1.0)
            orig_im_size: (2,) tensor [H, W]

        Returns:
            masks: (1, 1, H, W) tensor
            iou_preds: (1, 1) tensor
            low_res_masks: (1, 1, 256, 256) tensor
        """
        import tensorrt as trt

        cuda = self.cuda
        stream = cuda.Stream()

        # Set dynamic shapes for this inference
        num_points = point_coords.shape[1]

        self.context.set_tensor_shape(self.input_names[0], image_embeddings.shape)
        self.context.set_tensor_shape(
            self.input_names[1], (1, num_points, 2)
        )
        self.context.set_tensor_shape(
            self.input_names[2], (1, num_points)
        )
        self.context.set_tensor_shape(self.input_names[3], mask_input.shape)
        self.context.set_tensor_shape(self.input_names[4], (1,))
        self.context.set_tensor_shape(self.input_names[5], (2,))

        # Allocate buffers on demand
        bindings = {}
        for name in self.input_names + self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if dtype == np.float32:
                np_dtype = np.float32
            elif dtype == np.float16:
                np_dtype = np.float16
            elif dtype == np.int64:
                np_dtype = np.int64
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")

            size = int(np.prod(shape))
            gpu_mem = cuda.mem_alloc(size * np.dtype(np_dtype).itemsize)
            bindings[name] = int(gpu_mem)

        # Prepare input numpy data
        image_embeddings_np = image_embeddings.detach().cpu().numpy()
        point_coords_np = point_coords.detach().cpu().numpy()
        point_labels_np = point_labels.detach().cpu().numpy()
        mask_input_np = mask_input.detach().cpu().numpy()
        has_mask_input_np = has_mask_input.detach().cpu().numpy()
        orig_im_size_np = orig_im_size.detach().cpu().numpy()

        inputs = [
            ("image_embeddings", image_embeddings_np),
            ("point_coords", point_coords_np),
            ("point_labels", point_labels_np),
            ("mask_input", mask_input_np),
            ("has_mask_input", has_mask_input_np),
            ("orig_im_size", orig_im_size_np),
        ]

        # Transfer data to GPU
        for name, data in inputs:
            cuda.memcpy_htod(bindings[name], data)

        # Run inference
        success = self.context.execute_async_v2(
            bindings=[bindings[name] for name in self.input_names + self.output_names],
            stream_handle=stream.handle,
        )
        if not success:
            raise RuntimeError("TensorRT inference failed")

        stream.synchronize()

        # Read output
        outputs = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if dtype == np.float32:
                np_dtype = np.float32
            elif dtype == np.float16:
                np_dtype = np.float16
            else:
                np_dtype = np.float32

            output_np = np.empty(shape, dtype=np_dtype)
            cuda.memcpy_dtoh(output_np, bindings[name])
            outputs[name] = torch.from_numpy(output_np)

        return outputs["masks"], outputs["iou_predictions"], outputs["low_res_masks"]


def build_sam_vit_b_trt(checkpoint: str, trt_engine: str):
    """Build SAM with TensorRT decoder support."""
    from segment_anything import build_sam_vit_b

    sam = build_sam_vit_b(checkpoint=checkpoint)
    sam.trt_engine_path = trt_engine
    sam.use_trt = True
    return sam


def build_sam_vit_l_trt(checkpoint: str, trt_engine: str):
    """Build SAM with TensorRT decoder support."""
    from segment_anything import build_sam_vit_l

    sam = build_sam_vit_l(checkpoint=checkpoint)
    sam.trt_engine_path = trt_engine
    sam.use_trt = True
    return sam


def build_sam_vit_h_trt(checkpoint: str, trt_engine: str):
    """Build SAM with TensorRT decoder support."""
    from segment_anything import build_sam_vit_h

    sam = build_sam_vit_h(checkpoint=checkpoint)
    sam.trt_engine_path = trt_engine
    sam.use_trt = True
    return sam
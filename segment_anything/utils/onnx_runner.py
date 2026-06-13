# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
ONNX Runtime GPU runner for SAM's prompt encoder + mask decoder.

Usage:
    1. Export decoder to ONNX:
       python scripts/export_onnx_model.py \
           --checkpoint models/sam_vit_b_01ec64.pth \
           --model-type vit_b \
           --output sam_vit_b_decoder.onnx

    2. Use in amg.py with --use-ort --ort-engine sam_vit_b_decoder.onnx
"""

import numpy as np
import torch


class SamONNXRunner:
    """Runs SAM's decoder (prompt encoder + mask decoder) via ONNX Runtime GPU."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
        self.providers = self.session.get_providers()

        # Discover I/O names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

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
        Run ONNX Runtime inference for the decoder.

        Args:
            image_embeddings: (1, embed_dim, H, W) tensor from image_encoder
            point_coords: (1, N, 2) tensor of point coordinates in [0, 1024]
            point_labels: (1, N) tensor of point labels
            mask_input: (1, 1, 256, 256) tensor
            has_mask_input: (1,) tensor (0.0 or 1.0)
            orig_im_size: (2,) tensor [H, W]

        Returns:
            masks: (1, 4, H, W) tensor — ONNX always returns 4 multimask outputs
            iou_preds: (1, 4) tensor
            low_res_masks: (1, 4, 256, 256) tensor
        """
        feeds = {
            "image_embeddings": image_embeddings.detach().cpu().float().numpy(),
            "point_coords": point_coords.detach().cpu().float().numpy(),
            "point_labels": point_labels.detach().cpu().float().numpy(),
            "mask_input": mask_input.detach().cpu().float().numpy(),
            "has_mask_input": has_mask_input.detach().cpu().float().numpy(),
            "orig_im_size": orig_im_size.detach().cpu().float().numpy(),
        }

        outs = self.session.run(self.output_names, feeds)

        # ONNX returns (1, 4, H, W) and (1, 4) — always 4 multimask outputs
        masks = torch.from_numpy(outs[0])    # (1, 4, H, W)
        iou_preds = torch.from_numpy(outs[1])  # (1, 4)
        low_res_masks = torch.from_numpy(outs[2])  # (1, 4, 256, 256)

        return masks, iou_preds, low_res_masks
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
ONNX Runtime GPU runner for SAM's prompt encoder + mask decoder.

Optimizations:
- Pre-allocated constant buffers (mask_input, has_mask_input) — the runner
  reuses numpy arrays across calls instead of building fresh Python objects.
- The runner accepts a batched (1, N, 2) point set. The calling code is
  responsible for the points_per_batch chunking; this runner does NOT
  loop point-by-point. (That per-point loop was a critical perf bug in
  the original implementation — it caused 64 sequential ORT calls per
  AMG batch.)

NOTE: This model was exported with fixed batch dim B=1, so a single
inference returns 4 multimask outputs for ONE point prompt. Callers
that want to do real per-point batched inference (B=N) should re-export
the decoder with a dynamic batch dim, then call infer() once with the
whole batch of points.
"""

import numpy as np
import torch


class SamONNXRunner:
    """Runs SAM's decoder (prompt encoder + mask decoder) via ONNX Runtime GPU."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=providers
        )
        self.providers = self.session.get_providers()

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        # Pre-allocated constant "no prior mask" inputs (used for every call
        # in automatic mask generation).
        self._mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        self._has_mask_input = np.zeros((1,), dtype=np.float32)

        # Warmup: do a single dummy inference so ORT's CUDA kernels, memory
        # pools, and the first JIT compilation happen at construction time
        # (during "Loading model...") rather than burning 1-2 s on the first
        # real batch.  This is invisible to the user and ~free.
        try:
            import torch
            device = "cuda" if "CUDAExecutionProvider" in self.providers else "cpu"
            dummy_emb = torch.zeros(1, 256, 64, 64, dtype=torch.float, device=device)
            dummy_pt = torch.zeros(1, 1, 2, dtype=torch.float, device=device)
            dummy_lbl = torch.ones(1, 1, dtype=torch.float, device=device)
            dummy_ois = torch.tensor([512.0, 512.0], dtype=torch.float, device=device)
            self.infer(dummy_emb, dummy_pt, dummy_lbl, None, None, dummy_ois)
            torch.cuda.synchronize()
        except Exception as e:
            # Warmup is a perf hint, not a correctness requirement.
            print(f"[SamONNXRunner] warmup skipped: {e}")

    def infer(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor = None,
        has_mask_input: torch.Tensor = None,
        orig_im_size: torch.Tensor = None,
    ):
        """
        Run ONNX Runtime inference for the decoder.

        Args:
            image_embeddings: (1, embed_dim, H, W) GPU tensor.
            point_coords: (1, N, 2) GPU tensor of point coordinates in [0, 1024].
            point_labels: (1, N) GPU tensor of point labels.
            mask_input / has_mask_input / orig_im_size: optional overrides.

        Returns:
            masks: (1, 4, H, W) CPU tensor (numpy-backed torch).
            iou_preds: (1, 4) CPU tensor.
            low_res_masks: (1, 4, 256, 256) CPU tensor.
        """
        # Resolve the small constant inputs. Reuse pre-allocated numpy
        # buffers when the caller did not override them — this is the hot
        # path in AMG and avoids per-call allocations.
        if mask_input is None:
            mask_input_np = self._mask_input
        else:
            mask_input_np = mask_input.detach().contiguous().cpu().float().numpy()

        if has_mask_input is None:
            has_mask_input_np = self._has_mask_input
        else:
            has_mask_input_np = has_mask_input.detach().contiguous().cpu().float().numpy()

        if orig_im_size is None:
            raise ValueError("orig_im_size must be provided")
        orig_im_size_np = orig_im_size.detach().contiguous().cpu().float().numpy()

        # The dynamic inputs (image_embeddings, point_coords, point_labels)
        # still need to cross the host/device boundary. The .detach() calls
        # ensure we don't accidentally retain autograd graphs.
        image_embeddings_np = image_embeddings.detach().contiguous().cpu().float().numpy()
        point_coords_np = point_coords.detach().contiguous().cpu().float().numpy()
        point_labels_np = point_labels.detach().contiguous().cpu().float().numpy()

        feeds = {
            "image_embeddings": image_embeddings_np,
            "point_coords": point_coords_np,
            "point_labels": point_labels_np,
            "mask_input": mask_input_np,
            "has_mask_input": has_mask_input_np,
            "orig_im_size": orig_im_size_np,
        }

        outs = self.session.run(self.output_names, feeds)
        return (
            torch.from_numpy(outs[0]),
            torch.from_numpy(outs[1]),
            torch.from_numpy(outs[2]),
        )

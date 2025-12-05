# utils/gradcam.py

import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        """
        model        : the S-MAD model (classifier for bonafide vs morph)
        target_layer : the last conv layer module you want to visualize
        """
        self.model = model
        self.target_layer = target_layer

        self.gradients = None      # d(output) / d(features)
        self.activations = None    # feature maps from target layer

        # ---- Register hooks on the target conv layer ----
        self._forward_hook = target_layer.register_forward_hook(
            self._forward_hook_fn
        )
        self._backward_hook = target_layer.register_backward_hook(
            self._backward_hook_fn
        )

    # ---------- Hooks (now they use self correctly) ----------
    def _forward_hook_fn(self, module, input, output):
        # output shape: [B, C, H, W]
        self.activations = output

    def _backward_hook_fn(self, module, grad_input, grad_output):
        # grad_output[0] has shape [B, C, H, W]
        self.gradients = grad_output[0]

    # ---------- Main function ----------
    def generate(self, input_tensor, class_idx):
        """
        input_tensor : shape [1, 3, H, W], requires_grad=True
        class_idx    : int, e.g. 0 = bonafide, 1 = morph
        Returns: numpy 2D CAM map resized to input spatial size
        """
        # Make sure gradients are enabled
        input_tensor = input_tensor.requires_grad_(True)

        # Put model in eval mode
        self.model.eval()
        self.model.zero_grad()

        # Forward pass through S-MAD model
        output = self.model(input_tensor)

        # In case model returns (logits, something_else)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        # Pick the logit for the target class
        # logits shape: [B, num_classes]
        target = logits[:, class_idx].sum()

        # Backward to get gradients wrt target_layer feature maps
        target.backward()

        # Get stored activations and gradients
        gradients = self.gradients          # [B, C, H, W]
        activations = self.activations      # [B, C, H, W]

        # Global average pooling over H, W → weights per channel
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        # Weighted sum of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)

        # Upsample CAM to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],  # (H, W)
            mode="bilinear",
            align_corners=False,
        )

        # Normalize 0–1
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Return a 2D numpy map
        cam_np = cam.detach().cpu().numpy()[0, 0]
        return cam_np

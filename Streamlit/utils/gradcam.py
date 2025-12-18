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
        self._backward_hook = target_layer.register_full_backward_hook(
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
        input_tensor = input_tensor.requires_grad_(True)
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        logits = output[0] if isinstance(output, tuple) else output
        target = logits[:, class_idx].sum()
        target.backward()

        gradients = self.gradients           # [B,C,H,W]
        activations = self.activations       # [B,C,H,W]

        weights = gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # ✅ Paper-style normalization
        cam = cam - cam.min()
        cam = cam / (torch.quantile(cam, 0.99) + 1e-8)
        cam = cam.clamp(0, 1)

        # ✅ Smoothing
        cam = F.avg_pool2d(cam, kernel_size=3, stride=1, padding=1)

        return cam.detach().cpu().numpy()[0,0]


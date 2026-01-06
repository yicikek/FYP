import torch
import torch.nn.functional as F
import numpy as np

class GradCAMClassifier:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hooks
        target_layer.register_forward_hook(self._save_activation)
        # Use register_full_backward_hook for newer PyTorch versions
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_id, input_selfie, class_idx=1):
        """
        input_id: Tensor for ID image
        input_selfie: Tensor for Selfie image
        class_idx: 1 for Morph, 0 for Bona-fide
        """
        self.model.zero_grad()

        # 1. Forward pass with BOTH images
        logits, _ = self.model(input_id, input_selfie) 

        # 2. Safety check for class_idx to prevent IndexError
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
        
        # Ensure it's a standard Python integer
        if torch.is_tensor(class_idx):
            class_idx = class_idx.item()
        class_idx = int(class_idx)

        # 3. Backward pass on the specific class score
        # Using retain_graph=True if you plan to call it multiple times
        logits[0, class_idx].backward(retain_graph=True)

        # 4. Grad-CAM Calculation
        # Global Average Pooling (GAP) over gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        # Normalize 0–1
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam[0].cpu().numpy()
import torch
import torch.nn.functional as F

class GradCAMClassifier:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # forward hook → get activations
        target_layer.register_forward_hook(self._save_activation)

        # backward hook → get gradients
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor, class_idx=None):
        self.model.zero_grad()

        # forward pass through DMAD
        logits, _ = self.model(img_tensor, img_tensor)   # DMAD forward returns (logits, cosine)

        if class_idx is None:
            class_idx = torch.argmax(logits).item()

        logits[0, class_idx].backward()

        # GAP over gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # weighted sum of activations
        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)

        # normalize 0–1
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam[0].cpu().numpy()

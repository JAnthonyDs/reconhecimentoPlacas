import torch
import torch.nn as nn
import torch.nn.functional as F

class WPODLoss(nn.Module):
    def __init__(self, alpha=7.75, ns=24, gamma_obj=0.3):
        super(WPODLoss, self).__init__()
        self.alpha = alpha
        self.ns = ns
        self.gamma_obj = gamma_obj

    def forward(self, pred, target, object_mask):
        batch_size, _, H, W = pred.shape
        M, N = H, W  # Feature map dimensions

        # Separate the predicted values
        v1 = pred[:, 0, :, :]  # Object probabilities
        v2 = pred[:, 1, :, :]  # Non-object probabilities
        v3 = torch.max(pred[:, 2, :, :], torch.tensor(0.0, device=pred.device))
        v4 = pred[:, 3, :, :]
        v5 = pred[:, 4, :, :]
        v6 = torch.max(pred[:, 5, :, :], torch.tensor(0.0, device=pred.device))
        v7 = pred[:, 6, :, :]
        v8 = pred[:, 7, :, :]

        # Define the canonical square vertices
        q = torch.tensor([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], device=pred.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 1, 4, 2)
        
        # Compute Tmn(q)
        Tmn_q = torch.zeros((batch_size, M, N, 4, 2), device=pred.device)
        for i in range(4):
            Tmn_q[..., i, 0] = v3 * q[..., i, 0] + v4 * q[..., i, 1] + v7
            Tmn_q[..., i, 1] = v5 * q[..., i, 0] + v6 * q[..., i, 1] + v8

        # Normalize the annotated points pi
        A_p = torch.zeros_like(target)
        A_p[..., 0] = (1 / self.alpha) * (1 / self.ns) * target[..., 0] - torch.arange(N, device=pred.device).float().unsqueeze(0).unsqueeze(-1) * (1 / self.alpha)
        A_p[..., 1] = (1 / self.alpha) * (1 / self.ns) * target[..., 1] - torch.arange(M, device=pred.device).float().unsqueeze(1).unsqueeze(-1) * (1 / self.alpha)
        
        # Compute faffine(m, n)
        f_affine = torch.sum(torch.abs(Tmn_q - A_p), dim=[3, 4])

        # Compute fprobs(m, n)
        logloss = lambda y, p: -y * torch.log(p + 1e-10) - (1 - y) * torch.log(1 - p + 1e-10)
        f_probs = logloss(object_mask, v1) + logloss(1 - object_mask, v2)
        
        # Combine both parts of the loss
        loss = torch.sum(object_mask * f_affine + f_probs)

        return loss

# Example usage
# pred: Tensor of shape (batch_size, 8, M, N) - predicted output from the network
# target: Tensor of shape (batch_size, M, N, 4, 2) - ground truth corner points
# object_mask: Tensor of shape (batch_size, M, N) - mask indicating presence of object

# Instantiate the loss function
loss_fn = WPODLoss()

# Compute the loss
loss = loss_fn(pred, target, object_mask)

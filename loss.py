from torch import nn

class TripletCosineLoss(nn.Module):
    def __init__(self, margin=1.0, eps=1e-6, reduction="mean"):
        super(TripletCosineLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.cos = nn.CosineSimilarity(dim=2, eps=eps)

    def forward(self, anchor, positive, negative):
        d_p = 1 - self.cos(anchor, positive)
        d_n = 1 - self.cos(anchor, negative)
        losses = torch.clamp(d_p - d_n + self.margin, min=0.0)

        if self.reduction == "mean":
            return losses.mean(), d_p.mean(), d_n.mean()
        if self.reduction == "sum":
            return losses.sum(), d_p.sum(), d_n.sum()
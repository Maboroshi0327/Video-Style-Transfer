import torch
from torch import linalg as LA


def global_stylized_loss(fcs, fs, loss_fn):
    # Mean distance
    fcs_mean = fcs.mean(dim=(2, 3))
    fs_mean = fs.mean(dim=(2, 3))
    mean_dist = loss_fn(fcs_mean, fs_mean)

    # Standard deviation distance
    fcs_std = fcs.std(dim=(2, 3))
    fs_std = fs.std(dim=(2, 3))
    std_dist = loss_fn(fcs_std, fs_std)

    # Loss for each ReLU_x_1 layer
    return mean_dist + std_dist


def local_feature_loss(fcs, adaattn, loss_fn):
    dist = loss_fn(fcs, adaattn)
    return dist


def cosine_distance(fu, fv):
    """
    fu:   (b, c, h, w)
    fv:   (b, c, h, w)
    out: (b, c, c)
    """
    b, c, _, _ = fu.size()
    fu = fu.view(b, c, -1)
    fv = fv.view(b, c, -1).permute(0, 2, 1)
    fu_norm = LA.vector_norm(fu, dim=-1, keepdim=True)
    fv_norm = LA.vector_norm(fv, dim=1, keepdim=True)
    d = torch.bmm(fu, fv) / (torch.bmm(fu_norm, fv_norm) + 1e-6)
    d = 1 - d
    return d


def image_similarity_loss(fc1, fc2, fcs1, fcs2):
    _, _, h, w = fc1.size()
    n = h * w

    Dc1c2 = cosine_distance(fc1, fc2)
    Dcs1cs2 = cosine_distance(fcs1, fcs2)

    Dc1c2 = Dc1c2 / Dc1c2.sum(dim=1, keepdim=True)
    Dcs1cs2 = Dcs1cs2 / Dcs1cs2.sum(dim=1, keepdim=True)

    loss = torch.abs(Dc1c2 - Dcs1cs2).sum()
    loss = loss / n
    return loss

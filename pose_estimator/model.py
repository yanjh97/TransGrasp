import torch
import torch.nn as nn


class DeformNet(nn.Module):
    def __init__(self, n_cat=6, nv_prior=1024):
        super(DeformNet, self).__init__()
        self.n_cat = n_cat

        self.instance_geometry = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.instance_global = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.category_local = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.category_global = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.assignment = nn.Sequential(
            nn.Conv1d(2112, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, nv_prior, 1),
        )
        self.deformation = nn.Sequential(
            nn.Conv1d(2112, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 3, 1),
        )
        # Initialize weights to be small so initial deformations aren't so big
        self.deformation[4].weight.data.normal_(0, 0.0001)

    def forward(self, points, prior):
        """
        Args:
            points: bs x n_pts x 3
            prior: bs x nv x 3

        Returns:
            assign_mat: bs x n_pts x nv
            inst_shape: bs x nv x 3
            deltas: bs x nv x 3
            log_assign: bs x n_pts x nv, for numerical stability

        """
        bs, n_pts = points.size()[:2]
        nv = prior.size()[1]
        # instance-specific features
        points = points.permute(0, 2, 1)
        inst_local = self.instance_geometry(points)
        inst_global = self.instance_global(inst_local)    # bs x 1024 x 1
        # category-specific features
        cat_prior = prior.permute(0, 2, 1)
        cat_local = self.category_local(cat_prior)    # bs x 64 x n_pts
        cat_global = self.category_global(cat_local)  # bs x 1024 x 1
        # assignemnt matrix
        assign_feat = torch.cat((inst_local, inst_global.repeat(1, 1, n_pts), cat_global.repeat(1, 1, n_pts)), dim=1)   # bs x 2112 x n_pts
        assign_mat = self.assignment(assign_feat)
        assign_mat = assign_mat.permute(0, 2, 1).contiguous()    # bs x n_pts x nv
        # deformation field
        deform_feat = torch.cat((cat_local, cat_global.repeat(1, 1, nv), inst_global.repeat(1, 1, nv)), dim=1)   # bs x 2112 x n_pts
        deltas = self.deformation(deform_feat)
        deltas = deltas.permute(0, 2, 1).contiguous()   # bs x nv x 3

        return assign_mat, deltas

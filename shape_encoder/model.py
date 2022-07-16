import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self,  encoder_feat, global_feat=True, coord_transform=False, feature_transform = False, bn=False):
        super(PointNetfeat, self).__init__()
        self.coord_transform = coord_transform
        self.encoder_feat = encoder_feat
        if self.coord_transform:
            self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, encoder_feat[0], 1)
        self.conv2 = torch.nn.Conv1d(encoder_feat[0], encoder_feat[1], 1)
        self.conv3 = torch.nn.Conv1d(encoder_feat[1], encoder_feat[2], 1)
        self.bn1 = nn.BatchNorm1d(encoder_feat[0])
        self.bn2 = nn.BatchNorm1d(encoder_feat[1])
        self.bn3 = nn.BatchNorm1d(encoder_feat[2])
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        self.bn = bn


    def forward(self, x):
        n_pts = x.size()[2]

        if self.coord_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        else:
            trans = None

        if self.bn:
            x = F.relu(self.bn1(self.conv1(x)))
        else:
            x = F.relu(self.conv1(x))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x

        if self.bn:
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
        else:
            x = F.relu(self.conv2(x))
            x = self.conv3(x)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.encoder_feat[2])
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.encoder_feat[2], 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class ShapeEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, bn=False):
        super(ShapeEncoder, self).__init__()
        self.feature_transform = feature_transform

        shape_encoder_feat = [64, 128, 1024]
        fuse_feat = [1024, 1024]
        shape_decoder_feat = [512, 256, 128]

        self.shape_encoder_feat = shape_encoder_feat
        self.fuse_feat = fuse_feat
        self.shape_decoder_feat = shape_decoder_feat

        self.feat_shape = PointNetfeat(shape_encoder_feat, global_feat=global_feat, coord_transform=False, feature_transform=feature_transform, bn=True)

        self.dropout = nn.Dropout(p=0.3)
        self.n_pts = 1024
        self.global_feat = global_feat
        
        if not global_feat:
            self.conv0 = torch.nn.Conv1d(shape_encoder_feat[-1] + shape_encoder_feat[0], fuse_feat[0], 1)
            self.conv1 = torch.nn.Conv1d(fuse_feat[0], fuse_feat[1], 1)

        encoder_output_feat = shape_encoder_feat[-1] if global_feat else fuse_feat[-1]
        
        self.fc0_code = torch.nn.Linear(encoder_output_feat, shape_decoder_feat[0])
        self.fc1_code = torch.nn.Linear(shape_decoder_feat[0], shape_decoder_feat[1])
        self.fc2_code = torch.nn.Linear(shape_decoder_feat[1], shape_decoder_feat[2])

        self.fc3_code = torch.nn.Linear(shape_decoder_feat[2], 128)

        self.bn = bn
        if bn:
            self.bn_0_code = nn.BatchNorm1d(shape_decoder_feat[0])
            self.bn_1_code = nn.BatchNorm1d(shape_decoder_feat[1])
            self.bn_2_code = nn.BatchNorm1d(shape_decoder_feat[2])

    def forward(self, points):
        
        x_2 = points
        x_2 = x_2.transpose(2, 1).contiguous()

        f_shape,_,_ = self.feat_shape(x_2)

        if not self.global_feat:
            f_shape = self.conv0(f_shape)
            f_shape = self.conv1(f_shape)
            f_shape = torch.max(f_shape, 2, keepdim=True)[0]
            f_shape = f_shape.view(-1, self.fuse_feat[-1])

        if self.bn:
            code_x = F.leaky_relu(self.bn_0_code(self.fc0_code(f_shape)))
            code_x = F.leaky_relu(self.bn_1_code(self.fc1_code(code_x)))
            code_x = F.leaky_relu(self.bn_2_code(self.fc2_code(code_x)))

        else:
            code_x = F.leaky_relu(self.fc0_code(f_shape))
            code_x = F.leaky_relu(self.fc1_code(code_x))
            code_x = F.leaky_relu(self.fc2_code(code_x))

        code_x = self.fc3_code(code_x)
        out_code = code_x.contiguous()
        
        return out_code
        
def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,2500,3))

    encoder = ShapeEncoder()
    out = encoder(sim_data)
    print('ShapeEncoder out size: ', out.size())
import argparse
import random, os, sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

sys.path.append('./')
from pose_estimator.data import PartPointsDatset
from pose_estimator.model import DeformNet
from pose_estimator.loss import Loss

parser = argparse.ArgumentParser(description='Pose Network')
parser.add_argument('--data_root', type=str, default='datasets')
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--category', type=str, default='mug')
parser.add_argument('--num_points', type=int, default=1024)
parser.add_argument('--model_points', type=int, default=1024)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_rate', type=float, default=0.35)
parser.add_argument('--nepoch', type=int, default=50)
parser.add_argument('--resume_posenet', type=str, default='')
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--out_dir', type=str, default='pose_estimator/output/')
parser.add_argument('--logger_freq', type=int, default=10)
parser.add_argument("--model_name", default="mug_test")
opt = parser.parse_args()

opt.out_dir = opt.out_dir + opt.model_name
opt.corr_wt = 1.0
opt.cd_wt = 5.0
opt.entropy_wt = 0.0001
opt.deform_wt = 0.01

def train():
    mean_shapes = np.load('pose_estimator/assets/mean_points_emb.npy')
    CLASS_MAP_FOR_CATEGORY = {'bottle':1, 'bowl':2, 'camera':3, 'can':4, 'laptop':5, 'mug':6}
    prior = mean_shapes[CLASS_MAP_FOR_CATEGORY[opt.category]-1]
    prior = torch.from_numpy(prior).float().cuda()

    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    print(opt)
    
    writer = SummaryWriter(os.path.join(opt.out_dir, 'summaries'))

    net = DeformNet(n_cat=1, nv_prior=1024)
    net = nn.DataParallel(net)
    net.cuda()

    loss_module = Loss(opt.corr_wt, opt.cd_wt, opt.entropy_wt, opt.deform_wt)

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    dataset = PartPointsDatset(opt.data_root, opt.category, 'train', opt.num_points, opt.model_points)
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    val_dataset = PartPointsDatset(opt.data_root, opt.category, 'val', opt.num_points, opt.model_points)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    train_count = 0
    best_pose = np.Inf

    with tqdm(total=opt.nepoch * (len(dataloader) + len(val_dataloader))) as pbar:
        for epoch in range(opt.start_epoch, opt.nepoch+1):
            
            writer.add_scalar("train/lr", optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            net.train()
            optimizer.zero_grad()
            
            total_loss_sum = 0.0
            corr_loss_sum = 0.0
            cd_loss_sum = 0.0
            entropy_loss_sum = 0.0
            deform_loss_sum = 0.0

            for i, (cam_pcs, gt_labels, _, _) in enumerate(dataloader):
                pbar.update(1)
                cuda_cam_pcs = cam_pcs.cuda()
                cuda_gt_nocs = gt_labels['nocs'].cuda()
                cuda_model_points = gt_labels['model_points'].cuda()
                cuda_prior = prior.view(1, 1024, 3).repeat(cuda_cam_pcs.size()[0], 1, 1)

                assign_mat, deltas = net(cuda_cam_pcs, cuda_prior)
                loss, corr_loss, cd_loss, entropy_loss, deform_loss = loss_module(assign_mat, deltas, prior, cuda_gt_nocs, cuda_model_points)
                
                total_loss_sum += loss.item()
                corr_loss_sum += corr_loss.item()
                cd_loss_sum += cd_loss.item()
                entropy_loss_sum += entropy_loss.item()
                deform_loss_sum += deform_loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_count += 1

                if train_count % opt.logger_freq == 0:
                    tqdm.write('[epoch{:03d}] total_loss:{:.4f}, corr loss:{:.4f}, cd loss:{:.4f} entropy loss:{:.4f}, deform_loss:{:.4f}'.format(
                        epoch,
                        total_loss_sum / opt.logger_freq,
                        corr_loss_sum / opt.logger_freq,
                        cd_loss_sum / opt.logger_freq,
                        entropy_loss_sum / opt.logger_freq,
                        deform_loss_sum / opt.logger_freq,
                    ))
                    writer.add_scalar("train/total_loss", total_loss_sum / opt.logger_freq, train_count)
                    writer.add_scalar("train/corr_loss", corr_loss_sum / opt.logger_freq, train_count)
                    writer.add_scalar("train/cd_loss", cd_loss_sum / opt.logger_freq, train_count)
                    writer.add_scalar("train/entropy_loss", entropy_loss_sum / opt.logger_freq, train_count)
                    writer.add_scalar("train/deform_loss", deform_loss_sum / opt.logger_freq, train_count)

                total_loss_sum = 0.0
                corr_loss_sum = 0.0
                cd_loss_sum = 0.0
                entropy_loss_sum = 0.0
                deform_loss_sum = 0.0

            net.eval()
            test_count = 0

            test_total_loss_sum = 0.0
            test_corr_loss_sum = 0.0
            test_cd_loss_sum = 0.0
            test_entropy_loss_sum = 0.0
            test_deform_loss_sum = 0.0

            for i, (cam_pcs, gt_labels, _, _) in enumerate(val_dataloader, 0):
                pbar.update(1)
                cuda_cam_pcs = cam_pcs.cuda()
                cuda_gt_nocs = gt_labels['nocs'].cuda()
                cuda_model_points = gt_labels['model_points'].cuda()
                cuda_prior = prior.view(1, 1024, 3).repeat(cuda_cam_pcs.size()[0], 1, 1)

                assign_mat, deltas = net(cuda_cam_pcs, cuda_prior)
                loss, corr_loss, cd_loss, entropy_loss, deform_loss = loss_module(assign_mat, deltas, prior, cuda_gt_nocs, cuda_model_points)
                
                test_total_loss_sum += loss.item()
                test_corr_loss_sum += corr_loss.item()
                test_cd_loss_sum += cd_loss.item()
                test_entropy_loss_sum += entropy_loss.item()
                test_deform_loss_sum += deform_loss.item()

                test_count += 1

            writer.add_scalar("test/total_loss", test_total_loss_sum / test_count, train_count)
            writer.add_scalar("test/corr_loss", test_corr_loss_sum / test_count, train_count)
            writer.add_scalar("test/cd_loss", test_cd_loss_sum / test_count, train_count)
            writer.add_scalar("test/entropy_loss", test_entropy_loss_sum / test_count, train_count)
            writer.add_scalar("test/deform_loss", test_deform_loss_sum / test_count, train_count)

            if test_total_loss_sum <= best_pose:
                best_pose = test_total_loss_sum
                torch.save(net.module.state_dict(), '{0}/checkpoints/model_{1}.pth'.format(opt.out_dir, epoch))
                tqdm.write('>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

            torch.save(net.module.state_dict(), '{0}/checkpoints/model_current.pth'.format(opt.out_dir))

            scheduler.step()

if __name__ == "__main__":
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)
    if not os.path.exists(os.path.join(opt.out_dir, 'summaries')):
        os.makedirs(os.path.join(opt.out_dir, 'summaries'))
    if not os.path.exists(os.path.join(opt.out_dir, 'checkpoints')):
        os.makedirs(os.path.join(opt.out_dir, 'checkpoints'))   
    train()
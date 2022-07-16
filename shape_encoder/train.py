import argparse
import random, time, os, yaml, sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

sys.path.append('./')
sys.path.append('./shape_encoder')
from shape_encoder.data import PartPointsDatset
from shape_encoder.model import ShapeEncoder


parser = argparse.ArgumentParser(description='Shape Encoder')
parser.add_argument('--data_root', type=str, default = 'datasets')
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--category', type=str, default='')
parser.add_argument('--num_points', type=int, default=1024)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_rate', default=0.35)
parser.add_argument('--nepoch', type=int, default=30)
parser.add_argument('--resume_posenet', type=str, default='')
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--out_dir', type=str, default='shape_encoder/output/')
parser.add_argument('--logger_freq', type=int, default=10)
parser.add_argument("--model_name", default="test")
opt = parser.parse_args()

opt.out_dir = os.path.join(opt.out_dir, opt.model_name)

def train():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    print(opt)

    writer = SummaryWriter(os.path.join(opt.out_dir, 'summaries'))

    net = ShapeEncoder()
    net = nn.DataParallel(net)
    net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    dataset = PartPointsDatset(opt.data_root, opt.category, 'train', opt.num_points)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    val_dataset = PartPointsDatset(opt.data_root, opt.category, 'val', opt.num_points)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    st_time = time.time()

    train_count = 0
    best_code = np.Inf

    with tqdm(total=opt.nepoch * (len(dataloader) + len(val_dataloader))) as pbar:
        for epoch in range(opt.start_epoch, opt.nepoch+1):
            
            writer.add_scalar("train/lr", optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            net.train()
            optimizer.zero_grad()

            loss_code_sum = 0.0
            total_loss_sum = 0.0

            for i, (obj_pcs, gt_codes, _, _) in enumerate(dataloader):
                
                pbar.update(1)
                cuda_obj_pcs = obj_pcs.cuda()
                cuda_gt_codes = gt_codes.cuda()
                cuda_pred_codes = net(cuda_obj_pcs)
                loss_code = torch.mean(torch.abs(cuda_pred_codes - cuda_gt_codes))
                loss_code_sum += loss_code.item()
                total_loss = 1e2 * loss_code
                total_loss_sum += total_loss.item()
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_count += 1

                if train_count % opt.logger_freq == 0:
                    tqdm.write('epoch {:03d} loss-code {:.7f}'.format(
                        epoch,
                        total_loss_sum / opt.logger_freq,
                        ))

                    writer.add_scalar("train/loss_code", loss_code_sum / opt.logger_freq, train_count)

                loss_code_sum = 0.0
                total_loss_sum = 0.0

            net.eval()
            test_count = 0
            test_loss_code_sum = 0.0

            for i, (obj_pcs, gt_codes, _, _) in enumerate(val_dataloader, 0):

                pbar.update(1)
                cuda_obj_pcs = obj_pcs.cuda()
                cuda_gt_codes = gt_codes.cuda()
                cuda_pred_codes = net(cuda_obj_pcs)
                loss_code = torch.mean(torch.abs(cuda_pred_codes - cuda_gt_codes), dim=1)
                test_loss_code_sum += loss_code.item()
                test_count += 1

            test_loss_code_sum = test_loss_code_sum / test_count
            writer.add_scalar("test/loss_code", test_loss_code_sum, epoch)
            if test_loss_code_sum <= best_code:
                best_code = test_loss_code_sum
                torch.save(net.module.state_dict(), '{0}/checkpoints/model_best.pth'.format(opt.out_dir, epoch, test_loss_code_sum))
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
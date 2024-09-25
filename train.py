import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import gc
from dataset.tifdataload import tifdataset
import torch.nn.functional as tnf
from model import esrt
from model import hat
# from model import architecture, esrt
from data import DIV2K, Set5_val
import utils
import skimage.color as sc
import random
from collections import OrderedDict
import datetime
import warnings
import model.cmtHfeaturesV16

warnings.filterwarnings('ignore')
from importlib import import_module

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def train(epoch, path, scale):
    model.train()
    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])

    loss_total = 0

    for iteration, (lr_tensor, hr_tensor, lr_name, hr_name) in enumerate(training_data_loader, 1):

        if args.cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]

        optimizer.zero_grad()
        sr_tensor = model(lr_tensor)
        # sr_tensor = forward_chop(model, lr_tensor,2)

        loss_l1 = l1_criterion(sr_tensor, hr_tensor)
        loss_sr = loss_l1
        loss_sr.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss_l1: {:.5f}".format(epoch, iteration, len(training_data_loader),
                                                                  loss_l1.item()))
            with open(path, 'a') as file:
                file.write("===> Epoch[{}]({}/{}): Loss_l1: {:.5f}".format(epoch, iteration, len(training_data_loader),
                                                                           loss_l1.item()) + '\n')

        loss_total = loss_l1.item() + loss_total

    print("===> Epoch[{}]: Loss_l1: {:.5f}".format(epoch, loss_total/iteration))

    with open(path, 'a') as file:
        file.write("===> Epoch[{}]: Loss_l1: {:.5f}".format(epoch, loss_total/iteration) + '\n')


def forward_chop(model, x, scale, shave=6, min_size=10000):

    n_GPUs = 1  # min(self.n_GPUs, 4)
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2  # 取图像尺寸一半
    h_size, w_size = h_half + shave, w_half + shave  # 重叠10个像素shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]  # 分为4块

    if w_size * h_size < min_size:  # 如果每块面积小于min_size
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch = model(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(model, patch, scale=scale, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
    return output


def valid(scale, path):
    model.eval()
    avg_psnr, avg_ssim = 0, 0
    with torch.no_grad():  ###插在此处

        for batch in testing_data_loader:
            lr_tensor, hr_tensor = batch[0], batch[1]
            if args.cuda:
                lr_tensor = lr_tensor.to(device)
                hr_tensor = hr_tensor.to(device)

            with torch.no_grad():
                # pre = model(lr_tensor)
                pre = forward_chop(model, lr_tensor, 2)

            sr_img = utils.tensor2np(pre.detach()[0])
            gt_img = utils.tensor2np(hr_tensor.detach()[0])

            im_label = gt_img
            im_pre = sr_img

            avg_psnr += utils.compute_psnr(im_pre, im_label)
            avg_ssim += utils.compute_ssim(im_pre, im_label)
        print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr / len(testing_data_loader),
                                                              avg_ssim / len(testing_data_loader)))
        with open(path, 'a') as file:
            file.write("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr / len(testing_data_loader),
                                                                       avg_ssim / len(testing_data_loader)) + '\n')


def save_checkpoint(epoch):
    model_folder = r"D:\yxd\code\ESRT-RS-dong-EVIT/experiment/checkpoint_mycmtSEN2VENUS_V16_seed_x{}/".format(args.scale)
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description="ESRT-RS")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="training batch size")
    parser.add_argument("--testBatchSize", type=int, default=1,
                        help="testing batch size")
    parser.add_argument("-nEpochs", type=int, default=10000,
                        help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning Rate. Default=2e-4")
    parser.add_argument("--step_size", type=int, default=250,
                        help="learning rate decay per N epochs")
    parser.add_argument("--gamma", type=int, default=0.5,
                        help="learning rate decay factor for step decay")
    parser.add_argument("--cuda", action="store_true", default=True,
                        help="use cuda")
    parser.add_argument("--resume", default="", type=str,
                        help="path to checkpoint")
    parser.add_argument("--start-epoch", default=1, type=int,
                        help="manual epoch number")
    parser.add_argument("--threads", type=int, default=0,
                        help="number of threads for data loading")
    parser.add_argument("--root", type=str, default=r"D:\yxd\code\ESRT-RS\npydata\sen2venus\datacut_one_site\train",  # 改
                        help='dataset directory')

    parser.add_argument("--rootval", type=str, default=r"D:\yxd\code\ESRT-RS\npydata\sen2venus\datacut_one_site\val",  # 改
                        help='dataset directory')
    # parser.add_argument("--root", type=str, default=r"D:\yxd\code\ESRT\npydata\DF2K_decoded", #改
    #                     help='dataset directory')
    parser.add_argument("--n_train", type=int, default=5339,  # 改
                        help="number of training set")
    parser.add_argument("--n_val", type=int, default=1771,
                        help="number of validation set")
    parser.add_argument("--test_every", type=int, default=1000)
    parser.add_argument("--scale", type=int, default=2,
                        help="super-resolution scale")
    parser.add_argument("--patch_size", type=int, default=48,
                        help="output patch size")
    parser.add_argument("--rgb_range", type=int, default=1,
                        help="maxium value of RGB")
    parser.add_argument("--n_colors", type=int, default=4,
                        help="number of color channels to use")
    parser.add_argument("--pretrained", default="", type=str,
                        help="path to pretrained models")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--isY", action="store_true", default=True)
    parser.add_argument("--ext", type=str, default='.npy')  # 改
    parser.add_argument("--phase", type=str, default='train')
    parser.add_argument("--model", type=str, default='ESRT')

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    # random seed
    seed = args.seed
    if seed is None:
        seed = random.randint(1, 10000)

    random.seed(seed)
    torch.manual_seed(seed)

    cuda = args.cuda
    device = torch.device("cuda" if cuda else "cpu")

    print("===> Loading datasets")

    train_dataset = tifdataset(args, False)
    val_dataset = tifdataset(args, True)
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=args.threads, batch_size=args.batch_size,
                                      shuffle=True, pin_memory=False, drop_last=True)

    testing_data_loader = DataLoader(dataset=val_dataset, num_workers=args.threads, batch_size=args.testBatchSize,
                                     shuffle=False)

    print("===> Building models")
    args.is_train = True

    model = model.cmtHfeaturesV16.myCmt(upscale=args.scale)

    if cuda:
        # model = nn.DataParallel(model, device_ids=[0, 1])
        model = model.to(device)

    # if isinstance(model, torch.nn.DataParallel):
    #     net = model.module

    modelName = "mycmt_sen2venus"

    l1_criterion = nn.L1Loss()

    print("===> Setting GPU")
    if cuda:
        # model = model.to(device)
        l1_criterion = l1_criterion.to(device)

    if args.pretrained:

        if os.path.isfile(args.pretrained):
            print("===> loading models '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            new_state_dcit = OrderedDict()
            for k, v in checkpoint.items():
                if 'module' in k:
                    name = k[7:]
                else:
                    name = k
                new_state_dcit[name] = v
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in new_state_dcit.items() if k in model_dict}

            for k, v in model_dict.items():
                if k not in pretrained_dict:
                    print(k)
            model.load_state_dict(pretrained_dict, strict=True)

        else:
            print("===> no models found at '{}'".format(args.pretrained))

    print("===> Setting Optimizer")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("===> Training")
    print_network(model)

    code_start = datetime.datetime.now()
    timer = utils.Timer()

    loss_path = r"D:\yxd\code\ESRT-RS-dong-EVIT\train_process\{}\V16\loss.txt".format(modelName)
    val_path = r"D:\yxd\code\ESRT-RS-dong-EVIT\train_process\{}\V16\val.txt".format(modelName)

    for epoch in range(args.start_epoch, args.nEpochs + 1):
        print("epoch:", epoch)
        t_epoch_start = timer.t()
        epoch_start = datetime.datetime.now()



        train(epoch, loss_path, scale=args.scale)
        if epoch % 10 == 0:
            save_checkpoint(epoch)
            valid(args.scale, val_path)


        epoch_end = datetime.datetime.now()
        print('Epoch cost times: %s' % str(epoch_end - epoch_start))
        t = timer.t()
        prog = (epoch - args.start_epoch + 1) / (args.nEpochs + 1 - args.start_epoch + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        print('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        gc.collect()
        torch.cuda.empty_cache()

    code_end = datetime.datetime.now()
    print('Code cost times: %s' % str(code_end - code_start))

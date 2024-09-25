import argparse
import torch
import os
import numpy as np
import utils
import skimage.color as sc
import cv2
from data import DIV2K, Set5_val
from model import esrt
import model.cmtHfeaturesV16
import rasterio
from torch.utils.data import DataLoader

scale_str = "2"
seed_str = "89"
# Testing settings
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
parser = argparse.ArgumentParser(description='ESRT')
parser.add_argument("--test_hr_folder", type=str, default=r'D:\yxd\code\ESRT-RS\npydata\sen2venus\datacut_one_site\test\HR\\',
                    help='the folder of the target images')
parser.add_argument("--test_lr_folder", type=str, default=r'D:\yxd\code\ESRT-RS\npydata\sen2venus\datacut_one_site\test\LR\X2',
                    help='the folder of the input images')
parser.add_argument("--output_folder", type=str, default=r'D:\yxd\code\ESRT-RS\npydata\sen2venus\datacut_one_site\test\SR_result\MSCT\X{}'.format(scale_str))
parser.add_argument("--checkpoint", type=str, default=r'D:\yxd\code\ESRT-RS-dong-EVIT\experiment\checkpoint_mycmtSEN2VENUS_V16_seed_x{}\epoch_1180_1.pth'.format(scale_str),
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument("--upscale_factor", type=int, default=int(scale_str),
                    help='upscaling factor')
parser.add_argument("--is_y", action='store_true', default=True,
                    help='evaluate on y channel, if False evaluate on RGB channels')
opt = parser.parse_args()

# print(opt)
def forward_chop(model, x, shave=4, min_size=10000,scale =2):
    # scale = 2#self.scale[self.idx_scale]
    n_GPUs = 1#min(self.n_GPUs, 4)
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch = model(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(model, patch, shave=shave, min_size=min_size, scale= scale) \
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

cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')

# filepath = opt.test_hr_folder

filepath = opt.test_lr_folder
ext = '.tif'



filelist = utils.get_list(filepath, ext=ext)
psnr_list = np.zeros(len(filelist))
ssim_list = np.zeros(len(filelist))
time_list = np.zeros(len(filelist))

model = model.cmtHfeaturesV16.myCmt(upscale=opt.upscale_factor)
if cuda:
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)

model.load_state_dict(torch.load(opt.checkpoint), strict=True)

model.eval()

i = 0
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for imname in filelist:



    with rasterio.open(opt.test_hr_folder + "\\" + imname.split('\\')[-1].split('.')[0] + ext) as HRdataset:
        profile = HRdataset.profile
        bands = HRdataset.read()
        height, width = bands.shape[1], bands.shape[2]
        meta = HRdataset.meta
        nrows, ncols = HRdataset.shape
        data = HRdataset.read()
        data = np.transpose(data, (1, 2, 0))
        hr_tensor = torch.from_numpy(data.copy().astype(np.float32)).permute(2, 0, 1) / 10000
        hr_tensor = hr_tensor.to(device)
        hr_tensor = hr_tensor.unsqueeze(0)


        with rasterio.open(opt.test_lr_folder +"\\" +imname.split('\\')[-1].split('.')[0] + ext) as src:
            profile = src.profile
            bands = src.read()
            height, width = bands.shape[1], bands.shape[2]
            meta = src.meta
            nrows, ncols = src.shape
            data =src.read()
            data = np.transpose(data, (1, 2, 0))

            lr_tensor = torch.from_numpy(data.copy().astype(np.float32)).permute(2, 0, 1) / 10000
            lr_tensor = lr_tensor.to(device)
            lr_tensor = lr_tensor.unsqueeze(0)

            # out_tensor = model(lr_tensor)
            start.record()
            out_tensor = forward_chop(model, lr_tensor,8)
            end.record()
            torch.cuda.synchronize()
            time_list[i] = start.elapsed_time(end)  # milliseconds

            out_img = utils.tensor2np(out_tensor.detach()[0])
            hr_img = utils.tensor2np(hr_tensor.detach()[0])

            output_folder = os.path.join(opt.output_folder,
                                         imname.split('\\')[-1].split('.')[0]+ '.tif')

            psnr_list[i] = utils.compute_psnr(out_img, hr_img)
            ssim_list[i] = utils.compute_ssim(out_img, hr_img)
            print(imname, ":", psnr_list[i])




            profile.update({
                'height': 256,
                'width': 256,
                'transform': src.transform * src.transform.scale(
                    (src.width / 256), (src.height / 256)
                )
            })
            naip_data2 = np.moveaxis(out_img.squeeze(), -1, 0)
            with rasterio.open(output_folder, 'w', **profile) as dst:
                dst.write(naip_data2)
                i=i+1

print("Mean PSNR: {}, SSIM: {}, TIME: {} ms".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)))

import scipy.misc
from glob import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import rasterio
import utils
from rasterio.transform import from_origin
from skimage.transform import resize
# 要遍历的文件夹路径
# root_folder = r'D:\yxd\data\RS\UCMerced_LandUse\UCMerced_LandUse\HR'  # 替换为你要遍历的文件夹的实际路径
root_folder = r'D:\yxd\code\ESRT-RS\npydata\sen2venus\datacut_one_site\test\LR\X2'
hr_folder = r'D:\yxd\code\ESRT-RS\npydata\sen2venus\datacut_one_site\test\HR'
out_folder = r"D:\yxd\code\ESRT-RS\npydata\sen2venus\datacut_one_site\test\SR_result\bicubic\X2"
imgs_lr = []
scale = [2]


filelist = utils.get_list(root_folder, ext=".tif")
psnr_list = np.zeros(len(filelist))
ssim_list = np.zeros(len(filelist))
time_list = np.zeros(len(filelist))

count = 0

for root, dirs, files in os.walk(root_folder):
    for file in files:
        if file.endswith('.tif'):
            img_path = os.path.join(root, file)


            with rasterio.open(hr_folder + "\\" + file.split('\\')[-1].split('.')[0] + ".tif") as HRdataset:
                profile = HRdataset.profile
                bands = HRdataset.read()
                height, width = bands.shape[1], bands.shape[2]
                meta = HRdataset.meta
                nrows, ncols = HRdataset.shape
                data = HRdataset.read()
                hr = np.transpose(data, (1, 2, 0))


                with rasterio.open(img_path) as src:
                    # 读取所有波段数据
                    bands = src.read()
                    profile = src.profile

                    # 获取原始图像的大小
                    height, width = bands.shape[1], bands.shape[2]

                    # 二倍下采样的目标大小
                    new_height = height * 2
                    new_width = width * 2

                    # 创建一个存储下采样图像的数组
                    downsampled_bands = np.zeros(( new_height, new_width,bands.shape[0],), dtype=bands.dtype)

                    # 对每个波段执行下采样（bicubic）
                    for i in range(bands.shape[0]):
                        downsampled_bands[:,:,i] = resize(bands[i], (new_height, new_width), order=3,
                                                      preserve_range=True).astype(bands.dtype)

                    # 更新profile信息
                    profile.update({
                        'height': new_height,
                        'width': new_width,
                        'transform': src.transform * src.transform.scale(
                            (src.width / new_width), (src.height / new_height)
                        )
                    })

                    psnr_list[count] = utils.compute_psnr(downsampled_bands, hr)
                    ssim_list[count] = utils.compute_ssim(downsampled_bands, hr)
                    print(img_path, ":", psnr_list[count])
                    count=count+1



                    naip_data2 = np.moveaxis(downsampled_bands.squeeze(), -1, 0)

                    # 保存新的下采样图像
                    with rasterio.open(out_folder+"//"+file, 'w', **profile) as dst:
                        dst.write(naip_data2)

print("Mean PSNR: {}, SSIM: {}, TIME: {} ms".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)))

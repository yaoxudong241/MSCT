import torch
import rasterio
from rasterio.transform import from_origin
import numpy as np
from tqdm import tqdm
import os

out_05_folder = r"D:\yxd\code\ESRT-RS\npydata\sen2venus\HR"
out_10_folder = r"D:\yxd\code\ESRT-RS\npydata\sen2venus\LR"

pt_path = r"C:\Users\BUAA_D723_4\Downloads\6514159 (1)"
for root, dirs, files in os.walk(pt_path):
    for file in tqdm(files):
        if file.endswith('_05m_b2b3b4b8.pt'):
            print(root)
            filepath05 = file.split("_05m_b2b3b4b8.pt")[0]+'_05m_b2b3b4b8.pt'
            filepath10 = file.split("_05m_b2b3b4b8.pt")[0]+'_10m_b2b3b4b8.pt'

            images05_tensor = torch.load(root + "\\" + filepath05)
            images10_tensor = torch.load(root + "\\" + filepath10)

            for i in range (images05_tensor.shape[0]):
                images05_tensor[i, :, :, :] = images05_tensor[i, :, :, :] - images05_tensor[i, :, :, :].min()
                images10_tensor[i, :, :, :] = images10_tensor[i, :, :, :] - images10_tensor[i, :, :, :].min()

            images05 = images05_tensor.numpy().astype(np.uint16)
            images10 = images10_tensor.numpy().astype(np.uint16)
            transform = from_origin(0, 0, 1, 1)  # 左上角的坐标和像素分辨率

            for i in range(images05.shape[0]):

                file_05_name = file.split("_05m_b2b3b4b8.pt")[0] +f'_{i + 1}.tif'
                file_10_name = file.split("_05m_b2b3b4b8.pt")[0] + f'_{i + 1}.tif'

                # 打开新的tif文件
                with rasterio.open(
                        out_05_folder + "//" +file_05_name,
                        'w',
                        driver='GTiff',
                        height=images05.shape[2],
                        width=images05.shape[3],
                        count=images05.shape[1],  # 波段数
                        dtype=images05.dtype,
                        transform=transform
                ) as dst:
                    for b in range(images05.shape[1]):
                        dst.write(images05[i, b, :, :], b + 1)

                # 打开新的tif文件
                with rasterio.open(
                        out_10_folder + "//" + file_10_name,
                        'w',
                        driver='GTiff',
                        height=images10.shape[2],
                        width=images10.shape[3],
                        count=images10.shape[1],  # 波段数
                        dtype=images10.dtype,
                        transform=transform
                ) as dst:
                    for b in range(images10.shape[1]):
                        dst.write(images10[i, b, :, :], b + 1)

            print("保存完成！")



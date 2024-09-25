import os
import shutil
import random

# 目标文件夹路径
destination_folder = [r'D:\yxd\code\ESRT-RS\npydata\sen2venus\datacut\val',
                      r'D:\yxd\code\ESRT-RS\npydata\sen2venus\datacut\test',
                      r'D:\yxd\code\ESRT-RS\npydata\sen2venus\datacut\train'
                      ]  # 将此路径替换为目标文件夹的实际路径




# 设置要遍历的文件夹路径
folder_path = r'D:\yxd\code\ESRT-RS\npydata\sen2venus\HR'
lr_path = r'D:\yxd\code\ESRT-RS\npydata\sen2venus\LR'

# val_num = ["1", "50", "100", "150", "200"]
scale_num = ["2"]


for root, dirs, files in os.walk(folder_path):
    for file in files:
        # 获取文件的扩展名并将其转换为小写
        file_extension = os.path.splitext(file)[1].lower()
        # 检查文件是否是PNG文件
        if file_extension == '.tif':

            random_number = random.uniform(0, 10)
            if (random_number > 8):
                print("test")
                HR_png_path = os.path.join(root, file)
                destination_hr_path = os.path.join(destination_folder[1], "HR")

                shutil.copy(HR_png_path, destination_hr_path)
                for i in range(len(scale_num)):
                    LR_png_X2_path = os.path.join(lr_path,
                                                  file.split(".")[0] +  ".tif")
                    destination_lr_path = os.path.join(destination_folder[1], "LR", "X" + scale_num[i])

                    shutil.copy(LR_png_X2_path, destination_lr_path)

            elif (random_number < 2):
                print("val")
                HR_png_path = os.path.join(root, file)
                destination_hr_path = os.path.join(destination_folder[0], "HR")

                shutil.copy(HR_png_path, destination_hr_path)
                for i in range(len(scale_num)):
                    LR_png_X2_path = os.path.join(lr_path,
                                                  file.split(".")[0]  + ".tif")
                    destination_lr_path = os.path.join(destination_folder[0], "LR", "X" + scale_num[i])

                    shutil.copy(LR_png_X2_path, destination_lr_path)

            else:
                print("train")
                HR_png_path = os.path.join(root, file)
                destination_hr_path = os.path.join(destination_folder[2], "HR")

                shutil.copy(HR_png_path, destination_hr_path)
                for i in range(len(scale_num)):
                    LR_png_X2_path = os.path.join(lr_path,
                                                  file.split(".")[0] + ".tif")
                    destination_lr_path = os.path.join(destination_folder[2], "LR", "X" + scale_num[i])

                    shutil.copy(LR_png_X2_path, destination_lr_path)



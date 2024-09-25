import os
import glob

# 文件夹路径
folder_path = r'D:\yxd\code\I2SB-master\dataset\val'

# 用于存储图像路径的txt文件路径
image_list_file = 'D:\yxd\code\I2SB-master\dataset\image_list.txt'

# 用于存储类别的txt文件路径
category_list_file = 'D:\yxd\code\I2SB-master\dataset\category_list.txt'

# 获取所有子文件夹的名称（类别）
categories = sorted(os.listdir(folder_path))

# 逐行存储图像路径到txt文件
with open(image_list_file, 'w') as f:
    for category in categories:
        category_folder = os.path.join(folder_path, category)
        # 使用glob匹配所有png图像文件
        image_files = glob.glob(os.path.join(category_folder, '*.png'))
        for image_file in image_files:
            f.write(image_file + '\n')

# 存储类别到txt文件
with open(category_list_file, 'w') as f:
    count = 0
    for category in categories:
        category_folder = os.path.join(folder_path, category)
        # 使用glob匹配所有png图像文件
        image_files = glob.glob(os.path.join(category_folder, '*.png'))
        for image_file in image_files:
            f.write(str(count) + '\n')
        count = count + 1
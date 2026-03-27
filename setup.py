import os

import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

# Installs
setuptools.setup(
    name="diffusion_planner",
    version="1.0.0",
    author="Zheng Yinan, Ruiming Liang, Kexin Zheng @ Tsinghua AIR",
    packages=["diffusion_planner"],
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: Free for non-commercial use",
    ],
    license="MIT",
)


# # 生成 train_set_list.json
# import os
# import json

# data_dir = "/home/yiliu/test"
# file_list = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
# file_list.sort()

# with open('train_set_list.json', 'w') as f:
#     json.dump(file_list, f, indent=2)

# print(f'共找到 {len(file_list)} 个文件')
# print(f'已保存到 train_set_list.json')




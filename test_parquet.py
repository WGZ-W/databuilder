import pandas as pd
import pandas as pd
import io
from PIL import Image
import numpy as np

# 读取 Parquet 文件
file_path = r"D:\Server\OpenFlyData\liujunli_1___OpenFly\raw\traj\env_airsim_16\astar_data\high_average\2025-01-18_23-38-59_526011.parquet"  # 👈 替换为你的文件路径
df = pd.read_parquet(file_path)

# 1. 打印基本信息
# print("DataFrame shape:", df.shape)
# print("\nColumns:")
# print(df.columns.tolist())
# print("\nData types:")
# print(df.dtypes)
# print("\nFirst 2 rows (non-image columns):")
# print(df.drop(columns=['image']).head(2))  # 先不显示 image 列

image_dict = df['image']
print(type(image_dict))
# print(image_dict)
img_dict = df['image'].iloc[0]
print(img_dict.keys())
for img_dict in image_dict:
    print(img_dict['path'])
print(img_dict['path'])
img_bytes = img_dict['bytes']
img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
img.show()

# 2. 查看 image 列是否为 bytes
# print("\nType of first 'image' entry:", type(df['image'].iloc[0]))

# 3. 尝试解码并显示第一张图像（可选：保存或显示）
# img_bytes = df['image'].iloc[0]
# img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

# 保存图像到本地（方便查看）
# img.save("first_frame.jpg")
# print("\n✅ 第一帧图像已保存为 'first_frame.jpg'")

# 或者直接显示（在 Jupyter Notebook 中有效）
# img.show()
import tensorflow_datasets as tfds

# 加载数据集
ds = tfds.load(
    # 'vln',
    'vln_history',
    # 'vlnv1',
    # data_dir='D:/Server/tensorflow_datasets',  # 如果你指定了非默认路径
    # data_dir='D:/Server/OpenFly/OpenFly-rlds',  # 如果你指定了非默认路径
    data_dir=r'D:\Server\OpenFly_My_Data',  # 如果你指定了非默认路径
    split='train',
    shuffle_files=False
)

# 查看一个 episode
for episode in ds.take(1):
    print("Episode metadata:", episode['episode_metadata'])
    steps = episode['steps']

    # 遍历 steps（这是一个 tf.data.Dataset）
    print(f"Lengths of steps is {len(steps)}")
    for step in steps:
        print("Action:", step['action'])
        print("Instruction:", step['language_instruction'].numpy().decode('utf-8'))
        print("Image shape:", step['observation']['image_1'].shape)
        print("History shape:", step['observation']['history_images'].shape)
        print("observation keys:", step['observation'].keys())

        # print("Image shape:", step['observation']['images'].shape)
        break  # 只看第一步
import tensorflow as tf
import tensorflow_datasets as tfds

# 加载原始数据集（无 images 字段）
ds = tfds.load('vln', split='train')

# 扁平化 steps
flat_ds = ds.flat_map(lambda ep: ep['steps'])

# 批处理（注意：不跨 episode）
# 实际中应使用 tf.data 的 group_by_window 或自定义分组

# 简化：假设我们按固定 batch 处理（忽略 episode 边界）
batched_ds = flat_ds.batch(32)

for batch in batched_ds:
    current_image = batch['observation']['image_1']  # (B, 224, 224, 3)
    action = batch['action']  # (B, 8)

    # 如果需要历史，用 RNN：
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Conv2D(...),
    #     tf.keras.layers.Reshape((B, -1)),
    #     tf.keras.layers.LSTM(128),
    #     tf.keras.layers.Dense(8)
    # ])

    # 或用 Transformer + positional encoding
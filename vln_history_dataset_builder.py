import os
import json
import numpy as np
from PIL import Image
import io
import pyarrow.parquet as pq
import tensorflow_datasets as tfds
import tensorflow as tf
import logging


class Builder(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Place train.json and all .parquet files under manual_dir with structure:
    manual_dir/
    ├── train.json
    └── env_airsim_18/astar_data/low_short/xxx.parquet
    """

    # def __init__(self, **kwargs):
    #     config = tfds.core.BuilderConfig(
    #         name='vln16',
    #     )
    #
    #     super().__init__(config=config, **kwargs)


    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'action': tfds.features.Tensor(shape=(8,), dtype=tf.float32),
                    'history': tfds.features.Text(),
                    'is_terminal': tf.bool,
                    'is_last': tf.bool,
                    'language_instruction': tfds.features.Text(),
                    'observation': {
                        'image_1': tfds.features.Image(shape=(224, 224, 3)),
                        'image_2': tfds.features.Image(shape=(224, 224, 3)),
                        'image_3': tfds.features.Image(shape=(224, 224, 3)),
                        # 修改为固定形状的Tensor
                        'history_images': tfds.features.Tensor(
                            shape=(8, 224, 224, 3),
                            dtype=tf.uint8
                        ),
                        # 'history_images': tfds.features.Sequence(tfds.features.Image(shape=(224, 224, 3))),
                    },
                    'is_first': tf.bool,
                    'discount': tf.float32,
                    'reward': tf.float32,
                }),
                'episode_metadata': {
                    'file_path': tfds.features.Text(),
                    'episode_id': tf.int32,
                    'has_language': tf.bool,
                    'has_image_1': tf.bool,
                    'has_image_2': tf.bool,
                    'has_image_3': tf.bool,
                    'has_history_images': tf.bool,
                    'has_image_0': tf.bool,
                },
            }),
        )

    def _split_generators(self, dl_manager):
        json_path = os.path.join(dl_manager.manual_dir, "train.json")
        # json_path = os.path.join(dl_manager.manual_dir, "test.json")
        return {'train': self._generate_examples(json_path, dl_manager.manual_dir)}

    def _generate_examples(self, json_path, manual_dir):
        # Action mapping
        action_dict = {
            str(i): np.array(v, dtype=np.float32)
            for i, v in enumerate([
                [1,0,0,0,0,0,0,0],  # 0: stop
                [0,3,0,0,0,0,0,0],  # 1: forward
                [0,0,15,0,0,0,0,0], # 2: turn left
                [0,0,0,15,0,0,0,0], # 3: turn right
                [0,0,0,0,2,0,0,0],  # 4: up
                [0,0,0,0,0,2,0,0],  # 5: down
                [0,0,0,0,0,0,5,0],  # 6: left
                [0,0,0,0,0,0,0,5],  # 7: right
                [0,6,0,0,0,0,0,0],  # 8: forward
                [0,9,0,0,0,0,0,0],  # 9: forward
            ])
        }

        def history_recorder(actions_so_far, max_len=100):
            if not actions_so_far:
                s = ""
            else:
                summary = [actions_so_far[0]]
                for a in actions_so_far[1:]:
                    if a != summary[-1]:
                        summary.append(a)
                s = ' then '.join(summary)

            # 截断或填充到固定长度
            s = s[:max_len]
            s = s.ljust(max_len)  # 右侧填充空格
            return s


        with tf.io.gfile.GFile(json_path, "r") as f:
            episodes = json.load(f)

        for ep_idx, ep in enumerate(episodes):
            try:
                # 1. 构造 parquet 路径
                parquet_rel = ep["image_path"] + ".parquet"  # 添加 .parquet
                parquet_path = os.path.join(manual_dir, parquet_rel)

                # 2. 读取 Parquet
                table = pq.read_table(parquet_path)
                df = table.to_pandas()

                # ... 在读取 df 之后 ...

                raw_actions = ep["action"]
                index_list = ep["index_list"]
                instruction = ep["gpt_instruction"]

                if len(raw_actions) != len(index_list):
                    logging.warning(f"Episode {ep_idx}: 动作与 index_list 长度不匹配")
                    continue

                # All images
                # ✅ 从 Parquet 的 'image' 列构建 ID -> 图像映射（基于 path 去后缀）
                id_to_image = {}
                id_to_timestamp = {}  # 新：图像ID到时间戳（索引）的映射
                all_images = []
                # for img_dict in df['image']:
                for timestamp, img_dict in enumerate(df['image']):
                    if not isinstance(img_dict, dict):
                        continue
                    img_bytes = img_dict.get('bytes')
                    img_path = img_dict.get('path', '')
                    if not img_bytes or not img_path:
                        continue
                    if isinstance(img_path, bytes):
                        img_path = img_path.decode('utf-8')
                    # 去掉文件扩展名，提取图像ID
                    img_id = os.path.splitext(os.path.basename(img_path))[0]

                    # 处理图像
                    try:
                        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
                        # 存储映射
                        id_to_image[img_id] = np.array(img, dtype=np.uint8)
                        all_images.append(np.array(img, dtype=np.uint8))
                        id_to_timestamp[img_id] = timestamp  # 记录时间戳
                    except Exception as e:
                        logging.warning(f"图像解码失败: {img_path}, error: {e}")

                # 按 index_list 顺序提取图像
                images = []
                timestamps = []  # 每个动作对应的时间戳
                for img_id in index_list:
                    if img_id not in id_to_image:
                        logging.warning(f"Episode {ep_idx}: 图像 ID '{img_id}' 未找到")
                        images = None
                        break
                    images.append(id_to_image[img_id])
                    timestamps.append(id_to_timestamp[img_id])

                if images is None or len(images) != len(index_list):
                    continue

                # ... 后续步骤（拐点、steps）保持不变 ...

                # 5. 拐点计算
                keypoints = [0]
                for i in range(1, len(raw_actions)):
                    if raw_actions[i] != raw_actions[i - 1]:
                        keypoints.append(i)

                # 6. 构建 steps
                steps = []
                for t in range(len(raw_actions)):
                    act_str = str(raw_actions[t])
                    if act_str not in action_dict:
                        continue

                    # 获取当前动作对应的时间戳
                    current_timestamp = timestamps[t]

                    current_kp = max(kp for kp in keypoints if kp <= t)
                    prev_kps = [kp for kp in keypoints if kp < current_kp]
                    last_kp = prev_kps[-1] if prev_kps else current_kp

                    # === 构造 history_images：t 时刻前 8 帧，不足则用 images[0] 前向填充 ===
                    history_length = 8
                    start_timestamp = max(0, current_timestamp - history_length)
                    actual_history = all_images[start_timestamp:current_timestamp]
                    if len(actual_history) < history_length:
                        pad_count = history_length - len(actual_history)
                        if actual_history:
                            fill_frame = actual_history[0]
                        else:
                            fill_frame = all_images[0]
                        actual_history = [fill_frame] * pad_count + actual_history
                    # ===================================================================
                    history_array = np.array(actual_history, dtype=np.uint8)

                    # 确保形状正确
                    # if history_array.shape != (8, 224, 224, 3):
                    #     history_array = history_array.reshape((8, 224, 224, 3))

                    steps.append({
                        'action': action_dict[act_str],
                        'history': history_recorder([str(a) for a in raw_actions[:t]]),
                        'is_terminal': t == len(raw_actions) - 1,
                        'is_last': t == len(raw_actions) - 1,
                        'language_instruction': instruction,
                        'observation': {
                            'image_1': images[t],
                            'image_2': images[last_kp],
                            'image_3': images[current_kp],
                            # 'history_images': np.array(actual_history, dtype=np.uint8),
                            'history_images': history_array,
                            # 'history_images': actual_history,
                        },
                        'is_first': t == 0,
                        'discount': 1.0,
                        'reward': 1.0 if t == len(raw_actions) - 1 else 0.0,
                    })

                if steps:
                    yield f"ep_{ep_idx}", {
                        'steps': steps,
                        'episode_metadata': {
                            'file_path': parquet_path,
                            'episode_id': ep_idx,
                            'has_language': True,
                            'has_image_1': True,
                            'has_image_2': True,
                            'has_image_3': True,
                            'has_history_images': True,
                            'has_image_0': False,
                        }
                    }

            except Exception as e:
                logging.warning(f"Skipping episode {ep_idx} (path: {parquet_path}): {e}")
                continue

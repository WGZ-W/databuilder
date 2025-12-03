"""bridge_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from pathlib import Path
import os
from PIL import Image
import json


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for VLN trajectory dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {'1.0.0': 'Initial release.'}

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Place train.json in manual_dir. Expected structure:
    manual_dir/
      train.json
    Each entry in train.json should have:
      - image_path: root path to images
      - gpt_instruction: language instruction
      - action: list of int actions
      - index_list: list of image filenames (without extension)
    """

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="VLN trajectory dataset with language instructions and actions.",
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'action': tfds.features.Tensor(shape=(8,), dtype=tf.float32),
                    'history': tfds.features.Text(),
                    'is_terminal': tf.bool,
                    'is_last': tf.bool,
                    'language_instruction': tfds.features.Text(),
                    'observation': {
                        'image_1': tfds.features.Image(shape=(224, 224, 3), encoding_format='png'),
                        'image_2': tfds.features.Image(shape=(224, 224, 3), encoding_format='png'),
                        'image_3': tfds.features.Image(shape=(224, 224, 3), encoding_format='png'),
                        # 'images': tfds.features.Sequence(
                        #     tfds.features.Image(shape=(224, 224, 3), encoding_format='png')
                        # ),
                    },
                    'is_first': tf.bool,
                    'discount': tf.float32,
                    'reward': tf.float32,
                }),
                'episode_metadata': {
                    'has_image_2': tf.bool,
                    'has_image_3': tf.bool,
                    'file_path': tfds.features.Text(),
                    'has_language': tf.bool,
                    'has_image_1': tf.bool,
                    'has_image_0': tf.bool,
                    'episode_id': tf.int32,
                },
            }),
            supervised_keys=None,
            homepage='https://dataset-homepage/',
            citation=r"""@misc{vln_2024, title={VLN Dataset}, year={2024}}""",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        json_path = os.path.join(dl_manager.manual_dir, "train.json")
        return {'train': self._generate_examples(json_path)}

    def _generate_examples(self, json_path):
        with tf.io.gfile.GFile(json_path, "r") as f:
            episodes = json.load(f)

        # Action mapping (from int to 8D float vector)
        action_dict = {
            "0": np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),  # stop
            "1": np.array([0, 3, 0, 0, 0, 0, 0, 0], dtype=np.float32),  # move forward
            "2": np.array([0, 0, 15, 0, 0, 0, 0, 0], dtype=np.float32),  # turn left
            "3": np.array([0, 0, 0, 15, 0, 0, 0, 0], dtype=np.float32),  # turn right
            "4": np.array([0, 0, 0, 0, 2, 0, 0, 0], dtype=np.float32),  # go up
            "5": np.array([0, 0, 0, 0, 0, 2, 0, 0], dtype=np.float32),  # go down
            "6": np.array([0, 0, 0, 0, 0, 0, 5, 0], dtype=np.float32),  # move left
            "7": np.array([0, 0, 0, 0, 0, 0, 0, 5], dtype=np.float32),  # move right
            "8": np.array([0, 6, 0, 0, 0, 0, 0, 0], dtype=np.float32),  # move forward
            "9": np.array([0, 9, 0, 0, 0, 0, 0, 0], dtype=np.float32),  # move forward
        }

        def history_recorder(actions_so_far):
            if not actions_so_far:
                return ""
            summary = [actions_so_far[0]]
            for a in actions_so_far[1:]:
                if a != summary[-1]:
                    summary.append(a)
            return ' then '.join(summary)

        for ep_idx, ep in enumerate(episodes):
            # --- 字段对齐 ---
            try:
                img_root = ep["image_path"]  # e.g., "env_airsim_18/astar_data/..."
                instruction = ep["gpt_instruction"]
                raw_actions = ep["action"]  # list of int
                index_list = ep["index_list"]  # list of str
            except KeyError as e:
                tfds.core.logging.warn(f"Skipping episode {ep_idx}: missing key {e}")
                continue

            if len(raw_actions) != len(index_list):
                tfds.core.logging.warn(f"Skipping episode {ep_idx}: action and index_list length mismatch")
                continue

            # --- 加载所有图像 ---
            images = []
            valid = True
            for idx_name in index_list:
                img_path = os.path.join(img_root, idx_name + ".png")
                try:
                    img = np.array(Image.open(img_path).convert('RGB').resize((224, 224)), dtype=np.uint8)
                    images.append(img)
                except Exception as e:
                    tfds.core.logging.warn(f"Failed to load {img_path}: {e}")
                    valid = False
                    break
            if not valid:
                continue

            # --- 预计算拐点索引（动作变化的位置，包含起点 0）---
            keypoints = [0]
            for i in range(1, len(raw_actions)):
                if raw_actions[i] != raw_actions[i - 1]:
                    keypoints.append(i)

            # --- 构建 steps ---
            steps = []
            num_steps = len(raw_actions)

            for t in range(num_steps):
                act_str = str(raw_actions[t])
                if act_str not in action_dict:
                    tfds.core.logging.warn(f"Unknown action {act_str} in episode {ep_idx}")
                    continue

                # 构建 history
                prev_actions = [str(a) for a in raw_actions[:t]]
                history = history_recorder(prev_actions)

                current_img = images[t]
                image_1 = current_img

                # 找 ≤ t 的最大拐点 → 当前动作段的起始帧
                current_keypoint = max([kp for kp in keypoints if kp <= t])
                image_3 = images[current_keypoint]

                # 找 < current_keypoint 的最大拐点 → 上一个动作段的起始帧
                prev_keypoints = [kp for kp in keypoints if kp < current_keypoint]
                if prev_keypoints:
                    last_keypoint = prev_keypoints[-1]
                    image_2 = images[last_keypoint]
                else:
                    # 如果是第一个动作段，image_2 = image_3
                    image_2 = image_3

                # images: 所有历史图像（不包括当前）
                seq_images = images[:t]

                steps.append({
                    'action': action_dict[act_str],
                    'history': history,
                    'is_terminal': t == num_steps - 1,
                    'is_last': t == num_steps - 1,
                    'language_instruction': instruction,
                    'observation': {
                        'image_1': image_1,
                        'image_2': image_2,
                        'image_3': image_3,
                        'images': seq_images,
                    },
                    'is_first': t == 0,
                    'discount': 1.0,
                    'reward': 1.0 if t == num_steps - 1 else 0.0,
                })

            if not steps:
                continue

            yield "vln_norm", {
                'steps': steps,
                'episode_metadata': {
                    'has_image_2': True,
                    'has_image_3': True,
                    'file_path': img_root,
                    'has_language': True,
                    'has_image_1': True,
                    'has_image_0': False,
                    'episode_id': ep_idx,
                },
            }
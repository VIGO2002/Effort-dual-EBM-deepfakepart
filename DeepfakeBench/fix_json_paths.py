import json
import os
import glob
from tqdm import tqdm

# 配置你的路径
JSON_DIR = './preprocessing/dataset_json'  # JSON文件夹路径
NEW_ROOT = '/root/autodl-tmp'              # 你的数据存放根目录

# 需要修改的数据集名称映射（JSON文件名 -> 文件夹名）
DATASET_MAP = {
    'FaceForensics++': 'FaceForensics++',
    'Celeb-DF-v2': 'Celeb-DF-v2',
    # 如果有其他数据集继续添加，比如 'FaceShifter': 'FaceForensics++' (因为FaceShifter通常在FF++包里)
}

def replace_path(path, dataset_name):
    # 找到路径中数据集名称的位置，截断并替换前半部分
    folder_name = DATASET_MAP.get(dataset_name, dataset_name)
    if folder_name in path:
        # 保留数据集名称及之后的部分
        suffix = path.split(folder_name)[-1]
        # 拼接新路径: /root/autodl-tmp/FaceForensics++/original_sequences/...
        # 注意处理路径分隔符
        if suffix.startswith('/') or suffix.startswith('\\'):
            suffix = suffix[1:]
        new_path = os.path.join(NEW_ROOT, folder_name, suffix)
        return new_path.replace('\\', '/') # 统一转为 Linux 路径
    return path

json_files = glob.glob(os.path.join(JSON_DIR, '*.json'))
print(f"Found {len(json_files)} json files.")

for json_file in json_files:
    dataset_name = os.path.basename(json_file).replace('.json', '')
    # 只处理我们要用的数据集
    if dataset_name not in DATASET_MAP and 'FaceShifter' not in dataset_name: 
        continue

    print(f"Processing {json_file}...")

    with open(json_file, 'r') as f:
        data = json.load(f)

    modified = False
    # 递归遍历 JSON 结构
    for method_key in data.keys(): # 如 "FaceForensics++"
        for type_key in data[method_key].keys(): # 如 "Original", "Deepfakes"
            for split_key in data[method_key][type_key].keys(): # "train", "test"
                # FF++ 还有一层压缩率 'c23', 'c40'
                level_data = data[method_key][type_key][split_key]

                # 检查是否还有一层（比如 compression level）
                first_val = list(level_data.values())[0] if level_data else None
                if isinstance(first_val, dict) and 'frames' not in first_val:
                    # 还有一层，比如 'c23'
                    for comp_key in level_data.keys():
                        videos = level_data[comp_key]
                        for video_name, video_info in videos.items():
                            video_info['frames'] = [replace_path(p, dataset_name) for p in video_info['frames']]
                            modified = True
                else:
                    # 直接是视频列表
                    for video_name, video_info in level_data.items():
                        video_info['frames'] = [replace_path(p, dataset_name) for p in video_info['frames']]
                        modified = True

    if modified:
        with open(json_file, 'w') as f:
            json.dump(data, f)
        print(f"Updated {json_file}")

print("Done! Paths updated.")
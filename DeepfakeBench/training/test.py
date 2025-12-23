"""
eval pretained model with Score Fusion support.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='./training/config/detector/effort.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str,
                    default='./weights/effort_ckpt.pth')
# [新增] 第二个权重的路径参数
parser.add_argument('--weights_path_2', type=str,
                    default=None,
                    help='path to the second detector weights for score fusion')
#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

on_2060 = "2060" in torch.cuda.get_device_name()
def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        
        # 1. 使用原本正确的类名 (DeepfakeAbstractBaseDataset)
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test',
            )
            
        # 2. 创建 DataLoader
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'], 
                shuffle=False,
                num_workers=8,           # 保持高性能设置
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


# [修改] 增加 model2 参数
def test_one_dataset(model, data_loader, model2=None):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward
        predictions = inference(model, data_dict)
        
        # [核心修改] 如果存在 model2，则进行融合
        if model2 is not None:
            predictions2 = inference(model2, data_dict)
            
            # 权重设置：Model 1 (无噪) 权重 0.3，Model 2 (有噪) 权重 0.7
            # 逻辑：Model 2 泛化性更好，更值得信任；Model 1 只在它极其确信的时候做补充
            w1 = 0.6 
            w2 = 0.4
            
            predictions['prob'] = w1 * predictions['prob'] + w2 * predictions2['prob']

        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        feature_lists += list(predictions['feat'].cpu().detach().numpy())

    return np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)

# [修改] 增加 model2 参数
def test_epoch(model, test_data_loaders, model2=None):
    # set model to eval mode
    model.eval()
    if model2 is not None:
        model2.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        # [修改] 传入 model2
        predictions_nps, label_nps,feat_nps = test_one_dataset(model, test_data_loaders[key], model2)

        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset

        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")

    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions

# [新增] 辅助函数：加载权重
def load_weights_into_model(model, weights_path):
    print(f'Loading weights from: {weights_path}')
    try:
        epoch = 0
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        # 创建一个新的字典，删除module前缀
        new_weights = {}
        for key, value in ckpt.items():
            new_key = key.replace('module.', '')  # 删除module前缀
            new_weights[new_key] = value

        model.load_state_dict(new_weights, strict=True)
        print('===> Load checkpoint done!')
    except Exception as e:
        print(f'Fail to load the pre-trained weights: {e}')


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if on_2060:
        config['lmdb_dir'] = r'I:\transform_2_lmdb'
        config['train_batchSize'] = 10
        config['workers'] = 0
    else:
        config['workers'] = 8
        config['lmdb_dir'] = r'/mnt/chongqinggeminiceph1fs/geminicephfs/mm-base-vision/jikangcheng/data/LMDBs'
    
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model 1 (detector)
    print(">>> Initializing Model 1...")
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters in the model: {total_trainable_params}")
    
    if weights_path:
        load_weights_into_model(model, weights_path)
    else:
        print('Fail to load the pre-trained weights for Model 1')

    # [新增] 准备 model 2 (如果有)
    model2 = None
    if args.weights_path_2:
        print(">>> Initializing Model 2 for Fusion...")
        model2 = model_class(config).to(device)
        load_weights_into_model(model2, args.weights_path_2)

    # start testing
    best_metric = test_epoch(model, test_data_loaders, model2)
    print('===> Test Done!')

if __name__ == '__main__':
    main()
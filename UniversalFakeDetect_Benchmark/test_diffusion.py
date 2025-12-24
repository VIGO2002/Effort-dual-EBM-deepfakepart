import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, accuracy_score
from models.trainer import Trainer
from options.train_options import TrainOptions
import numpy as np
from tqdm import tqdm

# 配置
BASE_REAL_PATH = "/root/autodl-tmp/datasets/CNNDetection/biggan/0_real" # ImageNet Real
DIFFUSION_ROOT = "/root/autodl-tmp/datasets/Diffusion"
BASELINE_GUIDED = 95.39 

def load_diffusion_vs_imagenet(fake_path, transform):
    try:
        if not os.path.exists(BASE_REAL_PATH):
            print(f"❌ Error: Real path not found: {BASE_REAL_PATH}")
            return None
        
        # Load Real
        real_samples = []
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        for f in os.listdir(BASE_REAL_PATH):
            if f.lower().endswith(valid_ext):
                real_samples.append((os.path.join(BASE_REAL_PATH, f), 0))
        
        # Load Fake
        fake_samples = []
        for root, _, files in os.walk(fake_path):
            for f in files:
                if f.lower().endswith(valid_ext):
                    fake_samples.append((os.path.join(root, f), 1))
        
        if len(fake_samples) == 0: return None

        print(f"   📊 Data: {len(real_samples)} Real vs {len(fake_samples)} Fake")
        
        dataset = datasets.ImageFolder(root=os.path.dirname(BASE_REAL_PATH), transform=transform)
        full_samples = real_samples + fake_samples
        dataset.samples = full_samples
        dataset.targets = [s[1] for s in full_samples]
        return dataset

    except Exception as e:
        print(f"❌ Dataset Error: {e}")
        return None

def run_test(model, dataset_name, root_path, transform):
    fake_path = os.path.join(root_path, dataset_name)
    print(f"\n{'='*10} ⚔️  Challenging {dataset_name.upper()} ⚔️  {'='*10}")
    
    dataset = load_diffusion_vs_imagenet(fake_path, transform)
    if dataset is None: return None
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    y_true, y_pred = [], []
    model.model.cuda()
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
            model.set_input(data)
            model.test()
            
            # 【核心修改】EBM 评分标准
            # output = [-E_real, -E_fake]
            pred = model.output
            if pred.shape[1] == 2:
                # Score = (-E_fake) - (-E_real) = E_real - E_fake
                # 如果 E_real (真图能量) 高，E_fake (假图能量) 低，则 Score > 0 (判为假)
                # 这种 Logit 差值比 Softmax 包含更多信息
                prob = (pred[:, 1] - pred[:, 0]).cpu().numpy()
            else:
                prob = pred.cpu().numpy().flatten()
            
            y_true.extend(data[1].cpu().numpy())
            y_pred.extend(prob)

    mAP = average_precision_score(y_true, y_pred)
    acc = accuracy_score(y_true, [1 if p > 0 else 0 for p in y_pred]) # 阈值改为 0 (因为是 Logit 差值)
    
    status = "Fail ❌"
    if dataset_name == 'guided':
        if mAP * 100 > BASELINE_GUIDED: status = "VICTORY! 🏆 (SOTA)"
        else: status = f"Lagging by {BASELINE_GUIDED - mAP * 100:.2f}%"
            
    print(f"🎯 Result for {dataset_name}:")
    print(f"   mAP: {mAP:.4f} ({mAP*100:.2f}%) | Acc: {acc:.4f} | {status}")
    return mAP

if __name__ == "__main__":
    opt = TrainOptions().parse(print_options=False)
    opt.isTrain = False; opt.gpu_ids = [0]; opt.name = 'effort_universal_repro'; opt.checkpoints_dir = './checkpoints'
    opt.arch = 'CLIP:ViT-L/14_svd'; opt.fix_backbone = True; opt.noise_std = 0.02
    
    print("⚡️ Loading Pure Dual-EBM Model...")
    model = Trainer(opt)
    model.eval()
    
    # 建议使用训练后的最新权重
    ckpt_path = './checkpoints/effort_universal_repro/model_epoch_8.pth' 
    if not os.path.exists(ckpt_path):
        ckpt_path = './checkpoints/effort_universal_repro/model_epoch_3.pth'

    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'model' in state_dict: state_dict = state_dict['model']
    if hasattr(model.model, "module"): model.model.module.load_state_dict(state_dict, strict=False)
    else: model.model.load_state_dict(state_dict, strict=False)
    print(f"✅ Weights loaded from {ckpt_path}!")

    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    TARGETS = ['guided', 'ldm_100', 'ldm_200_cfg', 'glide_100_27', 'dalle']
    for d_name in TARGETS:
        run_test(model, d_name, DIFFUSION_ROOT, val_transform)

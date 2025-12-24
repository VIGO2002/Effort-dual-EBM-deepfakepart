import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import average_precision_score, accuracy_score
from models.trainer import Trainer
from options.train_options import TrainOptions
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 真图基准：Guided Diffusion 和 LDM 都是基于 ImageNet 的
# 所以必须使用 BigGAN 的真图 (ImageNet) 作为负样本
BASE_REAL_PATH = "/root/autodl-tmp/datasets/CNNDetection/biggan/0_real"

# 2. 扩散模型假图根目录
DIFFUSION_ROOT = "/root/autodl-tmp/datasets/Diffusion"

# 3. 你的目标：原论文在 Guided 上的分数
BASELINE_GUIDED = 95.39 
# ===========================================

def load_diffusion_vs_imagenet(fake_path, transform):
    """
    加载策略：
    Real: BigGAN/0_real (ImageNet) -> Label 0
    Fake: 指定的扩散模型文件夹 -> Label 1
    """
    try:
        # --- 1. 加载 Real (ImageNet) ---
        if not os.path.exists(BASE_REAL_PATH):
            print(f"❌ Error: Real path not found: {BASE_REAL_PATH}")
            return None
        
        # 手动构建 Real Dataset
        real_samples = []
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        for f in os.listdir(BASE_REAL_PATH):
            if f.lower().endswith(valid_ext):
                real_samples.append((os.path.join(BASE_REAL_PATH, f), 0)) # Label 0
        
        # --- 2. 加载 Fake (Diffusion) ---
        fake_samples = []
        # 递归扫描 fake_path 下的所有图片
        for root, _, files in os.walk(fake_path):
            for f in files:
                if f.lower().endswith(valid_ext):
                    fake_samples.append((os.path.join(root, f), 1)) # Label 1
        
        if len(fake_samples) == 0:
            print(f"⚠️  No images found in {fake_path}")
            return None

        # --- 3. 打印数据量对比 ---
        print(f"   📊 Data: {len(real_samples)} Real (ImageNet) vs {len(fake_samples)} Fake (Diffusion)")
        
        # --- 4. 组装 Dataset ---
        # 借用 ImageFolder 的结构，但替换 samples
        # 这里随便指一个存在的路径初始化即可，重点是后面的 samples 覆盖
        dataset = datasets.ImageFolder(root=os.path.dirname(BASE_REAL_PATH), transform=transform)
        
        # 合并样本
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
    
    # Batch size 32 保证显存安全
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    y_true, y_pred = [], []
    model.model.cuda()
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
            model.set_input(data)
            model.test()
            
            # 获取预测结果
            # 如果你的 EBM 有特殊的评分机制 (比如 output 是 energy score)，
            # 这里可能需要调整。目前假设 output 依然是 logits。
            pred = model.output
            
            if pred.shape[1] == 1:
                prob = torch.sigmoid(pred).cpu().numpy().flatten()
            else:
                prob = torch.softmax(pred, dim=1)[:, 1].cpu().numpy()
            
            y_true.extend(data[1].cpu().numpy())
            y_pred.extend(prob)

    mAP = average_precision_score(y_true, y_pred)
    acc = accuracy_score(y_true, [1 if p > 0.5 else 0 for p in y_pred])
    
    # 结果判定
    status = "Fail ❌"
    if dataset_name == 'guided':
        if mAP * 100 > BASELINE_GUIDED:
            status = "VICTORY! 🏆 (SOTA)"
        else:
            gap = BASELINE_GUIDED - mAP * 100
            status = f"Lagging by {gap:.2f}%"
            
    print(f"🎯 Result for {dataset_name}:")
    print(f"   mAP: {mAP:.4f} ({mAP*100:.2f}%) | Acc: {acc:.4f} | {status}")
    return mAP

if __name__ == "__main__":
    # --- 1. 初始化模型 (Epoch 8) ---
    opt = TrainOptions().parse(print_options=False)
    opt.isTrain = False; opt.gpu_ids = [0]; opt.name = 'effort_universal_repro'; opt.checkpoints_dir = './checkpoints'
    opt.arch = 'CLIP:ViT-L/14_svd'; opt.fix_backbone = True; opt.noise_std = 0.02
    
    print("⚡️ Loading Your Modified Model (Dual-Head EBM)...")
    model = Trainer(opt)
    model.eval()
    
    # 加载最强的 Epoch 8
    ckpt_path = './checkpoints/effort_universal_repro/model_epoch_3.pth'
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'model' in state_dict: state_dict = state_dict['model']
    
    # 兼容性加载
    if hasattr(model.model, "module"): model.model.module.load_state_dict(state_dict, strict=False)
    else: model.model.load_state_dict(state_dict, strict=False)
    print("✅ Weights loaded! Ready to fight Diffusion.")

    # --- 2. Transform ---
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    # --- 3. 目标数据集列表 (根据你的 ls 结果) ---
    # 重点关注 guided
    TARGETS = [
        'guided',          # 重点！Target: > 95.39
        'ldm_100',         # Latent Diffusion
        'ldm_200_cfg',     # Classifier Free Guidance
        'glide_100_27',    # GLIDE
        'dalle',           # DALL-E
        'pndm'             # PNDM Sampler
    ]

    results = {}
    print(f"\n🎯 Baseline to beat (Guided): {BASELINE_GUIDED}%")
    
    for d_name in TARGETS:
        score = run_test(model, d_name, DIFFUSION_ROOT, val_transform)
        if score is not None:
            results[d_name] = score

    print(f"\n{'='*20} 🏆 Final Diffusion Leaderboard 🏆 {'='*20}")
    for k, v in results.items():
        print(f"{k.ljust(15)}: {v:.4f}")

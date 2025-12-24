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

def test_generalization(epoch_num, test_dataset_path):
    dataset_name = os.path.basename(test_dataset_path)
    print(f"\n>> Testing Epoch {epoch_num} on {dataset_name}...")
    
    # --- 1. Mock Options ---
    opt = TrainOptions().parse(print_options=False)
    opt.isTrain = False
    opt.gpu_ids = [0]
    opt.name = 'effort_universal_repro'
    opt.checkpoints_dir = './checkpoints'
    opt.arch = 'CLIP:ViT-L/14_svd' 
    opt.fix_backbone = True
    opt.noise_std = 0.02
    
    # --- 2. Init Model ---
    try:
        model = Trainer(opt)
        model.eval()
    except Exception as e:
        print(f"❌ Init Error: {e}")
        return 0, 0
    
    # --- 3. Load Weights ---
    ckpt_path = os.path.join(opt.checkpoints_dir, opt.name, f'model_epoch_{epoch_num}.pth')
    
    if not os.path.exists(ckpt_path):
        return 0, 0

    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        if hasattr(model.model, "module"):
            model.model.module.load_state_dict(state_dict)
        else:
            model.model.load_state_dict(state_dict)
    except Exception as e:
        print(f"❌ Load Error: {e}")
        return 0, 0
    
    # --- 4. Data ---
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    try:
        dataset = datasets.ImageFolder(root=test_dataset_path, transform=val_transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    except Exception as e:
        print(f"❌ Data Error: {e}")
        return 0, 0

    # --- 5. Inference ---
    y_true = []
    y_pred = []
    
    model.model.cuda()
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
            model.set_input(data)
            model.test()
            pred = model.output
            if pred.shape[1] == 1:
                prob = torch.sigmoid(pred).cpu().numpy().flatten()
            else:
                prob = torch.softmax(pred, dim=1)[:, 1].cpu().numpy()
            y_true.extend(data[1].cpu().numpy())
            y_pred.extend(prob)

    if len(np.unique(y_true)) < 2:
        return 0, 0
    else:
        mAP = average_precision_score(y_true, y_pred)
        y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
        acc = accuracy_score(y_true, y_pred_binary)
        print(f"   [Epoch {epoch_num}] mAP: {mAP:.4f} | Acc: {acc:.4f}")
        return mAP, acc

if __name__ == "__main__":
    TEST_PATH = "/root/autodl-tmp/datasets/CNNDetection/biggan"
    
    # 自动搜索所有 checkpoint
    ckpt_dir = './checkpoints/effort_universal_repro'
    files = os.listdir(ckpt_dir)
    epochs = []
    for f in files:
        if f.startswith('model_epoch_') and f.endswith('.pth') and 'init' not in f:
            try:
                ep = int(f.split('_')[-1].split('.')[0])
                epochs.append(ep)
            except: pass
    epochs.sort()
    
    print(f"🚀 Found epochs to test: {epochs}")
    
    results = {}
    for ep in epochs:
        mAP, acc = test_generalization(ep, TEST_PATH)
        results[ep] = mAP
        
    print(f"\n{'='*20} 🏆 Final Ranking 🏆 {'='*20}")
    sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for ep, score in sorted_res:
        prefix = "🥇 " if score == sorted_res[0][1] else f"   "
        print(f"{prefix}Epoch {ep}: mAP {score:.4f}")

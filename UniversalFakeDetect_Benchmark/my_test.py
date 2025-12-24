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
    print(f"\n{'='*20} Testing Epoch {epoch_num} on {dataset_name} {'='*20}")
    
    # --- 1. 伪造参数 (Mock Options) ---
    # 【关键修复】必须手动指定和训练时一样的参数！
    opt = TrainOptions().parse(print_options=False)
    opt.isTrain = False
    opt.gpu_ids = [0]
    opt.name = 'effort_universal_repro'
    opt.checkpoints_dir = './checkpoints'
    
    # [核心修复点] 强制指定模型架构
    opt.arch = 'CLIP:ViT-L/14_svd' 
    opt.fix_backbone = True
    opt.noise_std = 0.02
    
    # --- 2. 初始化模型 ---
    try:
        model = Trainer(opt)
        model.eval()
    except Exception as e:
        print(f"❌ Model Init Error: {e}")
        return
    
    # --- 3. 手动加载权重 ---
    ckpt_path = os.path.join(opt.checkpoints_dir, opt.name, f'model_epoch_{epoch_num}.pth')
    print(f"⚡️ Loading weights from: {ckpt_path}")
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: File not found: {ckpt_path}")
        return

    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu') # 先加载到CPU防爆显存
        
        # 自动拆包逻辑
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # 加载到模型
        if hasattr(model.model, "module"):
            model.model.module.load_state_dict(state_dict)
        else:
            model.model.load_state_dict(state_dict)
        print("✅ Weights loaded!")
    except Exception as e:
        print(f"❌ Load Error: {e}")
        return
    
    # --- 4. 准备数据 (Standard Loader) ---
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    try:
        dataset = datasets.ImageFolder(root=test_dataset_path, transform=val_transform)
        # BigGAN可能没有 0_real/1_fake 结构，这里要做个兼容性检查
        # 如果是按类别分的(bedroom, cat...)，我们需要把它们都当做 Fake (因为这是BigGAN生成的)
        # 但为了简单，我们先假设目录结构是标准的。如果报错我们再改。
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        print(f"✅ Indexed {len(dataset)} images.")
    except Exception as e:
        print(f"❌ Data Error: {e}")
        return

    # --- 5. 开始推理 ---
    y_true = []
    y_pred = []
    
    # 将模型移动到GPU
    model.model.cuda()
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            model.set_input(data)
            model.test()
            
            # 获取预测结果
            pred = model.output
            
            # 取 "Fake" 类的概率
            if pred.shape[1] == 1:
                prob = torch.sigmoid(pred).cpu().numpy().flatten()
            else:
                prob = torch.softmax(pred, dim=1)[:, 1].cpu().numpy()
            
            label = data[1].cpu().numpy()
            
            y_true.extend(label)
            y_pred.extend(prob)

    # --- 6. 计算指标 ---
    if len(np.unique(y_true)) < 2:
        print("⚠️ Warning: Only one class detected in test set. mAP might be undefined.")
        # 如果只有一类，我们只打印平均预测分
        print(f"   Avg Prediction Score: {np.mean(y_pred):.4f}")
    else:
        mAP = average_precision_score(y_true, y_pred)
        y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
        acc = accuracy_score(y_true, y_pred_binary)
        
        print(f"\n🏆 Result for Epoch {epoch_num} on {dataset_name}:")
        print(f"   mAP: {mAP:.4f}")
        print(f"   Acc: {acc:.4f}")
    print("="*60)

if __name__ == "__main__":
    # 请确认这个路径下确实有图片，且最好包含 0_real 和 1_fake
    # 如果 BigGAN 只有假图，mAP 无法计算，只能看预测概率
    # 为了严谨，建议测试包含真假图的数据集

    # 自动搜索 checkpoints 目录下的所有 epoch
    print(f"\n🚀 Scanning all checkpoints in ./checkpoints/effort_universal_repro ...")
    
    # 找到所有的 model_epoch_X.pth 文件
    ckpt_dir = os.path.join('./checkpoints', 'effort_universal_repro')
    files = os.listdir(ckpt_dir)
    epochs = []
    for f in files:
        if f.startswith('model_epoch_') and f.endswith('.pth') and 'init' not in f:
            # 提取数字
            try:
                ep = int(f.split('_')[-1].split('.')[0])
                epochs.append(ep)
            except:
                pass
    
    # 按从小到大排序
    epochs.sort()
    print(f"📋 Found epochs: {epochs}")

    TEST_PATH = "/root/autodl-tmp/datasets/CNNDetection/biggan" 
    
    # 循环测试所有 Epoch
    results = {}
    for ep in epochs:
        # 为了防止内存泄漏，这里只做简单的调用。
        # 如果显存不够，可能需要把 test_generalization 里的 model 初始化移到外面。
        # 但考虑到只是测试，应该问题不大。
        mAP, acc = test_generalization(ep, TEST_PATH)
        results[ep] = mAP

    # 打印最终排名
    print(f"\n{'='*20} Final Ranking (mAP) {'='*20}")
    sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for ep, score in sorted_res:
        print(f"Epoch {ep}: mAP {score:.4f} {'🥇' if score == sorted_res[0][1] else ''}")

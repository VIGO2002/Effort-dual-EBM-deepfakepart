import os
import time
import random
from tensorboardX import SummaryWriter

from validate import validate, find_best_threshold, RealFakeDataset
from data import create_dataloader
from earlystop import EarlyStopping
from models.trainer import Trainer
from options.train_options import TrainOptions
from dataset_paths import DATASET_PATHS
import torch
import numpy as np


SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'

    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    val_opt = get_val_opt()
    
    set_seed()
 
    model = Trainer(opt)
    # --- [Auto-Patch] 断点续训 + 自动拆包逻辑 (修复版) ---
    if opt.continue_train:
        print(f"🔄 Resuming training from epoch {opt.epoch_count}...")
        try:
            last_epoch = opt.epoch_count - 1
            load_path = os.path.join(opt.checkpoints_dir, opt.name, f"model_epoch_{last_epoch}.pth")
            print(f"⚡️ Force loading weights from: {load_path}")
            
            if os.path.exists(load_path):
                checkpoint = torch.load(load_path)
                
                # 自动拆包逻辑 (处理包含了 optimizer 的情况)
                if 'model' in checkpoint:
                    print("📦 Detected checkpoint dictionary, extracting model keys...")
                    state_dict = checkpoint['model']
                    if 'total_steps' in checkpoint:
                        model.total_steps = checkpoint['total_steps']
                else:
                    state_dict = checkpoint
                
                # 加载权重
                if hasattr(model.model, "module"):
                    model.model.module.load_state_dict(state_dict)
                else:
                    model.model.load_state_dict(state_dict)
                print("✅ Weights loaded successfully!")
            else:
                print(f"❌ Error: Checkpoint file not found at {load_path}")
                exit()
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            exit()
    # ----------------------------------------------------
    
    data_loader = create_dataloader(opt)
    # --- [Auto-Patch] 强力替换：使用标准 ImageFolder 加载验证集 ---
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader
    
    print("🛡️ Switching to Standard ImageFolder for Validation...")
    # 1. 定义 CLIP 标准预处理 (Resize -> Crop -> Tensor -> Normalize)
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    # 2. 锁定路径 (指向 merged_dataset/val)
    val_root = os.path.join(opt.wang2020_data_path, 'val')
    if not os.path.exists(val_root):
        print(f"❌ Critical Error: Validation path not found: {val_root}")
        # 尝试回退查找
        val_root = opt.wang2020_data_path
    
    # 3. 创建数据集
    try:
        val_dataset = datasets.ImageFolder(root=val_root, transform=val_transform)
        print(f"✅ [Standard Loader] Successfully indexed {len(val_dataset)} validation images!")
        print(f"   Classes found: {val_dataset.classes}")
        # 4. 创建 Loader
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
    except Exception as e:
        print(f"⚠️ Standard Loader Failed: {e}")
        print("⚠️ Fallback to original loader...")
        val_loader = create_dataloader(val_opt)
    # ----------------------------------------------------------------

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
        
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    start_time = time.time()
    print ("Length of data loader: %d" %(len(data_loader)))
    with open( os.path.join(opt.checkpoints_dir, opt.name,'log.txt'), 'a') as f:
        f.write("Length of data loader: %d \n" %(len(data_loader)) )
    for epoch in range(opt.epoch_count, opt.niter + opt.epoch_count):
        model.save_networks( 'model_epoch_init.pth' )
        
        for i, data in enumerate(data_loader):
            model.total_steps += 1

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                # 【新增】获取当前学习率 (防止 NameError)
                current_lr = model.optimizer.param_groups[0]['lr']
                
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)
                print("Iter time: ", ((time.time()-start_time)/model.total_steps) )
                
                # 现在可以使用 current_lr 了
                with open( os.path.join(opt.checkpoints_dir, opt.name,'log.txt'), 'a') as f:
                    f.write(f"Iter time: {(time.time()-start_time)/model.total_steps}, Lr: {current_lr:.6f}, Train loss: {model.loss} at step: {model.total_steps}\n")
                # ... (后面的代码)

            if model.total_steps in []: # save models at these iters 
                model.train()
                model.save_networks('model_iters_%s.pth' % model.total_steps)
            
            # if model.total_steps % 500 == 0:
            #     model.adjust_learning_rate()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            model.train()
            model.save_networks( 'model_epoch_%s.pth' % epoch )

        # Validation
        model.eval()
        ap, r_acc, f_acc, acc = validate(model.model, val_loader)
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

        model.train()


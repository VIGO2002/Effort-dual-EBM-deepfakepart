import os

# ==========================================
# 1. 基础路径定义
# ==========================================
GAN_ROOT = '/root/autodl-tmp/datasets/CNNDetection' 
DM_ROOT = '/root/autodl-tmp/datasets/Diffusion'

# ==========================================
# 2. 共享真图路径
# ==========================================
SHARED_REAL_PATH = os.path.join(GAN_ROOT, 'progan/0_real')

# ⚠️ 注意：这里的变量名必须叫 DATASET_PATHS，不能改名
DATASET_PATHS = [
    # ==========================================
    # GANs 测试集
    # ==========================================
    dict(
        real_path=os.path.join(GAN_ROOT, 'progan/0_real'),
        fake_path=os.path.join(GAN_ROOT, 'progan/1_fake'),
        data_mode='wang2020',
        key='progan'
    ),
    dict(
        real_path=os.path.join(GAN_ROOT, 'cyclegan/0_real'),
        fake_path=os.path.join(GAN_ROOT, 'cyclegan/1_fake'),
        data_mode='wang2020',
        key='cyclegan'
    ),
    dict(
        real_path=os.path.join(GAN_ROOT, 'biggan/0_real'),
        fake_path=os.path.join(GAN_ROOT, 'biggan/1_fake'),
        data_mode='wang2020',
        key='biggan'
    ),
    dict(
        real_path=os.path.join(GAN_ROOT, 'stylegan/0_real'),
        fake_path=os.path.join(GAN_ROOT, 'stylegan/1_fake'),
        data_mode='wang2020',
        key='stylegan'
    ),
    dict(
        real_path=os.path.join(GAN_ROOT, 'stylegan2/0_real'),
        fake_path=os.path.join(GAN_ROOT, 'stylegan2/1_fake'),
        data_mode='wang2020',
        key='stylegan2'
    ),
    dict(
        real_path=os.path.join(GAN_ROOT, 'gaugan/0_real'),
        fake_path=os.path.join(GAN_ROOT, 'gaugan/1_fake'),
        data_mode='wang2020',
        key='gaugan'
    ),
    dict(
        real_path=os.path.join(GAN_ROOT, 'stargan/0_real'),
        fake_path=os.path.join(GAN_ROOT, 'stargan/1_fake'),
        data_mode='wang2020',
        key='stargan'
    ),
    dict(
        real_path=os.path.join(GAN_ROOT, 'deepfake/0_real'),
        fake_path=os.path.join(GAN_ROOT, 'deepfake/1_fake'),
        data_mode='wang2020',
        key='deepfake'
    ),

    # ==========================================
    # Diffusion Models 测试集
    # ==========================================
    dict(
        real_path=SHARED_REAL_PATH,
        fake_path=os.path.join(DM_ROOT, 'ldm_100'),
        data_mode='wang2020',
        key='ldm'
    ),
    dict(
        real_path=SHARED_REAL_PATH,
        fake_path=os.path.join(DM_ROOT, 'glide_100_27'),
        data_mode='wang2020',
        key='glide'
    ),
    dict(
        real_path=SHARED_REAL_PATH,
        fake_path=os.path.join(DM_ROOT, 'dalle'), 
        data_mode='wang2020',
        key='dalle'
    ),
    dict(
        real_path=SHARED_REAL_PATH,
        fake_path=os.path.join(DM_ROOT, 'guided'),
        data_mode='wang2020',
        key='guided'
    ),
    dict(
        real_path=SHARED_REAL_PATH,
        fake_path=os.path.join(DM_ROOT, 'pndm'),
        data_mode='wang2020',
        key='pndm'
    ),
    dict(
        real_path=SHARED_REAL_PATH,
        fake_path=os.path.join(DM_ROOT, 'vqdiffusion'),
        data_mode='wang2020',
        key='vqdiffusion'
    ),
]
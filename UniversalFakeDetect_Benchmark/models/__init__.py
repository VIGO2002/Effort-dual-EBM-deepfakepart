from .clip_models import ClipModel


VALID_NAMES = {
                'CLIP:ViT-B/16_svd': 'openai/clip-vit-base-patch16',
                'CLIP:ViT-B/32_svd': 'openai/clip-vit-base-patch32',
                'CLIP:ViT-L/14_svd': 'openai/clip-vit-large-patch14',  # <--- 主要是这一行
                'SigLIP:ViT-L/16_256_svd': 'google/siglip-large-patch16-256',
                # BEiT-v2 如果你也用的话，可能需要找一下对应的 HF ID，通常是 'microsoft/beit-v2-large-patch16-224'
                'BEiTv2:ViT-L/16_svd': 'microsoft/beit-v2-large-patch16-224',
}


def get_model(name, opt):
    assert name in VALID_NAMES.keys()
    if name.startswith("CLIP:"):
        return ClipModel(VALID_NAMES[name], opt)
    else:
        assert False 

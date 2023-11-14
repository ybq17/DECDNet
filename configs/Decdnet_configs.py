import ml_collections
import os
import wget
os.makedirs('./weights', exist_ok=True)

def decdnet_configs():
    cfg = ml_collections.ConfigDict()
    cfg.swin_pyramid_fm = [96, 192, 384,768]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9
    if not os.path.isfile('./weights/swin_tiny_patch4_window7_224.pth'):  #预训练权重
        print('Downloading Swin-transformer model ...')
        wget.download("https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth", "./weights/swin_tiny_patch4_window7_224.pth")
    cfg.swin_pretrained_path = './weights/swin_tiny_patch4_window7_224.pth'
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm  = [64, 128, 256,512] # 四层resnet进行并行训练
    cfg.resnet_pretrained = True
    return cfg
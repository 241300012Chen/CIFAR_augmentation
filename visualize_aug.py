import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import os

IMAGE_INDEX = 12 
SEED = 2024

def get_vis_transform(mode):
    if mode == 'Baseline':
        return T.Compose([T.ToTensor()])
    elif mode == 'Geometric':
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(p=1.0), 
            T.ToTensor()
        ])
    elif mode == 'Photometric':
        return T.Compose([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            T.ToTensor()
        ])
    elif mode == 'Composite':
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(p=1.0),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            T.ToTensor()
        ])

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data_root = os.path.join(os.path.dirname(__file__), "data")
    save_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(save_dir, exist_ok=True)

    dataset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True)
    img_pil, label = dataset[IMAGE_INDEX]
    class_names = dataset.classes

    print(f"选中图片: 索引 {IMAGE_INDEX}, 类别: {class_names[label]}")

    modes = ['Baseline', 'Geometric', 'Photometric', 'Composite']
    titles = ['Baseline\n(Original)', 'Geometric\n(Crop & Flip)', 'Photometric\n(Color Jitter)', 'Composite\n(Ours)']
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

    for ax, mode, title in zip(axes, modes, titles):
        transform = get_vis_transform(mode)
        img_tensor = transform(img_pil)
        
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        ax.imshow(img_np, interpolation='nearest')

        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.axis('off')

    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "method_viz_hd.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"高清大图已保存: {save_path}")
    plt.show()

if __name__ == '__main__':
    main()
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

plt.style.use('ggplot')

def plot_and_summarize():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(current_dir, 'results')
    fig_dir = os.path.join(current_dir, 'figures')
    
    os.makedirs(fig_dir, exist_ok=True)
    
    print(f"正在读取数据目录: {res_dir}")
    experiments = {
        'baseline': {'label': 'Baseline', 'color': '#e74c3c'},       
        'photometric': {'label': 'Photometric', 'color': '#f39c12'},  
        'geometric': {'label': 'Geometric', 'color': '#3498db'},     
        'composite': {'label': 'Composite (Ours)', 'color': '#2ecc71'} 
    }
    
    fractions = [0.1, 0.2]
    
    final_stats = {exp: {} for exp in experiments}

    for frac in fractions:
        frac_tag = int(frac * 100) 
        frac_str = f"{frac_tag}%"
        
        print(f"\n>>> 正在处理 {frac_str} 数据 (Tag: frac{frac_tag})...")
        
        fig_acc, ax_acc = plt.subplots(figsize=(8, 6))
        fig_gap, ax_gap = plt.subplots(figsize=(8, 6))
        
        for exp_name, style in experiments.items():
            pattern = os.path.join(res_dir, f"{exp_name}_frac{frac_tag}_*.csv")
             
            
            files = glob.glob(pattern)
            
            if not files:
                print(f"  [Error] 没找到 {exp_name} 的文件！")
                print(f"  请检查 results 目录下有没有以 {exp_name}_frac{frac_tag}_ 开头的csv文件")
                continue
            else:
                print(f"  - {exp_name}: 找到 {len(files)} 个文件 (Seeds)")
            
            dfs = [pd.read_csv(f) for f in files]
            combined = pd.concat(dfs)
            grouped = combined.groupby('Epoch')
            
            mean_df = grouped.mean()
            std_df = grouped.std()
            epochs = mean_df.index
            
            mean_acc = mean_df['Test_Acc'] * 100
            std_acc = std_df['Test_Acc'] * 100
            
            ax_acc.plot(epochs, mean_acc, label=style['label'], color=style['color'], linewidth=2.5 if exp_name=='composite' else 1.5)
            ax_acc.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, color=style['color'], alpha=0.15)
            
            mean_gap = mean_df['Gap'] * 100
            std_gap = std_df['Gap'] * 100
            
            ax_gap.plot(epochs, mean_gap, label=style['label'], color=style['color'], linewidth=2.5 if exp_name=='composite' else 1.5)
            ax_gap.fill_between(epochs, mean_gap - std_gap, mean_gap + std_gap, color=style['color'], alpha=0.15)
            
            final_stats[exp_name][frac] = {
                'acc_mean': mean_acc.iloc[-1],
                'acc_std': std_acc.iloc[-1],
                'gap_mean': mean_gap.iloc[-1],
                'gap_std': std_gap.iloc[-1]
            }

        ax_acc.set_title(f'Test Accuracy (CIFAR-10 {frac_str} Data)', fontsize=14)
        ax_acc.set_xlabel('Epoch', fontsize=12)
        ax_acc.set_ylabel('Accuracy (%)', fontsize=12)
        ax_acc.legend(loc='lower right')
        ax_acc.grid(True, linestyle='--', alpha=0.7)
        fig_acc.savefig(os.path.join(fig_dir, f'acc_frac{frac_tag}.png'), dpi=300)
        
        ax_gap.set_title(f'Generalization Gap ({frac_str} Data) - Lower is Better', fontsize=14)
        ax_gap.set_xlabel('Epoch', fontsize=12)
        ax_gap.set_ylabel('Gap (Train% - Test%)', fontsize=12)
        ax_gap.legend(loc='upper right')
        ax_gap.grid(True, linestyle='--', alpha=0.7)
        fig_gap.savefig(os.path.join(fig_dir, f'gap_frac{frac_tag}.png'), dpi=300)
        
        print(f"  -> 图片已保存至 figures/acc_frac{frac_tag}.png")

    print("\n" + "="*85)
    print(f"{'Method':<20} | {'10% Acc':<15} | {'10% Gap':<15} | {'20% Acc':<15} | {'20% Gap':<15}")
    print("-" * 85)
    
    for exp_name in experiments:
        row = final_stats[exp_name]
        
        acc10, gap10, acc20, gap20 = "N/A", "N/A", "N/A", "N/A"
        
        if 0.1 in row:
            acc10 = f"{row[0.1]['acc_mean']:.2f}±{row[0.1]['acc_std']:.2f}"
            gap10 = f"{row[0.1]['gap_mean']:.2f}±{row[0.1]['gap_std']:.2f}"
        if 0.2 in row:
            acc20 = f"{row[0.2]['acc_mean']:.2f}±{row[0.2]['acc_std']:.2f}"
            gap20 = f"{row[0.2]['gap_mean']:.2f}±{row[0.2]['gap_std']:.2f}"
        
        print(f"{experiments[exp_name]['label']:<20} | {acc10:<15} | {gap10:<15} | {acc20:<15} | {gap20:<15}")
    print("="*85 + "\n")

if __name__ == '__main__':
    plot_and_summarize()
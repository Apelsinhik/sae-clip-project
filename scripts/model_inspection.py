# model_inspection.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def inspect_sae_models():
    """Детальная проверка обученных SAE моделей"""
    
    base_path = "/content/drive/MyDrive/SAE_PROJECT/models"
    results_dir = "/content/drive/MyDrive/SAE_PROJECT/results/model_inspection"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    datasets = ["cifar", "food101"]
    analysis_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"ANALYZING {dataset.upper()} SAE")
        print('='*60)
        
        model_path = f"{base_path}/sae_{dataset}_512d/sae_epoch_50.pt"
        
        if not Path(model_path).exists():
            print(f"⚠️  Model not found: {model_path}")
            continue
        
        # Загружаем модель
        state_dict = torch.load(model_path, map_location='cpu')
        
        analysis_results[dataset] = {
            'model_path': model_path,
            'state_dict_keys': list(state_dict.keys())
        }
        
        # 1. Проверка размерностей
        print("\n[1] DIMENSIONS CHECK:")
        encoder_weight = state_dict['encoder.weight']
        decoder_weight = state_dict['decoder.weight']
        encoder_bias = state_dict['encoder.bias']
        decoder_bias = state_dict['decoder.bias']
        
        print(f"  Encoder weight: {encoder_weight.shape} (expected: [4096, 512])")
        print(f"  Decoder weight: {decoder_weight.shape} (expected: [512, 4096])")
        print(f"  Encoder bias:   {encoder_bias.shape} (expected: [4096])")
        print(f"  Decoder bias:   {decoder_bias.shape} (expected: [512])")
        
        # Проверка совместимости
        assert encoder_weight.shape == (4096, 512), f"Encoder weight shape mismatch: {encoder_weight.shape}"
        assert decoder_weight.shape == (512, 4096), f"Decoder weight shape mismatch: {decoder_weight.shape}"
        print("  ✅ Dimension check passed!")
        
        # 2. Статистика весов
        print("\n[2] WEIGHT STATISTICS:")
        
        stats = {}
        for name, tensor in state_dict.items():
            stats[name] = {
                'mean': tensor.mean().item(),
                'std': tensor.std().item(),
                'min': tensor.min().item(),
                'max': tensor.max().item(),
                'abs_mean': tensor.abs().mean().item()
            }
            print(f"  {name}: mean={stats[name]['mean']:.6f}, std={stats[name]['std']:.6f}, "
                  f"range=[{stats[name]['min']:.6f}, {stats[name]['max']:.6f}]")
        
        analysis_results[dataset]['weight_stats'] = stats
        
        # 3. Визуализация распределения весов
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (name, tensor) in enumerate(state_dict.items()):
            if idx >= 4:
                break
            
            ax = axes[idx]
            # Преобразуем в numpy для гистограммы
            weights = tensor.numpy().flatten()
            
            # Убираем выбросы для лучшей визуализации
            q_low, q_high = np.percentile(weights, [1, 99])
            weights_filtered = weights[(weights >= q_low) & (weights <= q_high)]
            
            ax.hist(weights_filtered, bins=100, alpha=0.7, density=True)
            ax.set_title(f'{name} Distribution\n'
                        f'μ={weights.mean():.4f}, σ={weights.std():.4f}')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{dataset.upper()} SAE - Weight Distributions', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/{dataset}_weight_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Проверка симметрии энкодера/декодера
        print("\n[3] ENCODER-DECODER SYMMETRY:")
        
        # Проверяем, являются ли веса декодера транспонированными весами энкодера
        encoder_norm = torch.norm(encoder_weight)
        decoder_norm = torch.norm(decoder_weight)
        encoder_decoder_corr = torch.corrcoef(
            torch.stack([encoder_weight.flatten(), decoder_weight.T.flatten()])
        )[0, 1].item()
        
        print(f"  Encoder norm: {encoder_norm:.4f}")
        print(f"  Decoder norm: {decoder_norm:.4f}")
        print(f"  Encoder-Decoder correlation: {encoder_decoder_corr:.4f}")
        
        # 5. Анализ разреженности
        print("\n[4] SPARSITY ANALYSIS (on random input):")
        
        # Генерируем случайный вход
        batch_size = 100
        random_input = torch.randn(batch_size, 512)
        
        # Создаем временную модель для теста
        class TempSAE(torch.nn.Module):
            def __init__(self, state_dict):
                super().__init__()
                self.encoder = torch.nn.Linear(512, 4096)
                self.decoder = torch.nn.Linear(4096, 512)
                self.load_state_dict(state_dict)
            
            def forward(self, x):
                z = torch.relu(self.encoder(x))
                x_hat = self.decoder(z)
                return x_hat, z
        
        temp_model = TempSAE(state_dict)
        temp_model.eval()
        
        with torch.no_grad():
            x_hat, z = temp_model(random_input)
            
            # Меры разреженности
            l0_sparsity = (z > 0).float().mean().item()  # % активных нейронов
            l1_sparsity = z.abs().mean().item()          # L1 норма
            
            reconstruction_error = torch.norm(random_input - x_hat, dim=1).mean().item()
            
            print(f"  L0 sparsity: {l0_sparsity:.4f} ({l0_sparsity*100:.1f}% активных нейронов)")
            print(f"  L1 sparsity: {l1_sparsity:.4f}")
            print(f"  Avg reconstruction error: {reconstruction_error:.6f}")
        
        analysis_results[dataset]['sparsity'] = {
            'l0': l0_sparsity,
            'l1': l1_sparsity,
            'reconstruction_error': reconstruction_error
        }
        
        # 6. Визуализация активаций
        print("\n[5] ACTIVATION VISUALIZATION:")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Случайные выборки для визуализации
        sample_indices = np.random.choice(batch_size, 6, replace=False)
        
        for idx, sample_idx in enumerate(sample_indices):
            ax = axes[idx // 3, idx % 3]
            
            # Активации для одного примера
            sample_z = z[sample_idx].numpy()
            
            # Сортируем по величине активации
            sorted_indices = np.argsort(sample_z)[::-1]
            sorted_activations = sample_z[sorted_indices]
            
            # Берем топ-100 активаций
            top_k = 100
            ax.bar(range(top_k), sorted_activations[:top_k])
            ax.set_title(f'Sample {sample_idx}: Top {top_k} Activations')
            ax.set_xlabel('Neuron Index (sorted)')
            ax.set_ylabel('Activation Value')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{dataset.upper()} SAE - Neuron Activations (Random Input)', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/{dataset}_neuron_activations.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Analysis complete. Plots saved to {results_dir}/")
    
    # 7. Сравнение двух моделей
    print(f"\n{'='*60}")
    print("COMPARISON BETWEEN MODELS")
    print('='*60)
    
    if 'cifar' in analysis_results and 'food101' in analysis_results:
        # Создаем сравнительную таблицу
        comparison_data = []
        
        for dataset in ['cifar', 'food101']:
            data = analysis_results[dataset]
            stats = data['weight_stats']
            sparsity = data['sparsity']
            
            comparison_data.append({
                'Dataset': dataset.upper(),
                'Encoder Mean': f"{stats['encoder.weight']['mean']:.6f}",
                'Encoder Std': f"{stats['encoder.weight']['std']:.6f}",
                'Decoder Mean': f"{stats['decoder.weight']['mean']:.6f}",
                'Decoder Std': f"{stats['decoder.weight']['std']:.6f}",
                'L0 Sparsity': f"{sparsity['l0']:.4f}",
                'L1 Sparsity': f"{sparsity['l1']:.4f}",
                'Recon Error': f"{sparsity['reconstruction_error']:.6f}"
            })
        
        # Выводим таблицу
        print("\nComparison Table:")
        print("-" * 100)
        headers = list(comparison_data[0].keys())
        print("| " + " | ".join(headers) + " |")
        print("|" + "-"*100 + "|")
        
        for row in comparison_data:
            print("| " + " | ".join(str(row[h]) for h in headers) + " |")
        
        # Создаем сравнительный график
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        metrics = ['L0 Sparsity', 'L1 Sparsity', 'Recon Error']
        datasets = ['CIFAR-10', 'Food-101']
        
        for idx, metric in enumerate(metrics):
            ax = axes[0, idx]
            
            values = [
                float(comparison_data[0][metric]),
                float(comparison_data[1][metric])
            ]
            
            bars = ax.bar(datasets, values, alpha=0.7)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            
            # Добавляем значения на столбцы
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom')
        
        # Распределение весов
        for idx, dataset in enumerate(['cifar', 'food101']):
            ax = axes[1, idx]
            
            encoder_weights = analysis_results[dataset]['weight_stats']['encoder.weight']
            decoder_weights = analysis_results[dataset]['weight_stats']['decoder.weight']
            
            x = ['Encoder', 'Decoder']
            means = [encoder_weights['mean'], decoder_weights['mean']]
            stds = [encoder_weights['std'], decoder_weights['std']]
            
            bars = ax.bar(x, means, yerr=stds, alpha=0.7, capsize=5)
            ax.set_title(f'{dataset.upper()} - Weight Stats')
            ax.set_ylabel('Mean Weight Value')
            ax.grid(True, alpha=0.3)
            
            # Добавляем значения
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.4f}±{std:.4f}', ha='center', va='bottom')
        
        axes[1, 2].axis('off')
        
        plt.suptitle('SAE Models Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Comparison plots saved to: {results_dir}/model_comparison.png")
    
    # 8. Сохраняем результаты анализа
    import json
    import pandas as pd
    
    # Сохраняем JSON
    with open(f'{results_dir}/model_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    # Сохраняем CSV
    if 'cifar' in analysis_results and 'food101' in analysis_results:
        df = pd.DataFrame(comparison_data)
        df.to_csv(f'{results_dir}/model_comparison.csv', index=False)
    
    print(f"\n📁 All analysis results saved to: {results_dir}")
    print("   - model_analysis.json")
    print("   - model_comparison.csv")
    print("   - *_weight_distributions.png")
    print("   - *_neuron_activations.png")
    print("   - model_comparison.png")
    
    return analysis_results

if __name__ == "__main__":
    results = inspect_sae_models()
# evaluate_and_save.py
import os
import json
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from sae_clip.models.clip_wrapper import CLIPWrapper
from sae_clip.models.sae import SparseAutoencoder
from sae_clip.data.datasets import (
    CIFAR10ZeroShotDataset,
    Food101ZeroShotDataset,
)

def pil_collate_fn(batch):
    images, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return list(images), labels

@torch.no_grad()
def zero_shot_accuracy(clip_model, dataset, classnames, sae=None, batch_size=256):
    device = clip_model.device
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pil_collate_fn,
        num_workers=2,
    )

    prompts = [f"a photo of {c.replace('_', ' ')}" for c in classnames]
    text_emb = clip_model.encode_text(prompts).to(device)

    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_confidences = []

    for images, labels in tqdm(loader, desc="Evaluation"):
        labels = labels.to(device)
        img_emb = clip_model.encode_images(images)

        if sae is not None:
            img_emb, _ = sae(img_emb)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        logits = img_emb @ text_emb.T
        confidences = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "predictions": np.array(all_predictions),
        "labels": np.array(all_labels),
        "confidences": np.array(all_confidences),
        "total_samples": total,
        "correct": correct
    }

def create_visualizations(results, save_dir):
    """Создание графиков и визуализаций"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Bar plot сравнения точности
    plt.figure(figsize=(10, 6))
    
    datasets = []
    clip_accuracies = []
    sae_accuracies = []
    
    for dataset_name in ["CIFAR-10", "Food-101"]:
        if f"{dataset_name}_CLIP" in results:
            datasets.append(dataset_name)
            clip_accuracies.append(results[f"{dataset_name}_CLIP"]["accuracy"])
            sae_accuracies.append(results[f"{dataset_name}_SAE"]["accuracy"])
    
    x = np.arange(len(datasets))
    width = 0.35
    
    plt.bar(x - width/2, clip_accuracies, width, label='CLIP', alpha=0.8)
    plt.bar(x + width/2, sae_accuracies, width, label='CLIP+SAE', alpha=0.8)
    
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Zero-shot Accuracy: CLIP vs CLIP+SAE')
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Добавление значений на столбцы
    for i, (clip_acc, sae_acc) in enumerate(zip(clip_accuracies, sae_accuracies)):
        plt.text(i - width/2, clip_acc + 0.01, f'{clip_acc:.3f}', ha='center')
        plt.text(i + width/2, sae_acc + 0.01, f'{sae_acc:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'), dpi=150)
    plt.close()
    
    # 2. Confusion matrices (только для CIFAR-10)
    if "CIFAR-10_SAE" in results:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        cifar_classes = ["airplane", "automobile", "bird", "cat", "deer",
                        "dog", "frog", "horse", "ship", "truck"]
        
        # Confusion matrix для CLIP
        cm_clip = confusion_matrix(
            results["CIFAR-10_CLIP"]["labels"],
            results["CIFAR-10_CLIP"]["predictions"]
        )
        
        sns.heatmap(cm_clip, annot=True, fmt='d', cmap='Blues',
                   xticklabels=cifar_classes, yticklabels=cifar_classes,
                   ax=axes[0])
        axes[0].set_title('CLIP Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        
        # Confusion matrix для CLIP+SAE
        cm_sae = confusion_matrix(
            results["CIFAR-10_SAE"]["labels"],
            results["CIFAR-10_SAE"]["predictions"]
        )
        
        sns.heatmap(cm_sae, annot=True, fmt='d', cmap='Greens',
                   xticklabels=cifar_classes, yticklabels=cifar_classes,
                   ax=axes[1])
        axes[1].set_title('CLIP+SAE Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'), dpi=150)
        plt.close()
    
    # 3. Confidence distribution
    if "CIFAR-10_SAE" in results:
        plt.figure(figsize=(12, 5))
        
        # Correct predictions confidence
        correct_clip = results["CIFAR-10_CLIP"]["confidences"][
            np.arange(len(results["CIFAR-10_CLIP"]["predictions"])),
            results["CIFAR-10_CLIP"]["predictions"]
        ]
        correct_sae = results["CIFAR-10_SAE"]["confidences"][
            np.arange(len(results["CIFAR-10_SAE"]["predictions"])),
            results["CIFAR-10_SAE"]["predictions"]
        ]
        
        plt.subplot(1, 2, 1)
        plt.hist(correct_clip, bins=30, alpha=0.5, label='CLIP', density=True)
        plt.hist(correct_sae, bins=30, alpha=0.5, label='CLIP+SAE', density=True)
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Density')
        plt.title('Confidence Distribution (All Predictions)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Only correct predictions
        clip_correct_mask = results["CIFAR-10_CLIP"]["predictions"] == results["CIFAR-10_CLIP"]["labels"]
        sae_correct_mask = results["CIFAR-10_SAE"]["predictions"] == results["CIFAR-10_SAE"]["labels"]
        
        correct_clip_conf = correct_clip[clip_correct_mask]
        correct_sae_conf = correct_sae[sae_correct_mask]
        
        plt.subplot(1, 2, 2)
        plt.hist(correct_clip_conf, bins=30, alpha=0.5, label='CLIP', density=True)
        plt.hist(correct_sae_conf, bins=30, alpha=0.5, label='CLIP+SAE', density=True)
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Density')
        plt.title('Confidence Distribution (Correct Predictions Only)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confidence_distribution.png'), dpi=150)
        plt.close()

def save_detailed_results(results, save_dir):
    """Сохранение детальных результатов"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. JSON с полными результатами
    json_results = {}
    for key, value in results.items():
        if key.endswith("_CLIP") or key.endswith("_SAE"):
            json_results[key] = {
                "accuracy": value["accuracy"],
                "total_samples": value["total_samples"],
                "correct": value["correct"]
            }
    
    json_path = os.path.join(save_dir, "detailed_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    
    # 2. CSV с основными метриками
    csv_path = os.path.join(save_dir, "metrics_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "accuracy", "improvement", "samples"])
        writer.writeheader()
        
        for dataset_name in ["CIFAR-10", "Food-101"]:
            clip_key = f"{dataset_name}_CLIP"
            sae_key = f"{dataset_name}_SAE"
            
            if clip_key in results and sae_key in results:
                clip_acc = results[clip_key]["accuracy"]
                sae_acc = results[sae_key]["accuracy"]
                improvement = sae_acc - clip_acc
                
                writer.writerow({
                    "dataset": dataset_name,
                    "model": "CLIP",
                    "accuracy": f"{clip_acc:.4f}",
                    "improvement": "0.0000",
                    "samples": results[clip_key]["total_samples"]
                })
                
                writer.writerow({
                    "dataset": dataset_name,
                    "model": "CLIP+SAE",
                    "accuracy": f"{sae_acc:.4f}",
                    "improvement": f"{improvement:.4f}",
                    "samples": results[sae_key]["total_samples"]
                })
    
    # 3. Markdown отчет
    md_path = os.path.join(save_dir, "RESULTS.md")
    with open(md_path, "w") as f:
        f.write("# SAE Evaluation Results\n\n")
        f.write(f"**Evaluation Date:** {os.path.basename(save_dir)}\n\n")
        
        f.write("## Summary\n\n")
        f.write("| Dataset | CLIP Accuracy | CLIP+SAE Accuracy | Improvement |\n")
        f.write("|---------|---------------|-------------------|-------------|\n")
        
        for dataset_name in ["CIFAR-10", "Food-101"]:
            clip_key = f"{dataset_name}_CLIP"
            sae_key = f"{dataset_name}_SAE"
            
            if clip_key in results and sae_key in results:
                clip_acc = results[clip_key]["accuracy"]
                sae_acc = results[sae_key]["accuracy"]
                improvement = sae_acc - clip_acc
                improvement_pct = (improvement / clip_acc * 100) if clip_acc > 0 else 0
                
                f.write(f"| {dataset_name} | {clip_acc:.4f} | {sae_acc:.4f} | {improvement:+.4f} ({improvement_pct:+.1f}%) |\n")
        
        f.write("\n## Visualizations\n\n")
        f.write("Generated plots:\n")
        f.write("- `accuracy_comparison.png` - Bar plot comparing CLIP vs CLIP+SAE\n")
        if "CIFAR-10_SAE" in results:
            f.write("- `confusion_matrices.png` - Confusion matrices for CIFAR-10\n")
            f.write("- `confidence_distribution.png` - Confidence distribution analysis\n")
        
        f.write("\n## SAE Models Used\n\n")
        f.write("| Dataset | SAE Path | Input Dim | Latent Dim |\n")
        f.write("|---------|----------|-----------|------------|\n")
        f.write("| CIFAR-10 | `/content/drive/MyDrive/SAE_PROJECT/models/sae_cifar_512d/sae_epoch_50.pt` | 512 | 4096 |\n")
        f.write("| Food-101 | `/content/drive/MyDrive/SAE_PROJECT/models/sae_food101_512d/sae_epoch_50.pt` | 512 | 4096 |\n")
    
    print(f"\n📊 Results saved to: {save_dir}")
    print(f"   JSON: {json_path}")
    print(f"   CSV:  {csv_path}")
    print(f"   MD:   {md_path}")

def main():
    import datetime
    
    # Создаем папку для результатов с timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"/content/drive/MyDrive/SAE_PROJECT/results/experiment_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"📁 Saving results to: {save_dir}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    
    # ==================== ЗАГРУЗКА CLIP ====================
    print("\n[1] Loading CLIP...")
    clip_model = CLIPWrapper(device=device)
    clip_model.eval()
    
    # ==================== CIFAR-10 ====================
    print("\n" + "="*50)
    print("CIFAR-10 EVALUATION")
    print("="*50)
    
    cifar_classes = ["airplane", "automobile", "bird", "cat", "deer",
                     "dog", "frog", "horse", "ship", "truck"]
    
    cifar = CIFAR10ZeroShotDataset(
        root="/content/yadisk/SAE_PROJECT/datasets/cifar-10-batches-py",
        train=False,
    )
    
    # CLIP baseline
    print("\n[1/2] Testing CLIP baseline...")
    results["CIFAR-10_CLIP"] = zero_shot_accuracy(
        clip_model, cifar, cifar_classes, sae=None, batch_size=64
    )
    print(f"  CLIP accuracy: {results['CIFAR-10_CLIP']['accuracy']:.4f}")
    
    # CLIP + SAE
    sae_cifar_path = "/content/drive/MyDrive/SAE_PROJECT/models/sae_cifar_512d/sae_epoch_50.pt"
    if os.path.exists(sae_cifar_path):
        print("\n[2/2] Testing CLIP + SAE (CIFAR-10)...")
        
        sae = SparseAutoencoder(input_dim=512, latent_dim=4096)
        sae.load_state_dict(torch.load(sae_cifar_path, map_location=device))
        sae.to(device)
        sae.eval()
        
        results["CIFAR-10_SAE"] = zero_shot_accuracy(
            clip_model, cifar, cifar_classes, sae=sae, batch_size=64
        )
        print(f"  CLIP+SAE accuracy: {results['CIFAR-10_SAE']['accuracy']:.4f}")
        
        diff = results['CIFAR-10_SAE']['accuracy'] - results['CIFAR-10_CLIP']['accuracy']
        if diff > 0:
            print(f"  ✅ Improvement: +{diff:.4f} (+{diff/results['CIFAR-10_CLIP']['accuracy']*100:.1f}%)")
        else:
            print(f"  ⚠️  Degradation: {diff:.4f} ({diff/results['CIFAR-10_CLIP']['accuracy']*100:.1f}%)")
    
    # ==================== FOOD-101 ====================
    print("\n" + "="*50)
    print("FOOD-101 EVALUATION (subset 2000)")
    print("="*50)
    
    food_full = Food101ZeroShotDataset(
        root="/content/yadisk/SAE_PROJECT/datasets",
        split="test",
    )
    food = Subset(food_full, range(2000))
    food_classes = food_full.classes
    
    # CLIP baseline
    print("\n[1/2] Testing CLIP baseline...")
    results["Food-101_CLIP"] = zero_shot_accuracy(
        clip_model, food, food_classes, sae=None, batch_size=64
    )
    print(f"  CLIP accuracy: {results['Food-101_CLIP']['accuracy']:.4f}")
    
    # CLIP + SAE
    sae_food_path = "/content/drive/MyDrive/SAE_PROJECT/models/sae_food101_512d/sae_epoch_50.pt"
    if os.path.exists(sae_food_path):
        print("\n[2/2] Testing CLIP + SAE (Food-101)...")
        
        sae = SparseAutoencoder(input_dim=512, latent_dim=4096)
        sae.load_state_dict(torch.load(sae_food_path, map_location=device))
        sae.to(device)
        sae.eval()
        
        results["Food-101_SAE"] = zero_shot_accuracy(
            clip_model, food, food_classes, sae=sae, batch_size=64
        )
        print(f"  CLIP+SAE accuracy: {results['Food-101_SAE']['accuracy']:.4f}")
        
        diff = results['Food-101_SAE']['accuracy'] - results['Food-101_CLIP']['accuracy']
        if diff > 0:
            print(f"  ✅ Improvement: +{diff:.4f} (+{diff/results['Food-101_CLIP']['accuracy']*100:.1f}%)")
        else:
            print(f"  ⚠️  Degradation: {diff:.4f} ({diff/results['Food-101_CLIP']['accuracy']*100:.1f}%)")
    
    # ==================== СОХРАНЕНИЕ ====================
    print("\n" + "="*50)
    print("SAVING RESULTS AND VISUALIZATIONS")
    print("="*50)
    
    # Сохраняем детальные результаты
    save_detailed_results(results, save_dir)
    
    # Создаем визуализации
    create_visualizations(results, save_dir)
    
    # ==================== ИТОГ ====================
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    
    for dataset in ["CIFAR-10", "Food-101"]:
        clip_key = f"{dataset}_CLIP"
        sae_key = f"{dataset}_SAE"
        
        if clip_key in results and sae_key in results:
            clip_acc = results[clip_key]["accuracy"]
            sae_acc = results[sae_key]["accuracy"]
            diff = sae_acc - clip_acc
            diff_pct = (diff / clip_acc * 100) if clip_acc > 0 else 0
            
            print(f"\n{dataset}:")
            print(f"  CLIP:       {clip_acc:.4f}")
            print(f"  CLIP+SAE:   {sae_acc:.4f}")
            print(f"  Difference: {diff:+.4f} ({diff_pct:+.1f}%)")
    
    print(f"\n📁 All results saved to: {save_dir}")
    print("📈 Visualizations generated:")
    print("   - accuracy_comparison.png")
    if "CIFAR-10_SAE" in results:
        print("   - confusion_matrices.png")
        print("   - confidence_distribution.png")

if __name__ == "__main__":
    main()
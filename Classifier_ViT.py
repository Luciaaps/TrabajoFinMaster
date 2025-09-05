# Librerías
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Solo usar GPU 0 para optimizar el rendimiento
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import shutil
import torch
from torchvision.datasets import ImageFolder
from datasets import Dataset, DatasetDict
import evaluate
from transformers import (
    AutoImageProcessor, ViTForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

set_seed(42)

# Configuración
model_checkpoint = 'google/vit-base-patch16-224-in21k'
TYPE = "_Balanced"
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
labels = ['inicial', 'avanzado']
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"PyTorch version: {torch.__version__}")

# Rutas de datos
base_path = 'CasosCáncer_241026-Separadas'
train_dir = os.path.join(base_path, 'train_original')
val_dir = os.path.join(base_path, 'val_original')
test_dir = os.path.join(base_path, 'test')

# Carga de imágenes
def load_image_data():
    transform = lambda img: image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
    train_data = ImageFolder(train_dir, transform=transform)
    val_data = ImageFolder(val_dir, transform=transform)
    test_data = ImageFolder(test_dir, transform=transform)

    def shuffle_samples(samples): return [samples[i] for i in np.random.permutation(len(samples))]

    return {
        'train': shuffle_samples(train_data.samples),
        'valid': shuffle_samples(val_data.samples),
        'test': shuffle_samples(test_data.samples)
    }

def create_my_dataset(image_samples):
    return Dataset.from_dict({
        'image': [img_path for img_path, _ in image_samples],
        'labels': [label for _, label in image_samples]
    })

def transform(example_batch):
    inputs = image_processor([Image.open(p).convert("RGB") for p in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['labels']
    return inputs

def prepare_datasets():
    data = load_image_data()
    dataset_dict = DatasetDict({
        split: create_my_dataset(data[split]) for split in ['train', 'valid', 'test']
    })
    return {
        split: dataset_dict[split].with_transform(transform) for split in dataset_dict
    }, dataset_dict

# Métricas
metric = evaluate.combine(["f1", "accuracy"])
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

# Colación
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

# Entrenamiento
def train_model(train_dataset, val_dataset):
    torch.cuda.empty_cache()
    model = ViTForImageClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2
    )

    training_args = TrainingArguments(
        output_dir=os.path.expanduser("~/ViTresults_user"),
        logging_dir=os.path.expanduser("~/ViT_tensorboard_user"),
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=1e-5,
        weight_decay=0.2,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=2,
        num_train_epochs=10,
        warmup_ratio=0.2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        max_grad_norm=1.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=image_processor
    )

    train_results = trainer.train()
    
    return trainer, model, train_results

# Evaluación
def evaluate_model(trainer, test_dataset, original_dataset, train_results=None, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # ======== Métricas de evaluación ========
    metrics = trainer.evaluate(test_dataset)

    # ======== Métricas del entrenamiento ========
    if train_results is not None:
        metrics.update(train_results.metrics)

    # ======== Métricas adicionales (custom) ========
    outputs = trainer.predict(test_dataset)
    y_pred = outputs.predictions.argmax(1)
    y_probs = outputs.predictions[:, 1]  # Para AUC, usamos probabilidad clase 1
    y_true = original_dataset['test']['labels']

    # Calcular F1 y accuracy
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Calcular precisión
    precision = precision_score(y_true, y_pred, average='weighted')

    # Matriz de confusión
    cmatrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cmatrix.ravel()

    # Calcular ROC y AUC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

     # ======== F1 en entrenamiento ========
    train_outputs = trainer.predict(trainer.train_dataset)
    y_train_pred = np.argmax(train_outputs.predictions, axis=1)
    y_train_true = original_dataset['train']['labels']
    f1_train = f1_score(y_train_true, y_train_pred, average='weighted')

    # Cálculo de métricas
    metrics.update({
        "f1_train": f1_train,
        "ROC AUC": roc_auc,
        "precision": precision,
        "recall/sensibilidad": tp / (tp + fn) if (tp + fn) else 0,
        "especificidad": tn / (tn + fp) if (tn + fp) else 0,
        "tasa_falsos_negativos": fn / (fn + tp) if (fn + tp) else 0,
        "VPP (Valor Predictivo Positivo)": tp / (tp + fp) if (tp + fp) else 0,
        "VPN (Valor Predictivo Negativo)": tn / (tn + fn) if (tn + fn) else 0,
        "f1_score": f1
    })

    # ======== Guardar JSON ========
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # ======== Reporte clásico ========
    report = classification_report(y_true, y_pred, digits=3, target_names=labels)
    print(report)
    with open(os.path.join(output_dir, "classification_report_test.txt"), "w") as f:
        f.write(report)

    # ======== Matriz de confusión visual ========
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cmatrix, interpolation='nearest', cmap=cm.Blues)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicciones')
    ax.set_ylabel('Etiquetas verdaderas')
    thresh = cmatrix.max() / 2.
    for i in range(cmatrix.shape[0]):
        for j in range(cmatrix.shape[1]):
            ax.text(j, i, format(cmatrix[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cmatrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # ======== Matriz de confusión normalizada ========
    cmatrix_normalized = cmatrix.astype('float') / cmatrix.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cmatrix_normalized, interpolation='nearest', cmap=cm.Blues)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicciones')
    ax.set_ylabel('Etiquetas verdaderas')
    thresh = cmatrix_normalized.max() / 2.
    for i in range(cmatrix_normalized.shape[0]):
        for j in range(cmatrix_normalized.shape[1]):
            ax.text(j, i, f"{cmatrix_normalized[i, j]:.2f}",
                            ha="center", va="center",
                            color="white" if cmatrix_normalized[i, j] > thresh else "black")
    plt.tight_layout()
    plt.colorbar(im)
    plt.savefig(os.path.join(output_dir, "confusion_matrix_normalized.png"))
    plt.close()

    # ======== Gráficas de entrenamiento ========
    if train_results is not None:
        log_history = trainer.state.log_history
        
        train_losses_by_epoch = {}
        eval_metrics_by_epoch = {}

        for log in log_history:
            epoch = int(log.get("epoch", -1)) 

            if epoch != -1: 
                if "loss" in log and "eval_loss" not in log: 
                    train_losses_by_epoch[epoch] = log["loss"]
                
                if "eval_loss" in log and "eval_f1" in log: 
                    eval_metrics_by_epoch[epoch] = {
                        "eval_loss": log["eval_loss"],
                        "eval_f1": log["eval_f1"]
                    }
        
        epochs_for_plot = sorted(eval_metrics_by_epoch.keys())
        
        final_train_loss = []
        final_eval_loss = []
        final_eval_f1 = []

        for epoch in epochs_for_plot:
            final_train_loss.append(train_losses_by_epoch[epoch])
            final_eval_loss.append(eval_metrics_by_epoch[epoch]["eval_loss"])
            final_eval_f1.append(eval_metrics_by_epoch[epoch]["eval_f1"])
                
        # Gráfica de pérdidas (train y val)
        plt.figure(figsize=(8, 5))
        plt.plot([e + 1 for e in epochs_for_plot], final_train_loss, 'b-', label='Entrenamiento')
        plt.plot([e + 1 for e in epochs_for_plot], final_eval_loss, 'r-', label='Validación')
        plt.title("Pérdidas de entrenamiento y validación")
        plt.xlabel("Épocas")
        plt.ylabel("Pérdidas")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "train_val_loss.png"))
        plt.close()

        # Gráfica de F1 en validación
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_for_plot, final_eval_f1, 'g-')
        plt.title("F1 Score en validación")
        plt.xlabel("Épocas")
        plt.ylabel("F1 Score")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "val_f1_score.png"))
        plt.close()

    # ======== Curva ROC y AUC ========
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

# Ejecución completa
def main():
    print("==> 📦 Preparando datasets...")
    processed_datasets, raw_dataset_dict = prepare_datasets()

    print("==> 🚀 Iniciando entrenamiento...")
    trainer, model, train_results = train_model(processed_datasets['train'],
                                                processed_datasets['valid'])
    print("==> 💾 Guardando modelo entrenado...")
    base_dir = os.path.expanduser("~/ViTresults_user")
    model_dir = os.path.join(base_dir, f"original{TYPE}")
    model_path = os.path.join(base_dir, "ViT_Orig.pth")
    trainer.save_model(model_dir)
    torch.save(model.state_dict(), model_path)

    print("==> 📊 Evaluando modelo en test...")
    # Obtener nombres de las carpetas base
    train_name = os.path.basename(train_dir.rstrip("/\\"))
    val_name = os.path.basename(val_dir.rstrip("/\\"))

    # Construir el nombre dinámico para la carpeta de salida
    output_dir = f"./outputs_{train_name}_{val_name}"

    evaluate_model(trainer, processed_datasets['test'], raw_dataset_dict, train_results, output_dir)
    
    print("✅ Proceso completo.")

if __name__ == "__main__":
    main()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMPipeline
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
import wandb

# Configuración global
torch.manual_seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("\n=== Verificación de GPU ===")
print(f"Dispositivo seleccionado: {device}")
if device == 'cuda':
    print(f"Nombre de la GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️ No se detectó GPU, usando CPU.")
print("=========================\n")

class LocalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                          if f.endswith(('jpg', 'jpeg', 'png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image) if self.transform else image

def train_model(config):
    """Función principal de entrenamiento"""
    wandb.init(project=config['wandb_project'], name=config['model_save_name'], config=config)
    start_time = time.time()
    
    # Cargar modelo preentrenado
    try:
        print(f"⏳ Cargando modelo {config['start_model']}...")
        pipe = DDPMPipeline.from_pretrained(config['start_model'], torch_dtype=torch.float32, use_safetensors=False).to(device)
        pipe.unet.enable_gradient_checkpointing()
        print("✅ Modelo cargado correctamente")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        raise

    for param in pipe.unet.parameters():
        param.requires_grad = True

    # Configurar scheduler y optimizador
    scheduler = DDPMScheduler.from_pretrained(config['start_model'])
    scheduler.set_timesteps(1000) 
    
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=config['lr'], weight_decay=1e-4)
    torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_norm=1.0)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.0)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # Preprocesamiento de imágenes
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Dataset y DataLoader
    dataset = LocalImageDataset(config['local_dataset_path'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Inicializar métricas
    epoch_losses  = []

    # Loop de entrenamiento
    for epoch in range(1, config['num_epochs'] + 1):
        pipe.unet.train()
        epoch_loss = 0
        
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{config['num_epochs']}")):
            clean_images = batch.to(device)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):  # <-- Activa mixed precision
                # Forward pass
                noise = torch.randn_like(clean_images)
                timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=device).long()
                noisy_images = pipe.scheduler.add_noise(clean_images, noise, timesteps)
                
                noise_pred = pipe.unet(noisy_images, timesteps)["sample"]
                loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)

            # Backward pass con GradScaler
            scaler.scale(loss).backward()
            
            if (step + 1) % config['grad_accumulation_steps'] == 0:
                scaler.step(optimizer)
                scaler.update()
            
            # Registrar métricas
            epoch_loss += loss.item()
            wandb.log({'batch_loss': loss.item()})

        # Fin de época
        epoch_loss_avg = epoch_loss / len(dataloader)
        epoch_losses.append(epoch_loss_avg)
        wandb.log({'epoch_loss': epoch_loss_avg, 'lr': lr_scheduler.get_last_lr()[0]})
        lr_scheduler.step()

    # Guardar modelo final
    os.makedirs(config['model_save_name'], exist_ok=True)
    pipe.save_pretrained(config['model_save_name'])
    wandb.finish()

    # Calcular métricas finales
    total_time = time.time() - start_time
    mean_loss = np.mean(epoch_losses)
    
    print(f"\n⏱️ Tiempo total entrenamiento '{config['model_save_name']}': {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    print(f"📉 Pérdida media: {mean_loss:.4f}")

    return epoch_losses

def generate_images(model_path, num_images, output_dir):
    """Generar imágenes usando el modelo entrenado"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"⏳ Cargando modelo desde {model_path}...")
    pipe = DDPMPipeline.from_pretrained(model_path).to(device)
    
    scheduler = pipe.scheduler
    scheduler.set_timesteps(100)
    
    print(f"🎨 Generando {num_images} imágenes...")
    batch_size = 32
    for i in tqdm(range(0, num_images, batch_size)):
        current_batch_size = min(batch_size, num_images - i)
        x = torch.randn(current_batch_size, 3, 256, 256).to(device)
        
        for t in scheduler.timesteps:
            model_input = scheduler.scale_model_input(x, t)
            with torch.no_grad():
                noise_pred = pipe.unet(model_input, t)["sample"]
            x = scheduler.step(noise_pred, t, x).prev_sample
        
        for j in range(current_batch_size):
            img = x[j].permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
            img = Image.fromarray((img.numpy() * 255).astype(np.uint8))
            img.save(os.path.join(output_dir, f"image_{i+j+1:04d}.png"))

def main():
    # Configuración para estadios iniciales
    config_iniciales = {
        'image_size': 256,
        'batch_size': 16,
        'grad_accumulation_steps': 4,
        'num_epochs': 200,
        'start_model': "google/ddpm-ema-celebahq-256",
        'local_dataset_path': "Estadios_iniciales_con_aum_3000",
        'device': device,
        'model_save_name': "modelo_iniciales",
        'wandb_project': "ddpm_estadios_lesion",
        'log_samples_every': 20,
        'lr': 1e-4,
        'calculate_fid': False,
        'scheduler_type': "DPMSolverSinglestep"
    }

    # Configuración para estadios avanzados
    config_avanzados = config_iniciales.copy()
    config_avanzados.update({
        'local_dataset_path': "Estadios_avanzados_con_aum_3000",
        'model_save_name': "modelo_avanzados"
    })

    # Entrenar ambos modelos
    print("🚀 Entrenando modelo para estadios iniciales...")
    losses_ini = train_model(config_iniciales)
    
    print("\n🚀 Entrenando modelo para estadios avanzados...")
    losses_ava = train_model(config_avanzados)

    # Generar imágenes
    print("\n🖼️ Generando imágenes para estadios iniciales...")
    generate_images("modelo_iniciales", 3000, "generadas_iniciales")
    
    print("\n🖼️ Generando imágenes para estadios avanzados...")
    generate_images("modelo_avanzados", 3000, "generadas_avanzados")

    # Guardar resumen de métricas en un archivo de texto
    print('Guardando el resumen del entrenamiento en un .txt...')
    resumen_txt = "resumen_entrenamiento.txt"
    with open(resumen_txt, "w") as f:
        f.write("Resumen del Entrenamiento\n")
        f.write("==========================\n\n")

        f.write("Estadios Iniciales\n")
        f.write("------------------\n")
        f.write(f"Tiempo total: {time.strftime('%H:%M:%S', time.gmtime(config_iniciales.get('training_time', 0)))}\n")
        f.write(f"Pérdida media: {np.mean(losses_ini):.4f}\n")
        f.write("\n")

        f.write("Estadios Avanzados\n")
        f.write("------------------\n")
        f.write(f"Tiempo total: {time.strftime('%H:%M:%S', time.gmtime(config_avanzados.get('training_time', 0)))}\n")
        f.write(f"Pérdida media: {np.mean(losses_ava):.4f}\n")

    print("\n✅ Todos los procesos completados exitosamente!")

if __name__ == "__main__":
    main()

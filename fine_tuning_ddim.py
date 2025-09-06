""" Librerías """
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Solo usar GPU 0 para optimizar el rendimiento
import math
import time
import numpy as np
import pathlib
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from PIL import Image
import concurrent.futures

from tensorflow.keras.layers import Conv2D, Dense, UpSampling2D, Add,\
    AveragePooling2D, Concatenate, Input, Activation, LayerNormalization
from tensorflow.keras import Model, Sequential

print("Dispositivos disponibles:", tf.config.list_physical_devices('GPU')) # Comprobamos si tensorflow está utilizando la GPU


# --------------------
# CONFIGURACIONES GENERALES
# --------------------
BATCH_SIZE = 16
IMG_SIZE = 224 
N_BLOCK = 2
EPOCHS = 200  # Reducir por demostración
CHECKPOINT_DIR = './results/models_weight'


class LinearAttention(tf.keras.layers.Layer):
    def __init__(self, channels, num_heads=4, head_dim=32):
        super(LinearAttention, self).__init__()
        self.inv_scale = head_dim ** -0.5
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        # proyecciones qkv
        self.qkv_proj = Conv2D(filters=self.inner_dim * 3, kernel_size=1, strides=1, use_bias=False)

        # proyección de salida
        self.output_proj = Sequential([
            Conv2D(filters=channels, kernel_size=1, strides=1),
            LayerNormalization()
        ])

    def call(self, inputs, training=True):
        skip_conn = inputs
        batch, height, width, channels = inputs.shape

        qkv_tensor = self.qkv_proj(inputs)
        q_tensor, k_tensor, v_tensor = tf.split(qkv_tensor, num_or_size_splits=3, axis=-1)

        q_tensor = tf.reshape(q_tensor, [-1, self.num_heads, self.head_dim, height * width])
        k_tensor = tf.reshape(k_tensor, [-1, self.num_heads, self.head_dim, height * width])
        v_tensor = tf.reshape(v_tensor, [-1, self.num_heads, self.head_dim, height * width])

        q_tensor = tf.nn.softmax(q_tensor, axis=-2)
        k_tensor = tf.nn.softmax(k_tensor, axis=-1)
        q_tensor = q_tensor * self.inv_scale

        context_mat = tf.einsum('b h d n, b h e n -> b h d e', k_tensor, v_tensor)

        attn_out = tf.einsum('b h d e, b h d n -> b h e n', context_mat, q_tensor)
        attn_out = tf.reshape(attn_out, [-1, height, width, self.inner_dim])
        attn_out = self.output_proj(attn_out, training=training)

        return attn_out + skip_conn

def residual_block(inputs, filters, embed):
    if inputs.shape[-1] == filters:
        shortcut = inputs
    else:
        shortcut = Conv2D(filters, 1, 1, padding='same')(inputs)

    h = Conv2D(filters, 3, 1, padding='same')(inputs)
    h = tfa.layers.GroupNormalization(groups=8, epsilon=1e-05)(h)

    embed_proj = Dense(filters * 2)(embed)
    scale, shift = tf.split(embed_proj, num_or_size_splits=2, axis=-1)
    h = h * (scale + 1) + shift
    h = Activation('swish')(h)

    h = Conv2D(filters, 3, 1, padding='same')(h)
    h = tfa.layers.GroupNormalization(groups=8, epsilon=1e-05)(h)
    h = Activation('swish')(h)

    result = Add()([h, shortcut])
    return result

def down_stage(feat, skip_conn, channels, num_layers, emb):
    for _ in range(num_layers):
        feat = residual_block(feat, channels, emb)
        skip_conn.append(feat)
    
    feat = LinearAttention(channels)(feat)
    down_out = AveragePooling2D()(feat)
    return down_out


def up_stage(feat, skip_conn, channels, num_layers, emb):
    feat = UpSampling2D()(feat)
    for _ in range(num_layers):
        feat = Concatenate()([feat, skip_conn.pop()])
        feat = residual_block(feat, channels, emb)
    
    feat = LinearAttention(channels)(feat)
    return feat

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, embed_dim=32, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.fc1 = Dense(self.embed_dim, activation='swish')
        self.fc2 = Dense(self.embed_dim, activation='swish')
        
    def call(self, inputs, training=True):
        inputs = tf.cast(inputs, tf.float32)
        half_size = self.embed_dim // 2
        scale = math.log(self.max_len) / (half_size - 1)
        freq = tf.exp(tf.range(half_size, dtype=tf.float32) * -scale)
        pos_emb = inputs * freq[None, :]

        pos_emb = tf.concat([tf.sin(pos_emb), tf.cos(pos_emb)], axis=-1)
        pos_emb = tf.reshape(pos_emb, [-1, 1, 1, self.embed_dim])
        pos_emb = self.fc1(pos_emb)
        pos_emb = self.fc2(pos_emb)
        return pos_emb

def build_unet(image_shape):
    input_img = Input(image_shape)
    time_step = Input((1,))
    
    t_emb = PositionalEncoding()(time_step)
    feat = Conv2D(32, 1, 1, padding='same')(input_img)
    skip_connections = []
    
    feat = down_stage(feat, skip_connections, 64, N_BLOCK, t_emb)
    feat = down_stage(feat, skip_connections, 128, N_BLOCK, t_emb)
    feat = down_stage(feat, skip_connections, 128, N_BLOCK, t_emb)

    for _ in range(N_BLOCK):
        feat = residual_block(feat, 256, t_emb)
        
    feat = up_stage(feat, skip_connections, 128, N_BLOCK, t_emb)
    feat = up_stage(feat, skip_connections, 128, N_BLOCK, t_emb)
    feat = up_stage(feat, skip_connections, 64, N_BLOCK, t_emb)
    feat = residual_block(feat, 32, t_emb)
    
    output_img = Conv2D(3, 1, 1, padding='same', kernel_initializer="zeros")(feat)
    return Model([input_img, time_step], output_img)


def train_and_save_loss(DATA_PATH, OUTPUT_PATH, LABEL):
    # ---------------------------------
    # CARGA Y PREPROCESAMIENTO DE DATOS
    # ---------------------------------
    inp_data_path = pathlib.Path(DATA_PATH) 
    file_list = [str(path) for path in inp_data_path.rglob('*') if path.suffix.lower() == '.jpg']

    def preprocess(file_path, img_size=IMG_SIZE):
        imgs = tf.io.read_file(file_path)
        imgs = tf.io.decode_jpeg(imgs, channels=3)
        imgs = tf.image.resize(imgs, [img_size, img_size])

        imgs = tf.image.convert_image_dtype(imgs, dtype=tf.float32)
        imgs = (imgs - 127.5) / 127.5
        return imgs


    data_path = tf.data.Dataset.from_tensor_slices(file_list)
    train_data = data_path.map(preprocess).shuffle(500).repeat(10).batch(BATCH_SIZE) 

    img = next(iter(train_data))
    img_to_show = (img[0] * 0.5) + 0.5  # Desnormalizar para mostrar

    plt.imshow(img_to_show)
    #plt.show()
    plt.close() 


    # -------------------------------
    # Mismo código que en ddim_DermaMNIST.py
    # -------------------------------
    num_timesteps = 1000
    beta_schedule = np.linspace(0.0001, 0.02, num_timesteps)
    alpha_vals = 1 - beta_schedule
    alpha_cumprod = np.cumprod(alpha_vals, axis=0)
    alpha_cumprod_shifted = np.concatenate((np.array([1.]), alpha_cumprod[:-1]), axis=0)
    sqrt_alpha_cumprod = np.sqrt(alpha_cumprod_shifted)
    sqrt_one_minus_alpha_cumprod = np.sqrt(1 - alpha_cumprod_shifted)


    def add_gaussian_noise(x_start, t_indices):
        noise_sample = tf.random.normal(shape=x_start.shape)
        
        # Obtener los coeficientes correspondientes a cada timestep
        sqrt_alpha_t = tf.reshape(tf.gather(sqrt_alpha_cumprod, t_indices), [-1, 1, 1, 1])
        sqrt_one_minus_alpha_t = tf.reshape(tf.gather(sqrt_one_minus_alpha_cumprod, t_indices), [-1, 1, 1, 1])
        
        # Asegurar tipo de dato uniforme
        sqrt_alpha_t = tf.cast(sqrt_alpha_t, tf.float32)
        sqrt_one_minus_alpha_t = tf.cast(sqrt_one_minus_alpha_t, tf.float32)
        noise_sample = tf.cast(noise_sample, tf.float32)

        noisy_output = sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise_sample
        return noisy_output, noise_sample

    model_unet = build_unet((IMG_SIZE, IMG_SIZE, 3))
    #model_unet.summary()

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    def ddim_step(x_t, noise_est, timestep, stride):
        alpha_t = np.reshape(np.take(alpha_cumprod_shifted, timestep), [-1, 1, 1, 1])
        alpha_prev = np.reshape(np.take(alpha_cumprod_shifted, timestep - stride), [-1, 1, 1, 1])
            
        x0_pred = (x_t - ((1 - alpha_t) ** 0.5) * noise_est) / (alpha_t ** 0.5)
        x0_pred = (alpha_prev ** 0.5) * x0_pred

        x0_pred = x0_pred + ((1 - alpha_prev) ** 0.5) * noise_est
        return x0_pred

    num_infer_steps = 200
    infer_schedule = range(0, num_timesteps, num_timesteps // num_infer_steps)
    step_size = num_timesteps // num_infer_steps

    def save_image_worker(img_array, path):
        img = np.clip(img_array * 0.5 + 0.5, 0, 1)
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(path)

    def generate_save_img(path=OUTPUT_PATH, save=True, num_images=16, global_image_counter=0):
        batches = num_images // BATCH_SIZE
        remainder = num_images % BATCH_SIZE

        for b in range(batches + (1 if remainder else 0)):
            current_batch_size = BATCH_SIZE if b < batches else remainder
            if current_batch_size == 0:
                continue

            x = tf.random.normal((current_batch_size, IMG_SIZE, IMG_SIZE, 3))

            for i in reversed(range(num_infer_steps)):
                t = np.repeat(infer_schedule[i], current_batch_size)
                pred_noise = model_unet([x, t])
                x = ddim_step(x, pred_noise, t, step_size)

                if any(t - step_size) == 0:
                    break

            if save:
                os.makedirs(path, exist_ok=True)
                save_jobs = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    for i in range(current_batch_size):
                        global_image_counter += 1
                        img_path = os.path.join(path, f"image_{global_image_counter:04d}.png")
                        save_jobs.append(executor.submit(save_image_worker, x[i].numpy(), img_path))
                    
                    # Espera a que todas las imágenes se hayan guardado
                    concurrent.futures.wait(save_jobs)

        print(f"Guardadas {num_images} imágenes. Total acumulado: {global_image_counter}")


    # ------------
    # Código nuevo
    # ------------
    # Cargar CHECKPOINT preentrenado y congelar capas
    ckpt = tf.train.Checkpoint(model=model_unet)

    specific_ckpt_path = os.path.join(CHECKPOINT_DIR, 'ckpt-99')  ### Coge el checkpoint que quieras
    if os.path.exists(specific_ckpt_path + ".index"):
        ckpt.restore(specific_ckpt_path)
        print(f"Modelo restaurado desde el checkpoint específico: {specific_ckpt_path}")
    else:
        print(f"Checkpoint específico no encontrado: {specific_ckpt_path}")

    # Congelar el primer tercio de las capas de la U-Net y deja entrenables las restantes para fine-tuning
    for layer in model_unet.layers[:len(model_unet.layers) // 3]:  
        layer.trainable = False
    for layer in model_unet.layers[len(model_unet.layers) // 3:]:  
        layer.trainable = True

    # Configuración del optimizador y pérdida
    loss_fn = tf.keras.losses.MeanSquaredError()
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=EPOCHS * len(train_data),
        alpha=0.01  # Reduce lentamente el learning rate
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Crear métrica para promediar la pérdida
    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")

    # Checkpoints
    finetune_ckpt_dir = os.path.join(OUTPUT_PATH, "finetune_ckpt")
    if not os.path.exists(finetune_ckpt_dir):
        os.makedirs(finetune_ckpt_dir)
    finetune_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model_unet)
    finetune_ckpt_manager = tf.train.CheckpointManager(finetune_ckpt, finetune_ckpt_dir, max_to_keep=3)

    @tf.function
    def train_step(images):
        t_size = images.shape[0]
        t = tf.random.uniform(shape=[t_size], minval=0, maxval=num_timesteps, dtype=tf.int32)
        noisy_images, noise = add_gaussian_noise(images, t)
        
        with tf.GradientTape() as tape:
            pred_noise = model_unet([noisy_images, tf.cast(t, tf.float32)])
            loss = loss_fn(noise, pred_noise)
        
        grads = tape.gradient(loss, model_unet.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_unet.trainable_variables))
        
        train_loss_metric.update_state(loss)
        return loss

    # Número total de imágenes a generar
    total_images_to_generate = 3000

    # Calcula cuántas imágenes generar por época
    images_per_epoch = total_images_to_generate // EPOCHS

    # Asegúrate de que al menos se genere una imagen por época
    images_per_epoch = max(images_per_epoch, 1)

    loss_history = []
    total_loss = 0.0

    start_time = time.time()  # Guardar el tiempo de inicio

    # Entrenamiento
    for epoch in range(1, EPOCHS + 1):
        train_loss_metric.reset_states()

        print(f"\nInicio de Epoch {epoch}/{EPOCHS}")

        for step, batch in enumerate(train_data):
            loss = train_step(batch)
            total_loss += loss  # Acumulamos la pérdida total

        epoch_loss = train_loss_metric.result().numpy()
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch} - Loss: {epoch_loss:.6f}")

        # Guardar checkpoint
        finetune_ckpt_manager.save()

    # Generar las 3000 imágenes al final del entrenamiento
    generate_save_img(
        num_images=total_images_to_generate,
        global_image_counter=0
    )

    # Calcular y mostrar la pérdida media total
    avg_loss = total_loss / (EPOCHS * len(train_data))  # Promedio de todas las pérdidas
    print(f"\navg_loss: {avg_loss:.6f}")

    end_time = time.time()
    total_training_time = end_time - start_time

    # Convertir a horas, minutos y segundos
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)

    print(f"\nTiempo total de entrenamiento: {hours}h {minutes}m {seconds}s")

    os.makedirs(OUTPUT_PATH, exist_ok=True)  # Crear la carpeta si no existe

    # Guardar resultados en un archivo de texto común
    summary_file = "resumen_entrenamiento.txt"
    with open(summary_file, 'a') as f:
        f.write(f"RESULTADOS PARA: {LABEL.upper()}\n")
        f.write(f"avg_loss: {avg_loss:.6f}\n")
        f.write(f"Tiempo total de entrenamiento: {hours}h {minutes}m {seconds}s\n")
        f.write("="*50 + "\n")

    return loss_history

# Antes de ejecutar train_and_save_loss
for path in ['Estadios_iniciales_con_aum_3000', 'Estadios_avanzados_con_aum_3000']:
    if not os.path.exists(path):
        raise FileNotFoundError(f"¡Directorio no encontrado: {path}!")

# Ejecutar para iniciales
initial_loss = train_and_save_loss(
    'Estadios_iniciales_con_aum_3000',
    './finetuning_3000_iniciales',
    'iniciales'
)

# Reiniciar modelo
model_unet = build_unet((IMG_SIZE, IMG_SIZE, 3))

# Ejecutar para avanzados
advanced_loss = train_and_save_loss(
    'Estadios_avanzados_con_aum_3000',
    './finetuning_3000_avanzados',
    'avanzados'
)

# Graficar
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(initial_loss) + 1), initial_loss, label='Estadios iniciales', color='blue')
plt.plot(range(1, len(advanced_loss) + 1), advanced_loss, label='Estadios avanzados', color='red')
plt.title('Evolución pérdidas de entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdidas')
plt.legend()
plt.savefig('comparacion_perdidas.png', dpi=300)
plt.close()

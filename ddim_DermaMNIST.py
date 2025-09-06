""" Librerías """
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Solo usar GPU 0 para optimizar el rendimiento
import math
import time
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from IPython.display import clear_output
from tensorflow.keras.layers import Conv2D, Dense, UpSampling2D, Add,\
    AveragePooling2D, Concatenate, Input, Activation, LayerNormalization
from tensorflow.keras import Model, Sequential

import medmnist
from medmnist import INFO

print("Dispositivos disponibles:", tf.config.list_physical_devices('GPU')) # Comprobamos si tensorflow está utilizando la GPU


""" Datos """
BATCH_SIZE = 16
IMG_SIZE = 224
N_BLOCK = 2
EPOCHS = 100
CURRENT_EPOCH = 1
SAVE_EVERY_N_EPOCH = 1

LOG_DIR = './results/logs/'
CKPT_DIR = './results/models_weight'
OUTPUT_PATH = r'./models'

data_flag = 'dermamnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])
data = DataClass(split="train", size=224, download=True)  # Sizes: 28, 64, 128, 224

def preprocess(img):
    img = np.array(img)  # Convierte la imagen PIL a un array de numpy
    img = tf.convert_to_tensor(img, dtype=tf.float32)  # Luego convierte el array numpy a tensor
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])  # Redimensiona la imagen
    img = (img - 127.5) / 127.5  # Normaliza la imagen a rango [-1, 1]
    return img

# Crear dataset con las imágenes del dataset cargado
images = [preprocess(img) for img, _ in data]  # Solo toma las imágenes, no las etiquetas
train_data = tf.data.Dataset.from_tensor_slices(images).shuffle(500).batch(BATCH_SIZE)

# Mostrar una imagen del batch
img = next(iter(train_data))
img_to_show = (img[0] * 0.5) + 0.5  # Desnormalizar para mostrar

plt.imshow(img_to_show)
#plt.show()
plt.close() 


num_timesteps = 1000
beta_schedule = np.linspace(0.0001, 0.02, num_timesteps)
alpha_vals = 1 - beta_schedule
alpha_cumprod = np.cumprod(alpha_vals, axis=0)
alpha_cumprod_shifted = np.concatenate((np.array([1.]), alpha_cumprod[:-1]), axis=0)
sqrt_alpha_cumprod = np.sqrt(alpha_cumprod_shifted)
sqrt_one_minus_alpha_cumprod = np.sqrt(1 - alpha_cumprod_shifted)

""" Visualización del proceso de difusión hacia adelante """
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


fig = plt.figure(figsize=(15, 30))

for index, i in enumerate([10, 100, 300, 600]):
    noisy_im, noise = add_gaussian_noise(img[0], np.array([i,]))
    noisy_im = np.squeeze(noisy_im)
    # Recortar los valores a [0, 1]
    noisy_im_clipped = np.clip((noisy_im + 1) / 2, 0, 1)
    plt.subplot(1, 4, index+1)
    plt.axis('off')
    plt.imshow(noisy_im_clipped)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
plt.savefig(os.path.join(OUTPUT_PATH, 'forward_noising_process.png'))
#plt.show()
plt.close() 


""" Proceso inverso """
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


model_unet = build_unet((IMG_SIZE, IMG_SIZE, 3))
# model_unet.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
criterion = tf.keras.losses.MeanSquaredError()

checkpoint = tf.train.Checkpoint(model=model_unet)

# Configuración de guardado
writer = tf.summary.create_file_writer(LOG_DIR)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, CKPT_DIR, max_to_keep=5)

if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    restored_epoch = int(checkpoint_manager.latest_checkpoint.split('-')[-1])
    START_EPOCH = restored_epoch * SAVE_EVERY_N_EPOCH + 1
    print("Checkpoint más reciente restaurado en la época {}!!".format(START_EPOCH))


""" DDIM Sampling """
samples_dir = os.path.join(OUTPUT_PATH, 'samples_by_epoch')
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

def ddim_step(x_t, noise_est, timestep, stride):
    alpha_t = np.reshape(np.take(alpha_cumprod_shifted, timestep), [-1, 1, 1, 1])
    alpha_prev = np.reshape(np.take(alpha_cumprod_shifted, timestep - stride), [-1, 1, 1, 1])
        
    x0_pred = (x_t - ((1 - alpha_t) ** 0.5) * noise_est) / (alpha_t ** 0.5)
    x0_pred = (alpha_prev ** 0.5) * x0_pred

    x0_pred = x0_pred + ((1 - alpha_prev) ** 0.5) * noise_est
    return x0_pred


""" Proceso inverso (inferencia) """
num_infer_steps = 200
infer_schedule = range(0, num_timesteps, num_timesteps // num_infer_steps)
step_size = num_timesteps // num_infer_steps

def sample_and_save(epoch_idx, output_dir=OUTPUT_PATH, save_fig=True):
    imgs = tf.random.normal((16, IMG_SIZE, IMG_SIZE, 3))
    for _, step in enumerate(reversed(range(num_infer_steps))):
        t_batch = np.repeat(infer_schedule[step], 16)
        
        noise_pred = model_unet([imgs, t_batch])
        imgs = ddim_step(imgs, noise_pred, t_batch, step_size)
        
        if any(t_batch - step_size) == 0:
            break
    
    for k in range(imgs.shape[0]):
        ax = plt.subplot(4, 4, k + 1)
        norm_img = np.clip(imgs[k] * 0.5 + 0.5, 0, 1)  # Normalizar y recortar
        ax.imshow(norm_img)
        plt.axis('off')

    if save_fig:
        plt.savefig(os.path.join(samples_dir, 'epoch_{:04d}.png'.format(epoch_idx)))
    plt.close()

    return imgs


"""# Entrenamiento del modelo"""
@tf.function
def train_one_step(real_imgs):
    batch_size = real_imgs.shape[0]
    t_indices = tf.random.uniform(shape=[batch_size,], minval=0, maxval=num_timesteps, dtype=tf.int32)
    noisy_batch, noise_true = add_gaussian_noise(real_imgs, t_indices)

    with tf.GradientTape() as tape:
        noise_pred = model_unet([noisy_batch, t_indices])
        loss_value = criterion(noise_pred, noise_true)

    grads = tape.gradient(loss_value, model_unet.trainable_variables)
    optimizer.apply_gradients(zip(grads, model_unet.trainable_variables))
    return loss_value


start_time = time.time()
epoch_losses = []

for epoch_id in range(START_EPOCH, EPOCHS + 1):
    epoch_start = time.time()
    print('Comienzo de la época {}'.format(epoch_id))
    
    batch_losses = []
    for step_idx, batch_data in enumerate(train_data):
        loss_val = train_one_step(batch_data)
        batch_losses.append(loss_val)
        if step_idx % 100 == 0:
            print('.', end='')
        if step_idx > 1000:
            break

    mean_loss = np.mean(batch_losses)
    epoch_losses.append(mean_loss)
    print('\n Época {} finalizada ~ ~ ~ ~ ~'.format(epoch_id))

    with writer.as_default():
        tf.summary.scalar('loss', mean_loss, step=epoch_id)

    if epoch_id % SAVE_EVERY_N_EPOCH == 0:
        clear_output(wait=True)
        # Guardar checkpoint
        checkpoint_path = checkpoint_manager.save()
        print('Checkpoint guardado en la época {} en {}'.format(epoch_id, checkpoint_path))
    
    print('Época {} - pérdida promedio: {} \n'.format(epoch_id, mean_loss)) 
    print('Duración de la época {}: {} segundos\n'.format(epoch_id, time.time() - epoch_start))                                   

# Tiempo total de ejecución del entrenamiento
elapsed_time = time.time() - start_time
hrs = int(elapsed_time // 3600)
mins = int((elapsed_time % 3600) // 60)
secs = int(elapsed_time % 60)

# Pérdida media global
global_mean_loss = np.mean(epoch_losses)
print(f'Pérdida promedio: {global_mean_loss}')
print(f'Duración total del entrenamiento: {hrs}h {mins}m {secs}s')

# Curva de pérdida de entrenamiento
plt.figure(figsize=(8, 5))
plt.plot(range(START_EPOCH, EPOCHS + 1), epoch_losses, linestyle='-', color='b')
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.title("Pérdida de entrenamiento")
plt.savefig("training_loss_ddim_derma.png", dpi=300, bbox_inches='tight')
plt.close()

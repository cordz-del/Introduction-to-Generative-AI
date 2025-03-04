import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

latent_dim = 2  # Dimension of the latent space

# Encoder Model
class Encoder(Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')
        self.conv2 = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(latent_dim + latent_dim)  # Outputs both mean and logvar

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)
        mean, logvar = tf.split(x, num_or_size_splits=2, axis=1)
        return mean, logvar

# Reparameterization trick
def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=tf.shape(mean))
    return eps * tf.exp(logvar * 0.5) + mean

# Decoder Model
class Decoder(Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense = layers.Dense(7 * 7 * 64, activation='relu')
        self.reshape_layer = layers.Reshape((7, 7, 64))
        self.deconv1 = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')
        self.deconv2 = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')
        self.deconv3 = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')

    def call(self, x):
        x = self.dense(x)
        x = self.reshape_layer(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x

# VAE combining encoder and decoder
class VAE(Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder()

    def call(self, x):
        mean, logvar = self.encoder(x)
        z = reparameterize(mean, logvar)
        x_recon = self.decoder(z)
        return x_recon, mean, logvar

# Loss function: reconstruction loss + KL divergence
def compute_loss(model, x):
    x_recon, mean, logvar = model(x)
    recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, x_recon))
    kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
    return recon_loss + kl_loss

optimizer = tf.keras.optimizers.Adam(1e-4)
vae = VAE(latent_dim)

# Training step for VAE
@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Example training loop on MNIST
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = np.expand_dims(x_train, -1)
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(64)

epochs = 5
for epoch in range(epochs):
    for train_x in train_dataset:
        loss = train_step(vae, train_x, optimizer)
    print("Epoch:", epoch, "Loss:", loss.numpy())

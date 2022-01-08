import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pathlib
import numpy as np
import json
tfds.disable_progress_bar()
AUTOTUNE = tf.data.experimental.AUTOTUNE
FOLDER = 'distributed_checkpoints_UTK_young2old_cyclegan'
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["192.168.0.108:12345", "192.168.0.101:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})
"""
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["192.168.0.108:12345", "192.168.0.101:23456"]
    },
    'task': {'type': 'worker', 'index': 1}
})
"""

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
BUFFER_SIZE = 5
BATCH_SIZE_PER_REPLICA = 1
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 10
EPOCHS = 3

def decode_img(image):
  # convert the compressed string to a 3D uint8 tensor
  image = tf.image.decode_jpeg(image, channels=OUTPUT_CHANNELS)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  #image = tf.image.convert_image_dtype(image, tf.float32)
  # resize the image to the desired size.
  #return tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return image

def process_path(file_path):
  # load the raw data from the file as a string
  image = tf.io.read_file(file_path)
  image = decode_img(image)
  return image

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
  return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)
  # random mirroring
  image = tf.image.random_flip_left_right(image)
  return image

def preprocess_image_train(image):
  #image = tf.image.decode_jpeg(image, channels=3)
  image = random_jitter(image)
  image = normalize(image)
  return image

def preprocess_image_test(image):
  image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  image = normalize(image)
  return image

def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)
  generated_loss = loss_obj(tf.zeros_like(generated), generated)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

def generate_images(model, test_input):
  prediction = model(test_input)
  plt.figure(figsize=(12, 12))
  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']
  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

@tf.function
def train_step(real_x_out, real_y_out):
  def train_step_in(real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
      # Generator G translates X -> Y
      # Generator F translates Y -> X.
      fake_y = generator_g(real_x, training=True)
      cycled_x = generator_f(fake_y, training=True)
      fake_x = generator_f(real_y, training=True)
      cycled_y = generator_g(fake_x, training=True)
      # same_x and same_y are used for identity loss.
      same_x = generator_f(real_x, training=True)
      same_y = generator_g(real_y, training=True)
      disc_real_x = discriminator_x(real_x, training=True)
      disc_real_y = discriminator_y(real_y, training=True)
      disc_fake_x = discriminator_x(fake_x, training=True)
      disc_fake_y = discriminator_y(fake_y, training=True)
      # calculate the loss
      gen_g_loss = generator_loss(disc_fake_y)
      gen_f_loss = generator_loss(disc_fake_x)
      total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
      # Total generator loss = adversarial loss + cycle loss
      total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
      total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)
      disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
      disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)
    discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)
    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))
  strategy.experimental_run_v2(train_step_in, args=(real_x_out, real_y_out))

def load_epoch():
	with open(FOLDER + "/epoch.txt", "r") as f:
		try:
			epoch = f.read()
		except:
			print("File Error !")
	if(epoch==""):
		print("File Is Empty !!!")
		return 0
	return int(epoch)

def write_epoch(epoch):
	with open(FOLDER + "/epoch.txt", "w") as f:
		f.write(str(epoch))



#data_dir = 'Y:\\Projects\\Datasets\\horse2zebra'
#train_young = tf.data.Dataset.list_files(str(data_dir+'\\trainA\\*'))
#train_old = tf.data.Dataset.list_files(str(data_dir+'\\trainB\\*'))
#test_young = tf.data.Dataset.list_files(str(data_dir+'\\testA\\*'))
#test_old = tf.data.Dataset.list_files(str(data_dir+'\\testB\\*'))
data_dir = 'Y:\\Projects\\Datasets\\young2old-dataset\\train\\train'
data_dir_utk = 'Y:\\Projects\\Datasets\\YoungOld\\'
train_young = tf.data.Dataset.list_files(str(data_dir_utk+'\\Young\\Male\\*.jpg')).take(BUFFER_SIZE)
train_old = tf.data.Dataset.list_files(str(data_dir_utk+'\\Old\\Male\\*.jpg')).take(BUFFER_SIZE)
test_young = tf.data.Dataset.list_files(str(data_dir+'\\A_test\\*.jpg'))
test_old = tf.data.Dataset.list_files(str(data_dir+'\\B_test\\*.jpg'))
train_young = train_young.map(process_path, num_parallel_calls=AUTOTUNE)
train_old = train_old.map(process_path, num_parallel_calls=AUTOTUNE)
test_young = test_young.map(process_path, num_parallel_calls=AUTOTUNE)
test_old = test_old.map(process_path, num_parallel_calls=AUTOTUNE)

train_young = train_young.map(preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_old = train_old.map(preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_zip = tf.data.Dataset.zip((train_young, train_old))
distributed_train_zip = strategy.experimental_distribute_dataset(train_zip)
#distributed_train_young = strategy.experimental_distribute_dataset(train_young)
#distributed_train_old = strategy.experimental_distribute_dataset(train_old)
test_young = test_young.map(preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_old = test_old.map(preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

with strategy.scope():
  generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
  generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
  
  discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
  discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)
  
  loss_obj = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)
  
  generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  
  discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

"""
checkpoint_path = "./" + FOLDER + "/train"
ckpt = tf.train.Checkpoint(generator_g=generator_g, generator_f=generator_f, discriminator_x=discriminator_x, discriminator_y=discriminator_y, generator_g_optimizer=generator_g_optimizer, generator_f_optimizer=generator_f_optimizer, discriminator_x_optimizer=discriminator_x_optimizer, discriminator_y_optimizer=discriminator_y_optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')
"""

#for inp in test_young.take(6):
  #generate_images(generator_g, inp)

for epoch in range(load_epoch(), EPOCHS):
  start = time.time()
  n = 0
  #for image_x, image_y in zip((distributed_train_young, distributed_train_old)):
  for image_x, image_y in distributed_train_zip:
    train_step(image_x, image_y)
    if n % 2 == 0:
      print('.', end='')
    n+=1
  clear_output(wait=True)
  # Using a consistent image (sample_horse) so that the progress of the model
  # is clearly visible.
  #generate_images(generator_g, sample_horse)
#  if (epoch + 1) % 1 == 0:
    #ckpt_save_path = ckpt_manager.save()
    #print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
  write_epoch(epoch+1)

for inp in test_young.take(1):
  generate_images(generator_g, inp)

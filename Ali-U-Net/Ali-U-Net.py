import argparse
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Set random seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Argument parser
parser = argparse.ArgumentParser(description="Train the simplified Ali-U-Net on NPZ alignments.")
parser.add_argument("rows", type=int, help="The rows in the training data.")
parser.add_argument("columns", type=int, help="The columns in the training data.")
parser.add_argument("activation", type=str, help="The activation function.")
parser.add_argument("initialization", type=str, help="The initialization function.")
parser.add_argument("data_file", type=str, help="The .npz file containing training data.")
parser.add_argument("file_path", type=str, help="The file path for the checkpoint file.")
parser.add_argument("file_name", type=str, help="The file name for the trained h5 model.")
args = parser.parse_args()

# Parse arguments
rows = args.rows
columns = args.columns
act_fun = args.activation
act_init = args.initialization

# Load data from NPZ
data = np.load(args.data_file)
x_data = data['x']
y_data = data['y']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=seed
)

# Define the simplified U-Net model
inputs = tf.keras.layers.Input(shape=(rows, columns, 5))

# Downsampling
c1 = tf.keras.layers.Conv2D(32, (11, 11), activation=act_fun, kernel_initializer=act_init, padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(32, (11, 11), activation=act_fun, kernel_initializer=act_init, padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(64, (7, 7), activation=act_fun, kernel_initializer=act_init, padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(64, (7, 7), activation=act_fun, kernel_initializer=act_init, padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(128, (5, 5), activation=act_fun, kernel_initializer=act_init, padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(128, (5, 5), activation=act_fun, kernel_initializer=act_init, padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

# Bottleneck
c4 = tf.keras.layers.Conv2D(256, (3, 3), activation=act_fun, kernel_initializer=act_init, padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.3)(c4)
c4 = tf.keras.layers.Conv2D(256, (3, 3), activation=act_fun, kernel_initializer=act_init, padding='same')(c4)

# Upsampling
u5 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
u5 = tf.keras.layers.concatenate([u5, c3])
c5 = tf.keras.layers.Conv2D(128, (5, 5), activation=act_fun, kernel_initializer=act_init, padding='same')(u5)
c5 = tf.keras.layers.Dropout(0.2)(c5)
c5 = tf.keras.layers.Conv2D(128, (5, 5), activation=act_fun, kernel_initializer=act_init, padding='same')(c5)

u6 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c2])
c6 = tf.keras.layers.Conv2D(64, (7, 7), activation=act_fun, kernel_initializer=act_init, padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.1)(c6)
c6 = tf.keras.layers.Conv2D(64, (7, 7), activation=act_fun, kernel_initializer=act_init, padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c1])
c7 = tf.keras.layers.Conv2D(32, (11, 11), activation=act_fun, kernel_initializer=act_init, padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.1)(c7)
c7 = tf.keras.layers.Conv2D(32, (11, 11), activation=act_fun, kernel_initializer=act_init, padding='same')(c7)

# Output layers
outputs = tf.keras.layers.Conv2D(5, (1, 1), activation='sigmoid')(c7)
final_outputs = tf.keras.layers.Conv2D(5, (1, 1), activation='softmax')(outputs)

# Compile the model
model = tf.keras.Model(inputs=inputs, outputs=final_outputs)
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Prepare datasets
batch_size = 64
epochs = 5
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Checkpoint callback
filepath = args.file_path + "/checkpoint-epoch-{epoch:02d}-{val_accuracy:.4f}.keras"
my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=filepath, save_weights_only=False,
                                       monitor='val_accuracy', save_best_only=True)
]

# Train the model
history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, verbose=1, callbacks=my_callbacks)

# Save metrics
np.save('acc.npy', history.history['accuracy'])
np.save('val_acc.npy', history.history['val_accuracy'])
np.save('loss.npy', history.history['loss'])
np.save('val_loss.npy', history.history['val_loss'])

# Save model
model.save(args.file_name, save_format='h5')

import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelBinarizer
import matplotlib.pyplot as plt

def create_model(loss):
    model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(4,), name='Input_Layer'),
    tf.keras.layers.Dense(12, activation='relu', input_dim=(4,)),
    
    # You can test more hiddenlayers, but with 10 neurones we already achieve 100% or almost 100%.
    #tf.keras.layers.Dense(8, activation='relu'),
    #tf.keras.layers.Dense(400, activation='relu'),
    #tf.keras.layers.Dense(400, activation='relu'),

    tf.keras.layers.Dense(3, activation='softmax') # 3 classes
    ])
    
    # Compile the model
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-7)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    return model


def plot_training_history(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    ax1.plot(epochs, loss, label='Training Loss')
    if val_loss:
        ax1.plot(epochs, val_loss, label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot Accuracy
    if acc:
        ax2.plot(epochs, acc, label='Training Accuracy')
    if val_acc:
        ax2.plot(epochs, val_acc, label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(0, 1.001)

    # Display the plots
    plt.tight_layout()
    plt.show()

    return

if __name__ == '__main__':

    # Load the dataset:
    iris = load_iris()
    X = iris.data       # Features
    y = iris.target     # Labels
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(X.min())
    print(X.max())
    
    # If we use an integer encoding for the labels, we specify:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    loss='sparse_categorical_crossentropy'
    
    # specify randon seed for tensorflow in order to get more reproducable traiing results:
    tf.random.set_seed(42)
    # This does not make the training entirely reproducable!
    
    model = create_model(loss)
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=150, batch_size=16, validation_data=(X_test, y_test))
    
    plot_training_history(history)

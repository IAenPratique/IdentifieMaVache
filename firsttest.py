from tensorflow import keras
import numpy as np

from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

from tkinter import *
from PIL import Image, ImageTk
import random

# Load CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Print data shapes
print(x_train.shape) # (50000, 32, 32, 3)
print(x_test.shape) # (10000, 32, 32, 3)
print(y_train.shape) # (50000, 10)
print(y_test.shape) # (10000, 10)

# Define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Load the saved model
model = load_model('my_model.h5')

# Choose 100 random images from test set
random_indices = random.sample(range(len(x_test)), 100)
random_images = np.array([x_test[i] for i in random_indices]).reshape(100, 32, 32, 3)
random_labels = [y_test[i] for i in random_indices]

predictions = model.predict(random_images)

# Convert test images from numpy array to a list of file paths
test_images = [f'test_image_{i}.png' for i in range(len(random_images))]
for i, img in enumerate(random_images):
    img = img * 255
    img = img.astype(np.uint8)
    im = Image.fromarray(img)
    im.save(test_images[i], format='PNG')

# Initialize Tkinter window
root = Tk()
root.title("BasicAI001 Test Results")

def update_window(test_images, test_labels, predictions, num_columns):
    num_images = len(test_images)
    image_columns = []
    label_columns = []
    for i in range(num_columns):
        image_columns.append([])
        label_columns.append([])

    for i in range(num_images):
        # Open image and convert to PhotoImage
        img = Image.open(test_images[i])
        img = ImageTk.PhotoImage(img)

        # Create a label for the image
        img_label = Label(root, image=img)
        img_label.image = img
        column_index = i % num_columns
        image_columns[column_index].append(img_label)

        # Create a label for the prediction
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        pred_label = Label(root, text=class_names[np.argmax(predictions[i])])
        label_columns[column_index].append(pred_label)

        # Check if the prediction is correct
        if np.argmax(predictions[i]) == np.argmax(test_labels[i]):
            pred_label.config(fg='green')
        else:
            pred_label.config(fg='red')

    for i in range(num_columns):
        for j in range(len(image_columns[i])):
            image_columns[i][j].grid(row=j, column=i*2)
            label_columns[i][j].grid(row=j, column=i*2+1)


def create_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)

    predictions = model.predict(x_test)
    model.save('my_model.h5')


# Call the function and pass in the test images, labels, and predictions
update_window(test_images, random_labels, predictions, 7)
# Start the Tkinter event loop
root.mainloop()




from tkinter import *
from PIL import Image, ImageTk
import random
import numpy as np
from keras.datasets import cifar10


class ShowImagesTab(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.controller = controller
        self.num_columns = 10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        model = self.controller.model

        if model is not None:
            self.create_random_dataset(model, x_test, y_test)

    def create_random_dataset(self, model, x_test, y_test):
        random_indices = random.sample(range(len(x_test)), 100)
        random_images = np.array([x_test[i] for i in random_indices]).reshape(100, 32, 32, 3)
        random_labels = [y_test[i] for i in random_indices]
        predictions = model.predict(random_images)

        test_images = [f'tmp/test_image_{i}.png' for i in range(len(random_images))]
        for i, img in enumerate(random_images):
            img = img * 255
            img = img.astype(np.uint8)
            im = Image.fromarray(img)
            im.save(test_images[i], format='PNG')

        self.update_window(test_images, random_labels, predictions, self.num_columns)

    def update_window(self, test_images, test_labels, predictions, num_columns):
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
            img_label = Label(self, image=img)
            img_label.image = img
            column_index = i % num_columns
            image_columns[column_index].append(img_label)

            # Create a label for the prediction
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            pred_label = Label(self, text=class_names[np.argmax(predictions[i])])
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

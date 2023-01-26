from tkinter import *
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from show_images_tab import ShowImagesTab

class CreateModelTab(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        self.controller = controller

        self.conv_layers_label = Label(self, text="Number of Convolutional Layers:")
        self.conv_layers_label.pack()
        self.conv_layers_cursor = Entry(self)
        self.conv_layers_cursor.pack()

        self.filters_label = Label(self, text="Number of Filters:")
        self.filters_label.pack()
        self.filters_cursor = Entry(self)
        self.filters_cursor.pack()

        self.kernel_size_label = Label(self, text="Kernel Size:")
        self.kernel_size_label.pack()
        self.kernel_size_cursor = Entry(self)
        self.kernel_size_cursor.pack()

        self.pooling_label = Label(self, text="Pooling Size:")
        self.pooling_label.pack()
        self.pooling_cursor = Entry(self)
        self.pooling_cursor.pack()

        self.create_model_button = Button(self, text="Create Model", command=self.create_model)
        self.create_model_button.pack()
        self.reset_button = Button(self, text="Reset", command=self.reset_values)
        self.reset_button.pack()
        self.reset_values()

    def reset_values(self):
        self.conv_layers_cursor.delete(0, END)
        self.conv_layers_cursor.insert(0, "2")
        self.filters_cursor.delete(0, END)
        self.filters_cursor.insert(0, "32")
        self.kernel_size_cursor.delete(0, END)
        self.kernel_size_cursor.insert(0, "3")
        self.pooling_cursor.delete(0, END)
        self.pooling_cursor.insert(0, "2")

    def save_values(self):
        self.nb_conv_layers = int(self.conv_layers_cursor.get())
        self.nb_filters = int(self.filters_cursor.get())
        self.kernel_size = int(self.kernel_size_cursor.get())
        self.pooling_size = int(self.pooling_cursor.get())

    def create_model(self):
        nb_conv_layers = int(self.conv_layers_cursor.get())
        nb_filters = int(self.filters_cursor.get())
        kernel_size = int(self.kernel_size_cursor.get())
        pooling_size = int(self.pooling_cursor.get())
        self.save_values()

        model = Sequential()

        for i in range(nb_conv_layers):
            model.add(Conv2D(nb_filters, kernel_size=(kernel_size, kernel_size), activation='relu', input_shape=(32, 32, 3)))
            model.add(MaxPooling2D(pool_size=(pooling_size, pooling_size)))

        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        self.controller.set_model(model)
        self.controller.add_show_images_tab()
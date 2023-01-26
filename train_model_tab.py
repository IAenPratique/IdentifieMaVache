

from tkinter import *
from tkinter import messagebox

from keras.datasets import cifar10
from keras.utils import np_utils
import keras.callbacks
from keras.callbacks import ModelCheckpoint



class TrainModelTab(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.controller = controller
        self.model_name = StringVar(value='my_model')

        self.epochs_label = Label(self, text="Number of Epochs:")
        self.epochs_label.pack()
        self.epochs_cursor = Entry(self)
        self.epochs_cursor.insert(0, '10')
        self.epochs_cursor.pack()

        self.save_interval_label = Label(self, text="Save Interval:")
        self.save_interval_label.pack()
        self.save_interval_cursor = Entry(self)
        self.save_interval_cursor.insert(0, '1000')
        self.save_interval_cursor.pack()

        self.batch_size_label = Label(self, text="Batch Size:")
        self.batch_size_label.pack()
        self.batch_size_cursor = Entry(self)
        self.batch_size_cursor.insert(0, '32')
        self.batch_size_cursor.pack()

        self.save_model_label = Label(self, text="Save Model As:")
        self.save_model_label.pack()
        self.save_model_cursor = Entry(self, textvariable=self.model_name)
        self.save_model_cursor.pack()

        self.train_model_button = Button(self, text="Train Model", command=self.train_model)
        self.train_model_button.pack()

        self.reset_button = Button(self, text="Reset", command=self.reset)
        self.reset_button.pack()

    def reset(self):
        self.epochs_cursor.delete(0, END)
        self.epochs_cursor.insert(0, '10')
        self.batch_size_cursor.delete(0, END)
        self.batch_size_cursor.insert(0, '32')
        self.save_model_cursor.delete(0, END)
        self.save_model_cursor.insert(0, 'my_model')
        self.save_interval_cursor.delete(0, END)
        self.save_interval_cursor.insert(0, '1000')
        self.model_name.set('my_model')

    def train_model(self):
        if self.controller.model is None:
            print("Error: No model has been created.")
            return

        nb_epochs = int(self.epochs_cursor.get())
        batch_size = int(self.batch_size_cursor.get())
        save_name = self.model_name.get()
        save_interval = int(self.save_interval_cursor.get())  # new line

        if save_name == "":
            messagebox.showerror("Error", "Please enter a name for the model")
            return

        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

        # preprocess data
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        Y_train = np_utils.to_categorical(Y_train, 10)
        Y_test = np_utils.to_categorical(Y_test, 10)

        save_callback = ModelCheckpoint(save_name + ".h5", save_weights_only=False,
                                                        save_freq=save_interval)  # new line
        self.controller.model.fit(X_train, Y_train,
                                  batch_size=batch_size,
                                  epochs=nb_epochs,
                                  validation_data=(X_test, Y_test),
                                  callbacks=[save_callback])

        messagebox.showinfo("Success", "Model has been trained and saved")

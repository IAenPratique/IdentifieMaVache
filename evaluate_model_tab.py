
from tkinter import *
from keras.datasets import cifar10
from keras.utils import to_categorical

class EvaluateModelTab(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.controller = controller

        self.evaluate_button = Button(self, text="Evaluate Model", command=self.evaluate_model)
        self.evaluate_button.pack()

    def evaluate_model(self):
        model = self.controller.model
        if model is None:
            print("Error: No model has been created.")
            return

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_test = x_test.astype('float32')
        x_test /= 255
        y_test = to_categorical(y_test, 10)

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

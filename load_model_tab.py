
from tkinter import *
from keras.models import load_model

class LoadModelTab(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.controller = controller

        self.model_path_label = Label(self, text="Model Path:")
        self.model_path_label.pack()
        self.model_path_cursor = Entry(self)
        self.model_path_cursor.pack()

        self.load_model_button = Button(self, text="Load Model", command=self.load_model)
        self.load_model_button.pack()

    def load_model(self):
        model_path = self.model_path_cursor.get()
        if not model_path:
            print("Error: No model path specified.")
            return
        try:
            model = load_model(model_path)
            self.controller.set_model(model)
            model.summary()
        except:
            print("Error: Could not load model.")

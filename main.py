
import os
from tkinter import *
from tkinter import ttk
from create_model_tab import CreateModelTab
from train_model_tab import TrainModelTab
from load_model_tab import LoadModelTab
from evaluate_model_tab import EvaluateModelTab
from show_images_tab import ShowImagesTab

class MainApp(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title("Main App")


        if not os.path.exists('tmp'):
            os.makedirs('tmp')

        self.model = None

        self.tab_control = ttk.Notebook(self, padding=20)

        self.create_model_tab = CreateModelTab(self.tab_control, self)
        self.tab_control.add(self.create_model_tab, text="Create Model")

        self.train_model_tab = TrainModelTab(self.tab_control, self)
        self.tab_control.add(self.train_model_tab, text="Train Model")

        self.load_model_tab = LoadModelTab(self.tab_control, self)
        self.tab_control.add(self.load_model_tab, text="Load Model")

        self.evaluate_model_tab = EvaluateModelTab(self.tab_control, self)
        self.tab_control.add(self.evaluate_model_tab, text="Evaluate Model")

        self.tab_control.pack(expand=1, fill='both')
        self.is_show_images_tab_loaded = False

    def set_model(self, model):
        self.model = model

    def add_show_images_tab(self):
        if not self.is_show_images_tab_loaded :
            self.show_images_tab = ShowImagesTab(self.tab_control, self)
            self.tab_control.add(self.show_images_tab, text="Test the model on random images")
            self.is_show_images_tab_loaded = True

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()




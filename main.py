from tkinter import *
from tkinter import ttk
from create_model_tab import CreateModelTab
from train_model_tab import TrainModelTab
from load_model_tab import LoadModelTab

class MainApp(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title("Main App")

        self.model = None

        self.tab_control = ttk.Notebook(self, padding=20)

        self.create_model_tab = CreateModelTab(self.tab_control, self)
        self.tab_control.add(self.create_model_tab, text="Create Model")

        self.train_model_tab = TrainModelTab(self.tab_control, self)
        self.tab_control.add(self.train_model_tab, text="Train Model")

        self.load_model_tab = LoadModelTab(self.tab_control, self)
        self.tab_control.add(self.load_model_tab, text="Load Model")

        self.tab_control.pack(expand=1, fill='both')

    def set_model(self, model):
        self.model = model




if __name__ == "__main__":
    app = MainApp()
    app.mainloop()




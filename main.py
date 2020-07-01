import pandas as pd
import numpy as np
import tkinter
from tkinter import ttk
import sklearn
from sklearn.model_selection import *
from sklearn.linear_model import LinearRegression
from tkinter import messagebox
from tkinter.filedialog import *


class learning():
    def __init__(self,train,data,data_x,data_y):
        self.train = train
        self.data  = data
        self.model = None
        self.result = None
        self.x_data = self.data.drop([data_y], axis=1)
        self.y_data = self.data[[data_y]]
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x_data,self.y_data,test_size=0.2)
        if(self.train=="linear_reg"):
            self.linear_reg()
    # TODO
    def data_checker(self):
        pass

    # TODO
    def linear_reg(self):
        self.model = LinearRegression()
        self.model.fit(self.x_train,self.y_train)
        self.result = self.model.score(self.x_test,self.y_test)

    # TODO
    def svm_cls(self):
        self.model = sklearn.svm.SVM()
        self.model.fit(self.x_train, self.y_train)
        # self.result=

    # TODO
    def k_means_ctn(self):
        self.model = sklearn.cluster.KMeans()
        self.model.fit(self.x_train, self.y_train)
        # self.result =

    # TODO
    def k_nearest_nbr(self):
        self.model = sklearn.neighbors.KNeighborsClassifier()
        self.model.fit(self.x_train, self.y_train)
        # self.result =

    """
    further dev
    def logistic_reg(self):
        pass
    def naive_bayes(self):
        pass
    def Apriori(self):
        pass
    def Prin_comp_anlys(self):
        pass
    def random_forests(self):
        pass
    """



class gui():
    def __init__(self, win):
        self.win = win
        self.s = ttk.Style(self.win)
        self.data = None
        self.data_x = None

    # TODO
    def draw_gui(self):
        # defining top,mid and end frames for loading data, Applying ML Algos and showing results
        self.top_frame = ttk.Frame(win)
        self.top_frame.grid(row=0, sticky="ew")
        self.mid_frame = ttk.Frame(win)
        self.mid_frame.grid(row=1, sticky="ew")
        self.end_frame = ttk.Frame(win)
        self.end_frame.grid(row=2, sticky="ew")

        self.load = ttk.Label(self.top_frame, text="Load Your Clean Dataset")
        self.load.pack(side="left", padx=5, pady=5)
        self.loadbtn = ttk.Button(self.top_frame, text="Load", command=self.load_data)
        self.loadbtn.pack(side="left", padx=5, pady=5)

        self.algolabel = ttk.Label(self.mid_frame, text="Select Model You want to train")
        self.algolabel.pack(side="left", padx=5, pady=5)
        self.algochooser = ttk.Combobox(self.mid_frame)
        self.algochooser['values'] = ("linear_reg", "svm_cls", "k_means_ctn", "k_nearest_nbr")
        self.algochooser.current(0)
        self.algochooser.pack(side="left", padx=5, pady=5)
        # self.features =
        # rad1 = ttk.Radiobutton(win, text="pandas", value=2)
        # rad2 = ttk.Radiobutton(win, text="numpy", value=2)
        # rad3 = ttk.Radiobutton(win, text="matplotlib", value=3)
        # rad1.grid(column=0, row=0)
        # rad2.grid(column=1, row=0)
        # rad3.grid(column=2, row=0)
        self.target = ttk.Combobox(self.mid_frame)
        self.target['values'] = [i for i in self.data.columns]
        self.target.current(0)
        self.target.pack(side="right", padx=5, pady=5)

        self.trainbtn = ttk.Button(self.mid_frame, text="Train", command=self.train_model)
        self.trainbtn.pack(side="right", padx=5, pady=5)

        self.resultlabel = ttk.Label(self.end_frame, text="The accuracy of the model is:")
        self.resultlabel.pack(side="left", padx=5, pady=5)
        self.result = ttk.Label(self.end_frame, text="test.result")
        self.result.pack(side="left", padx=5, pady=5)



    def styling(self):
        self.s.theme_use("vista")
        self.s.configure("TButton",font =('Times',10,'bold'), foreground="black", background='black')
        self.s.configure("TLabel",font =('Times',10,'bold'), foreground="white", background="black")
        self.s.configure("TFrame", foreground="white", background="black")
        self.s.configure("TCombobox", font=('Times', 10, 'bold'), foreground="black", background="black")
        # print(s.theme_names())

    # TODO
    def train_model(self):
        # print(self.algochooser.get())
        test = learning(self.algochooser.get(),self.data,"all",self.target.get())
        print(test.result)

    def load_data(self):
        file = askopenfile(mode='r', filetypes=[('csv files', '*.csv',)])
        if file is not None:
            self.data = pd.read_csv(file)
        else:
            messagebox.showinfo("Note", "You haven't loaded clean data")


if __name__ == "__main__" :
    # defining tkinter window
    win = tkinter.Tk()
    win.title("Machine Learning Visuals")
    # win.geometry("720x720")
    # win.attributes('-fullscreen', True)

    # TODO
    GUI = gui(win)
    GUI.styling()
    GUI.load_data()
    GUI.draw_gui()


    win.mainloop()
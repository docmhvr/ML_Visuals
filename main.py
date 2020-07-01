import pandas as pd
import numpy as np
import tkinter
from tkinter import ttk
import sklearn
from tkinter import messagebox
from tkinter.filedialog import *


class learning():
    def __init__(self,train):
        self.train = train
        self.model = None
        self.result = None
        self.x_train,self.x_test,self.y_train,self.y_test = sklearn.model_selection.train_test_split(x_data,y_data,test_size=0.2)

    # TODO
    def data_checker(self):
        pass

    # TODO
    def linear_reg(self):
        self.model = sklearn.linear_model.LinearRegression()
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
def train_model():
    print(algochooser.get())

def load_data():
    file = askopenfile(mode='r', filetypes=[('csv files', '*.csv',)])
    if file is not None:
        data = pd.read_csv(file)
    else:
        messagebox.showinfo("Note", "You haven't loaded clean data")

# TODO
def main():
    #defining tkinter window
    win = tkinter.Tk()
    win.title("Machine Learning Visuals")
    # win.geometry("720x720")
    # win.attributes('-fullscreen', True)

    # Styling
    s = ttk.Style(win)
    s.theme_use("vista")
    s.configure("TButton",font =('Times',10,'bold'), foreground="black", background='black')
    s.configure("TLabel",font =('Times',10,'bold'), foreground="white", background="black")
    s.configure("TFrame", foreground="white", background="black")
    s.configure("TCombobox", font=('Times', 10, 'bold'), foreground="black", background="black")
    # print(s.theme_names())

    # defining top,mid and end frames for loading data, Applying ML Algos and showing results
    top_frame = ttk.Frame(win)
    top_frame.grid(row=0, sticky="ew")
    mid_frame = ttk.Frame(win)
    mid_frame.grid(row=1, sticky="ew")
    end_frame = ttk.Frame(win)
    end_frame.grid(row=2, sticky="ew")

    load = ttk.Label(top_frame, text="Load Your Clean Dataset")
    load.pack(side="left", padx=5, pady=5)
    loadbtn = ttk.Button(top_frame, text="Load", command=load_data)
    loadbtn.pack(side="left", padx=5, pady=5)

    algolabel = ttk.Label(mid_frame, text="Select Model You want to train")
    algolabel.pack(side="left", padx=5, pady=5)
    algochooser = ttk.Combobox(mid_frame)
    algochooser['values'] = ("linear_reg", "svm_cls", "k_means_ctn", "k_nearest_nbr")
    algochooser.current(0)
    algochooser.pack(side="left", padx=5, pady=5)
    trainbtn = ttk.Button(mid_frame, text="Train", command=train_model)
    trainbtn.pack(side="right", padx=5, pady=5)

    resultlabel = ttk.Label(end_frame, text="The accuracy of the model is:")
    resultlabel.pack(side="left", padx=5, pady=5)
    result = ttk.Label(end_frame, text="test.result")
    result.pack(side="left", padx=5, pady=5)

    win.mainloop()

if __name__ == "__main__" : main()
import pandas as pd
import numpy as np
import tkinter
from tkinter import ttk
import sklearn
from sklearn.model_selection import *
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from tkinter import messagebox
from tkinter.filedialog import *
import re
import time


class learning():
    def __init__(self,train,data,data_x,data_y):
        self.train = train
        self.data  = data
        self.model = None
        self.result = "0"
        self.x_data = self.data.drop([data_y], axis=1)
        self.y_data = self.data[[data_y]]
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x_data,self.y_data,test_size=0.2)
        self.run_model()
    # TODO
    def data_checker(self):
        pass

    def run_model(self):
        runalgodict = {"Linear regression": "linear_reg",
                    "Suppport vector machine": "svm_cls",
                    "K Means Clustering": "k_means_ctn",
                    "K Nearest Neighbours": "k_nearest_nbr"}
        if (runalgodict[self.train]=="linear_reg"):
            self.linear_reg()
        elif (runalgodict[self.train]=="svm_cls"):
            self.svm_cls()
        elif (runalgodict[self.train]=="k_means_ctn"):
            self.k_means_ctn()
        elif (runalgodict[self.train]=="k_nearest_nbr"):
            self.k_nearest_nbr()

    def linear_reg(self):
        self.model = LinearRegression()
        self.model.fit(self.x_train,self.y_train)
        self.result = self.model.score(self.x_test,self.y_test)

    # TODO
    def svm_cls(self):
        self.model = SVC()
        self.model.fit(self.x_train, self.y_train)
        print("Success")
        # self.result=

    # TODO
    def k_means_ctn(self):
        self.model = KMeans()
        self.model.fit(self.x_train, self.y_train)
        print("Success")
        # self.result =

    # TODO
    def k_nearest_nbr(self):
        self.model = KNeighborsClassifier()
        self.model.fit(self.x_train, self.y_train)
        print("Success")
        # self.result =

    """
    further dev
    def logistic_reg(self):
        pass
    def naive_bayes(self):
        pass
    def apriori(self):
        pass
    def pca(self):
        pass
    def random_forests(self):
        pass
    """



class gui():
    def __init__(self, win):
        self.win = win
        self.s = ttk.Style(self.win)
        self.data = pd.DataFrame()
        self.data_x = None
        self.result = "0"

    # TODO
    def draw_gui(self):
        # defining top,mid and end frames for loading data, Applying ML Algos and showing results
        self.full_frame = ttk.Frame(win)
        self.full_frame.pack(expand="true", fill="both")
        self.top_frame = ttk.Frame(self.full_frame)
        self.top_frame.pack(side="top", expand="true", fill="x")
        self.mid_frame_1 = ttk.Frame(self.full_frame)
        self.mid_frame_1.pack(expand="true", fill="x")
        self.mid_frame_2 = ttk.Frame(self.full_frame)
        self.mid_frame_2.pack(expand="true", fill="x")
        self.end_frame = ttk.Frame(self.full_frame)
        self.end_frame.pack(side="bottom", expand="true", fill="x")

        self.load = ttk.Label(self.top_frame, text="Load Your Clean Dataset")
        self.loadbtn = ttk.Button(self.top_frame, text="Load", command=self.load_data)

        self.algolabel = ttk.Label(self.mid_frame_1, text="Select Model You want to train:")
        self.algochooser = ttk.Combobox(self.mid_frame_1)
        self.algochooser['values'] = ["Linear regression", "Suppport vector machine", "K Means Clustering", "K Nearest Neighbours"]
        self.algochooser.current(0)

        self.selectlbl = ttk.Label(self.mid_frame_2, text="Select Features and target:")
        self.features = ttk.Menubutton(self.mid_frame_2, text="Select Features")
        self.features.menu = Menu(self.features, tearoff=1)
        self.features["menu"] = self.features.menu
        for item in self.data.columns:
            self.features.menu.add_checkbutton(label=item, variable=item+"name")
        self.target = ttk.Combobox(self.mid_frame_2)
        self.target['values'] = [i for i in self.data.columns]
        self.target.set("Select Target")

        self.trainbtn = ttk.Button(self.end_frame, text="Train", command=self.train_model)
        self.training = ttk.Progressbar(self.end_frame, orient=HORIZONTAL, length=100, mode="determinate")
        self.resultlabel = ttk.Label(self.end_frame, text="The accuracy of the model is:")
        self.resultdisp = ttk.Label(self.end_frame, text=f"{float(self.result):.3f}")

        self.load.pack(side="left", padx=5, pady=5)
        self.loadbtn.pack(side="left", padx=5, pady=5)

        self.algolabel.pack(side="left", padx=5, pady=5)
        self.selectlbl.pack(side="left", padx=5, pady=5)
        self.algochooser.pack(side="left", padx=5, pady=5)
        self.features.pack(side="left", padx=5, pady=5)
        self.target.pack(side="left", padx=5, pady=5)

        self.trainbtn.pack(side="left", padx=5, pady=5)
        self.training.pack(side="left", padx=5, pady=5)
        self.resultlabel.pack(side="left", padx=5, pady=5)
        self.resultdisp.pack(side="left", padx=5, pady=5)

    def styling(self):
        self.s.theme_use("clam")
        self.s.configure("TButton",font =('Times',10,'bold'), foreground="red", background='black')
        self.s.configure("TLabel",font =('Times',10,'bold'), foreground="white", background="black")
        self.s.configure("TFrame", foreground="white", background="black")
        self.s.configure("TCombobox", font=('Times', 10, 'bold'), foreground="black", background="white")
        self.s.configure("TMenubutton", foreground="black", background="white")
        self.s.configure("TProgressbar",foreground="black", background="green")
        # print(self.s.theme_names())

    # TODO
    def train_model(self):
        self.training.start(30)
        try:
            test = learning(self.algochooser.get(), self.data, "all", self.target.get())
            self.result = test.result
            self.resultdisp['text'] = (f"{float(self.result):.3f}")
        except Exception as e:
            print("You haven't selected proper settings")
        self.training.stop()
    # TODO
    def load_hyperparams(self):
        pass

    def load_data(self):
        file = askopenfile(mode='r', filetypes=[('csv files', '*.csv',)])
        if file is not None:
            self.data = pd.read_csv(file)
        else:
            messagebox.showinfo("Note", "You haven't loaded clean data")
        self.full_frame.destroy()
        self.draw_gui()
        r = re.search("[a-zA-Z0-9]+.[a-z]+$", file.name)
        self.load['text'] = f"{r.group(0)} has loaded!"


if __name__ == "__main__" :
    # defining tkinter window
    win = tkinter.Tk()
    win.title("Machine Learning Visuals")
    # win.geometry("720x720")
    # win.attributes('-fullscreen', True)

    # TODO
    GUI = gui(win)
    GUI.styling()
    GUI.draw_gui()


    win.mainloop()
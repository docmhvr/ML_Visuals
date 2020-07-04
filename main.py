import pandas as pd
import numpy as np
import tkinter
from tkinter import ttk
from sklearn.model_selection import *
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
from tkinter import messagebox
from tkinter import simpledialog
from tkinter.filedialog import *
import re
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv

global run_algo_dict
run_algo_dict= {"Linear regression": "linear_reg",
               "Support vector machine": "svm_cls",
               "K Means Clustering": "k_means_ctn",
               "K Nearest Neighbours": "k_nearest_nbr"}

class Learning:
    def __init__(self, train, data, data_x, data_y, loadhp=None):
        self.train = train
        self.data = data
        self.model = None
        self.result = "0"
        self.x_data = self.data[data_x]
        # self.x_data = self.data.drop(["sale_price"],axis=1)
        self.target = data_y
        self.y_data = self.data[[data_y]]
        self.hyperparam = loadhp
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_data, self.y_data, test_size=0.1)
        self.plotfig = None
        self.run_model()


    # TODO
    def data_checker(self):
        pass
        # Preprocessing the data
        # LE = preprocessing.LabelEncoder()
        # self.data["buying"] = LE.fit_transform(list(self.data["buying"]))
        # print(self.data["buying"])

    def run_model(self):
        if run_algo_dict[self.train] == "linear_reg":
            self.linear_reg()
        elif run_algo_dict[self.train] == "svm_cls":
            self.svm_cls()
        elif run_algo_dict[self.train] == "k_means_ctn":
            self.k_means_ctn()
        elif run_algo_dict[self.train] == "k_nearest_nbr":
            self.k_nearest_nbr()

    def linear_reg(self):
        self.model = LinearRegression()
        self.model.fit(self.x_train, self.y_train)
        self.result = self.model.score(self.x_test, self.y_test)

    # TODO
    def linear_reg_plot(self, Plot_var):
        f = Figure(figsize=(3, 3), dpi=50)
        a = f.add_subplot(111)
        plt.style.use("ggplot")
        a.scatter(self.x_data[Plot_var], self.y_data)
        a.set_xlabel(Plot_var)
        a.set_ylabel(self.target)
        a.set_title(self.target+" against "+Plot_var)
        self.plotfig = f

    def svm_cls(self):
        self.model = SVC(kernel=self.hyperparam[0],C=int(self.hyperparam[1]))
        self.model.fit(self.x_train, self.y_train)
        self.result = metrics.accuracy_score(self.y_test, y_pred=self.model.predict(self.x_test))

    def svm_cls_plot(self, Plot_var):
        pass

    def k_means_ctn(self):
        self.model = KMeans(n_clusters=self.hyperparam)
        self.model.fit(self.x_train)
        self.result = self.model.score(self.x_test)

    def k_means_ctn_plot(self, Plot_var):
        pass

    def k_nearest_nbr(self):
        self.model = KNeighborsClassifier(self.hyperparam)
        self.model.fit(self.x_train, self.y_train)
        self.result = self.model.score(self.x_test, self.y_test)

    def k_nearest_nbr_plot(self, Plot_var):
        f = Figure(figsize=(3, 3), dpi=50)
        a = f.add_subplot(111)
        plt.style.use("ggplot")
        a.scatter(self.x_data[Plot_var], self.y_data)
        a.set_xlabel(Plot_var)
        a.set_ylabel(self.target)
        a.set_title(self.target + " against " + Plot_var)
        self.plotfig = f

    def draw_plot(self, Plot_var):
        if run_algo_dict[self.train] == "linear_reg":
            self.linear_reg_plot(Plot_var)
        elif run_algo_dict[self.train] == "svm_cls":
            self.svm_cls_plot(Plot_var)
        elif run_algo_dict[self.train] == "k_means_ctn":
            self.k_means_ctn_plot(Plot_var)
        elif run_algo_dict[self.train] == "k_nearest_nbr":
            self.k_nearest_nbr_plot(Plot_var)

    #TODO
    def predict(self):
        pass

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


class Gui:
    def __init__(self, win):
        self.win = win
        self.s = ttk.Style(self.win)
        self.data = pd.DataFrame()
        self.datapath = None
        self.data_x = None
        self.result = "0"
        self.feature_list = []
        self.learning_obj = None
        self.plotfig = None
        self.styling()
        self.draw_gui()

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
        self.mid_frame_3 = ttk.Frame(self.full_frame)
        self.mid_frame_3.pack(expand="true", fill="x")
        self.end_frame = ttk.Frame(self.full_frame)
        self.end_frame.pack(side="bottom", expand="true", fill="x")


        self.load = ttk.Label(self.top_frame, text="Load Your Clean Dataset")
        self.loadbtn = ttk.Button(self.top_frame, text="Load", command=self.load_data)
        self.viewbtn = ttk.Button(self.top_frame, text="View", command=self.view_data)

        self.algolabel = ttk.Label(self.mid_frame_1, text="Select Model You want to train:")
        self.algochooser = ttk.Combobox(self.mid_frame_1)
        self.algochooser['values'] = ["Linear regression", "Support vector machine", "K Means Clustering", "K Nearest Neighbours"]
        self.algochooser.current(0)

        self.selectlbl = ttk.Label(self.mid_frame_2, text="Select Features and target:")
        self.features = ttk.Menubutton(self.mid_frame_2, text="Select Features")
        self.features.menu = Menu(self.features, tearoff=1)
        self.features["menu"] = self.features.menu
        self.varlist = {i:tkinter.IntVar() for i in self.data.columns.to_list()}
        for item in self.varlist:
            self.features.menu.add_checkbutton(label=item, variable=self.varlist[item])
            self.varlist[item].set(1)
        self.target = ttk.Combobox(self.mid_frame_2)
        self.target['values'] = [i for i in self.data.columns]
        self.target.set("Select Target")

        self.trainbtn = ttk.Button(self.mid_frame_3, text="Train", command=self.train_model)
        self.resultlabel = ttk.Label(self.mid_frame_3, text="The accuracy of the model is:")
        self.resultdisp = ttk.Label(self.mid_frame_3, text=f"{float(self.result):.3f}")

        self.drawplotbtn = ttk.Button(self.end_frame, text="Plot", command=self.draw_plot)
        self.plotvar = ttk.Combobox(self.end_frame)
        self.plotvar.set("Select feature to plot")

        self.load.pack(side="left", padx=5, pady=5)
        self.loadbtn.pack(side="left", padx=5, pady=5)
        self.viewbtn.pack(side="left", padx=5, pady=5)

        self.algolabel.pack(side="left", padx=5, pady=5)
        self.selectlbl.pack(side="left", padx=5, pady=5)
        self.algochooser.pack(side="left", padx=5, pady=5)
        self.features.pack(side="left", padx=5, pady=5)
        self.target.pack(side="left", padx=5, pady=5)

        self.trainbtn.pack(side="left", padx=5, pady=5)
        self.resultlabel.pack(side="left", padx=5, pady=5)
        self.resultdisp.pack(side="left", padx=5, pady=5)

        self.plotvar.pack(side="left", padx=5, pady=5)
        self.drawplotbtn.pack(side="left", padx=5, pady=5)

    def styling(self):
        self.s.theme_use("clam")
        self.s.configure("TButton", font=('Times',10,'bold'), foreground="red", background='black')
        self.s.configure("TLabel", font=('Times',10,'bold'), foreground="white", background="black")
        self.s.configure("TFrame", foreground="white", background="black")
        self.s.configure("TCombobox", font=('Times', 10, 'bold'), foreground="black", background="white")
        self.s.configure("TMenubutton", foreground="black", background="white")
        # print(self.s.theme_names())

    # TODO
    def train_model(self):
        loadhp = None
        try:
            loadhp = self.load_hyperparams()
            # getting train_algo. data, features ,target and starting learning
            self.feature_list=[i for i in self.varlist if(self.varlist[i].get()==1)]
            self.learning_obj = Learning(self.algochooser.get(), self.data, self.feature_list, self.target.get(), loadhp)
            self.result = self.learning_obj.result
            self.resultdisp['text'] = f"{float(self.result):.3f}"
            self.plotvar['values'] = [i for i in self.feature_list]
        except Exception as e:
            print("You haven't selected proper settings\n")
            print(e)

    def draw_plot(self):
        Plot_var = self.plotvar.get()
        self.learning_obj.draw_plot(Plot_var)
        self.plotfig = self.learning_obj.plotfig
        self.canvas = FigureCanvasTkAgg(self.plotfig, master=self.end_frame)
        self.canvas.get_tk_widget().pack(side="left", padx=5, pady=5)

    # TODO
    def load_hyperparams(self):
        # Enter hyperparameters
        if self.algochooser.get() == "Linear Regression":
            return None
        elif self.algochooser.get() == "Support vector machine":
            # kernel and c
            self.win.lower()
            hypprm = simpledialog.askstring(title="Hyper Parameters", prompt="Enter kernel,c:")
            return hypprm.split(",")
        elif self.algochooser.get() == "K Means Clustering":
            # n_clusters = 2
            self.win.lower()
            hypprm = simpledialog.askstring(title="Hyper Parameters", prompt="Enter number of clusters:")
            return int(hypprm)
        elif self.algochooser.get() == "K Nearest Neighbours":
            # n_neighbors = 1
            self.win.lower()
            hypprm = simpledialog.askstring(title="Hyper Parameters", prompt="Enter number of neighbours:")
            return int(hypprm)
        else:
            return None


    def load_data(self):
        file = askopenfile(mode='r', filetypes=[('csv files', '*.csv')])
        if file is not None:
            # Using Sniffer to determine separator for pd.read_csv to run
            with open(file.name) as csvfile:
                dialect = csv.Sniffer().sniff(csvfile.read())
                csvfile.seek(0)
                # reader = csv.reader(csvfile, dialect)
            self.data = pd.read_csv(file.name, sep=dialect.delimiter)
            self.datapath = file.name
            self.full_frame.destroy()
            self.draw_gui()
            r = re.search("[a-zA-Z0-9]+.[a-z]+$", file.name)
            self.load['text'] = f"{r.group(0)} has loaded!"
        else:
            messagebox.showinfo("Note", "You haven't loaded clean data")


    def view_data(self):
        # print(self.feature_list)
        if (len(self.feature_list) == 0):
            print(self.data.head(10))
        else:
            print(self.data[self.feature_list])

if __name__ == "__main__":
    # defining tkinter window
    win = tkinter.Tk()
    win.title("Machine Learning Visuals")
    # win.geometry("720x720")
    # win.attributes('-fullscreen', True)
    # TODO
    GUI = Gui(win)

    win.mainloop()

import pandas as pd
import numpy
import tkinter
import tkinter.ttk as ttk
import sklearn
from tkinter import scrolledtext
from tkinter import messagebox
from tkinter.filedialog import *


class learning():
    def __init__(self):
        pass

    def data_checker(self):
        pass

    def linear_reg(self):
        pass

    def svm_cls(self):
        pass

    def k_means_ctn(self):
        pass

    def k_nearest_nbr(self):
        pass
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

def load_data():
    file = askopenfile(mode='r', filetypes=[('csv files', '*.csv',)])
    if file is not None:
        data = pd.read_csv(file)
        print(data.head())
    else:
        messagebox.showinfo("Note", "You haven't loaded clean data")
    # mid_frame.configure(bg="green")
    # top_frame.configure(bg="white")

def main():
    #defining tkinter window
    win = tkinter.Tk()
    win.title("Machine Learning Visuals")
    win.geometry("720x720")
    #win.attributes('-fullscreen', True)
    #defining top,mid and end frames for loading data, Applying ML Algos and showing results
    top_frame = tkinter.Frame(win, borderwidth=1).pack(side="top", fill=BOTH, expand=True)
    mid_frame = tkinter.Frame(win, borderwidth=-1).pack(fill=BOTH, expand=True)
    end_frame = tkinter.Frame(win, borderwidth=2).pack(side="bottom", fill=BOTH, expand=True)

    load = tkinter.Label(top_frame, text = "Load Your Clean Dataset")
    load.pack(side="left", padx=5, pady=5)
    loadbtn = tkinter.Button(top_frame, text = "Load", bg = 'black', fg = 'red', command = load_data)
    loadbtn.pack(side="left", padx=5, pady=5)

    algolabel = tkinter.Label(mid_frame, text="Select Model You want to train")
    algolabel.pack()
    algochooser = ttk.Combobox(mid_frame)
    algochooser['values'] = ("linear_reg", "svm_cls", "k_means_ctn", "k_nearest_nbr")
    algochooser.current(0)
    algochooser.pack()

    win.mainloop()



if __name__ == "__main__" : main()
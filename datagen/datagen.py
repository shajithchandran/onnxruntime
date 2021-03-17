import sys
import getopt
from sklearn import datasets
from sklearn.datasets import make_classification
from pprint import pprint
import numpy as np
import pandas as pd

def usage():
    print("Use this function to generate data for training")
    print("-B : Generates the Boston House Pricing data. It will be saved into the file boston.csv")
    print("-D : Generates data using sklearn make_classification with features=12, samples=1000.... It will be saved in the file mc.csv")
    exit(-1)


def boston_data():
    boston = datasets.load_boston()
    X=pd.DataFrame(boston.data, columns=boston.feature_names)
    y=pd.Series(boston.target)
    data = np.concatenate((X,y[:,None]),axis=1)
    np.savetxt('../data/boston.csv', data, delimiter= ",")
    print("Saved the data to ../data/boston.csv file")

def mc_data(samples=1000, features=12):
    X, y = make_classification(n_samples=samples, n_features=features, n_informative=6, random_state=1)
    data = np.concatenate((X, y[:,None]), axis=1)
    np.savetxt('../data/mc.csv', data, delimiter= ",")
    print("Saved the data to ../data/mc.csv file")



def main(argv):
    try:
        opts, args = getopt.getopt(sys.argv[1:], "BD")
    except getopt.GetoptError:
        usage()

    for opt, arg in opts:
        if opt == '-B':
            boston_data()
        elif opt == '-D':
            mc_data()
        else:
            usage()


if __name__ == "__main__":
    main(sys.argv[1:])

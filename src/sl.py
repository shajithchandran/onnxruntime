import sys
import getopt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from sklearn.metrics import mean_squared_error
from os import path
import numpy as np
import pandas as pd
import onnxruntime as rt
from pprint import pprint
import joblib

def usage():
    print("Usage error")
    print("./sl.py [-C|-R] input.csv")
    print("-R : Use GradientBoostingRegressor model")
    print("-C : Use GradientBoostingClassifier. Please note the output labels in the input data should be classes")
    print("The model will be saved in native and onnx format in ../model/")
    exit()

def main(argv):
    Cflag = Rflag = 0
    try:
        opts, args = getopt.getopt(sys.argv[1:], "CR")
    except getopt.GetoptError:
        usage()

    if len(args) != 1:
        usage()

    fname = args[0]
    
    for opt, arg in opts:
        if opt == '-C':
            Cflag = 1
        elif opt == '-R':
            Rflag = 1
        else:
            usage()

    if ((Cflag + Rflag) != 1):
        usage()

    if not path.exists(fname):
        print ("Error: The file ", argv[0], "doesn't exist")
        exit(-1)
    dataframe = pd.read_csv(fname, header=None)
    array = dataframe.values
    X = array[:,:-1]    #Last but one col is the input features
    y = array[:,-1:].ravel()    #Last col is the label

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)

    if Rflag == 1:
        model = GradientBoostingRegressor()
    elif Cflag == 1:
        model = GradientBoostingClassifier()

    model.fit(X_train, y_train)

    results = model.score(X_test, y_test)
    print("Native Model Score: ", results)

    #Perform the prediction on all the data natively
    native_prediction = model.predict(X)
    model_name = "../models/model_"
    if Cflag == 1:
        model_name += 'C'
    elif Rflag == 1:
        model_name += 'R'

    joblib.dump(model, model_name+'.pkl')

    #Save the mode in onnx format for inferencing using onnxruntime
    initial_type = [('float_input', FloatTensorType([None,X.shape[1]]))]
    onx = convert_sklearn(model, initial_types=initial_type)

    with open(model_name+'.onnx', "wb") as f:
        f.write(onx.SerializeToString())


    #Lets do the inferencing using onnxruntime and compare the results with native prediction
    rtmodel = rt.InferenceSession(model_name+'.onnx')
    input_name = rtmodel.get_inputs()[0].name
    label_name = rtmodel.get_outputs()[0].name

    xarray = X.tolist()
    yarray = y.tolist()
    onnx_prediction = rtmodel.run([label_name], {input_name: xarray})   #Prediction using onnx runtime

    #Lets compare the results. For regression use mse and for classification, just compare!
    if Rflag == 1:
        result = mean_squared_error(native_prediction, onnx_prediction[0])
        print("MSE for native vs onnx prediction is : ", result)
    else:
        match = 0
        total = 0
        idx = 0
        for i in onnx_prediction[0]:
            total += 1
            if i == native_prediction[idx]:
                match += 1
            idx += 1

        print("Native vs ONNX prediction comparison:")
        print("Total Samples:", total, "\nMatched Prediction:", match)

if __name__ == "__main__":
    main(sys.argv[1:])

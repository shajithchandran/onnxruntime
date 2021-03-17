# onnxruntime
Simple exercise to compare the prediction results using the native model vs onnx converted model.

datagen dir contains code to generate data for training. It can generate data for both regression and classification.
For regression, it uses the Boston House Price Dataset.
For classification, it generates the data using sklearn.make_classification().
Data will be saved in data dir in csv format.

src contains the code to perform training on the above data and compare the prediction results b/w native and onnx model.
For regression, it computes the mean square error and for classification, it compares directly the label for each input.

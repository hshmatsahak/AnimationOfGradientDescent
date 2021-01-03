import numpy as np 
import gzip
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import expit
import celluloid
from celluloid import Camera
from matplotlib import animation

# Open MNIST-files
def open_images(filename):
    with gzip.open(filename, "rb") as file:
        data=file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32)

def open_labels(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=8).astype(np.float32)

X_train=open_images("C:\\Users\\hshma\\Documents\\gradient_descent\\train-images-idx3-ubyte.gz").reshape(-1,784).astype(np.float32)
X_train=X_train/255 #rescale

y_train=open_labels("C:\\Users\\hshma\\Documents\\gradient_descent\\train-labels-idx1-ubyte.gz")
oh=OneHotEncoder(categories='auto')
y_train_oh=oh.fit_transform(y_train.reshape(-1,1)).toarray() # one-hot-encoding of y-values
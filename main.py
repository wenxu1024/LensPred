import sys
from PySide2.QtUiTools import *
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from PySide2.QtGui import *

from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from scipy import interpolate
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import keras.backend as K

# example of tending the vgg16 model
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, add
from keras.layers import Flatten
from keras.layers import LSTM, Embedding
from keras.utils import plot_model
from keras.callbacks import Callback

import re

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

img_size = (300, 300, 3)
Dvas = [55.04, 82.56, 110.08, 137.6, 206.4, 275.2, 344, 412.8, 481.6, 550.4, 619.2, 768.4, 960.5, 1152.6, 1344.7, 1536.8, 1728.9, 1921, 2305.2, 2689.4, 3073.6, 3457.8]
tescale = 100
numclasses = 110

def create_model():
    # load model without classifier layers
    vgg = VGG16(include_top=False, input_shape=img_size)
    for layer in vgg.layers:
        layer.trainable = False
    # add new classifier layers
    flat = Flatten()(vgg.output)
    fe1 = Dropout(0.9)(flat)
    fe2 = Dense(256, activation='relu')(fe1)

    # sequence model
    embedding_dim = 200
    inputs2 = Input(shape=(len(Dvas),))
    se1 = Embedding(numclasses, embedding_dim, mask_zero = True)(inputs2)
    se2 = Dropout(0.9)(se1)
    se3 = LSTM(256)(se2)

    #decoder (feed foward) model
    decorder1 = add([fe2, se3])
    decorder2 = Dense(256, activation='relu')(decorder1)
    outputs = Dense(numclasses, activation='softmax')(decorder2)
    model = Model(inputs=[vgg.input, inputs2], outputs=outputs)
    return model

        
class Form(QObject):
    
    def __init__(self, ui_file, parent=None):
        super(Form, self).__init__(parent)
        ui_file = QFile(ui_file)
        ui_file.open(QFile.ReadOnly)
        
        loader = QUiLoader()
        self.window = loader.load(ui_file)
        ui_file.close()
        
        self.line1 = self.window.findChild(QLineEdit, 'trainpictpath')
        self.line2 = self.window.findChild(QLineEdit, 'traincurvepath')
        self.testimgholder = self.window.findChild(QLabel, 'testimgholder')
        self.batchbar = self.window.findChild(QProgressBar, 'batchbar')
        self.epochbar = self.window.findChild(QProgressBar, 'epochbar')
        self.batchbar.setMinimum(0)
        self.batchbar.setMaximum(69)
        self.epochbar.setMinimum(0)
        self.epochbar.setMaximum(19)
        self.tecurvegrpbox = self.window.findChild(QGroupBox, 'tecurvegrpbox')
        
        
        btn1 = self.window.findChild(QPushButton, 'trainpictbutt')
        btn1.clicked.connect(self.trainpictbutt_clicked)

        btn2 = self.window.findChild(QPushButton, 'traincurvebutt')
        btn2.clicked.connect(self.traincurvebutt_clicked)

        btn3 = self.window.findChild(QPushButton, 'selecttestbutt')
        btn3.clicked.connect(self.selecttestbutt_clicked)

        self.trainbutt = self.window.findChild(QPushButton, 'trainbutt')
        self.trainbutt.clicked.connect(self.trainbutt_clicked)

        self.loadmodelbutt = self.window.findChild(QPushButton, 'loadMButt')
        self.loadmodelbutt.clicked.connect(self.loadmodelbutt_clicked)
        
        self.predbutt = self.window.findChild(QPushButton, 'predbutt')
        self.predbutt.clicked.connect(self.predbutt_clicked)

        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        fig = Figure(figsize=(600,600), dpi=72, facecolor=(1,1,1), edgecolor=(0,0,0))
        self.ax = fig.add_subplot(111)
        self.ax.set_xlim(20,4000)
        self.ax.set_ylim(0, 1.2)
        self.ax.set_xscale('log')
        self.ax.set_xlabel('Dva (nm)')
        self.ax.set_ylabel('Lens Transmission Efficiency')
        # generate the canvas to display the plot
        canvas = FigureCanvas(fig)
        layout = QtWidgets.QVBoxLayout(self.tecurvegrpbox)
        layout.addWidget(canvas)

        self.window.show()


    def trainpictbutt_clicked(self):
        self.trainpictpath = str(QFileDialog.getExistingDirectory(self.window, "Select Directory"))
        self.line1.setText(self.trainpictpath)

    def traincurvebutt_clicked(self):
        self.traincurvepath = str(QFileDialog.getExistingDirectory(self.window, "Select Directory"))
        self.line2.setText(self.traincurvepath)

    def selecttestbutt_clicked(self):
        temp = QFileDialog.getOpenFileName(self.window, "Select A Picture", '', "Images (*.png *.xpm *.jpg *.bmp)")
        self.testimgpath, _ = temp
        if self.testimgpath:
            picture = QPixmap(self.testimgpath)
            self.testimgholder.setPixmap(picture)
            self.testimgholder.setScaledContents(True)
        self.ax.clear()
        self.ax.set_xlim(20,4000)
        self.ax.set_ylim(0, 1.2)
        self.ax.set_xscale('log')
        self.ax.set_xlabel('Dva (nm)')
        self.ax.set_ylabel('Lens Transmission Efficiency')
        self.ax.figure.canvas.draw()

    def trainbutt_clicked(self):
        worker = Worker(self.trainpictpath, self.traincurvepath)
        #excecute
        worker.signals.batch_ended.connect(self.update_batch_progress_bar)
        worker.signals.epoch_ended.connect(self.update_epoch_progress_bar)
        self.threadpool.start(worker)

    def update_batch_progress_bar(self, batch):
        self.batchbar.setValue(batch)

    def update_epoch_progress_bar(self, epoch):
        self.epochbar.setValue(epoch + 1)

    def loadmodelbutt_clicked(self):
        temp = QFileDialog.getOpenFileName(self.window, "Select A Model .h5 File", '', "Model (*.h5)")
        self.model_path , _ = temp
        if self.model_path:
            self.model = load_model(self.model_path)
            self.predbutt.setEnabled(1)
        
        
    def predbutt_clicked(self):
        #worker = PredWorker(self, self.testimgpath)
        #worker.predfinished.predfinished.connect(self.update_pred_graph)
        #self.threadpool.start(worker)
        testpath = self.testimgpath
        testimgdata = load_img(testpath, target_size = img_size)
        testimgdata = img_to_array(testimgdata)
        testimgdata = testimgdata.reshape((1, testimgdata.shape[0], testimgdata.shape[1], testimgdata.shape[2]))
        testimgdata = preprocess_input(testimgdata)
        tes = []
        for i in range(len(Dvas)):
            in_seq = [f for f in tes]
            in_seq = pad_sequences([in_seq], maxlen=len(Dvas))
            yhat = self.model.predict([testimgdata, in_seq])
            yhat = np.argmax(yhat)
            tes.append(yhat)
        #scale back
        tes = [f/tescale for f in tes]
        self.update_pred_graph(tes)
        #clear

    def update_pred_graph(self, tes):
        self.ax.clear()
        self.ax.set_xlim(20,4000)
        self.ax.set_ylim(0, 1.2)
        self.ax.set_xscale('log')
        self.ax.set_xlabel('Dva (nm)')
        self.ax.set_ylabel('Lens Transmission Efficiency')
        # generate/update the plot
        self.ax.plot(Dvas, tes)
        #re-draw
        self.ax.figure.canvas.draw()
        #print(tes)


def isImage(path):
    try:
        Image.open(path)
    except IOError:
        return False
    return True


def readCurve(path):
    #print(path)
    df = pd.read_table(path, header=None)
    df = df.dropna()
    return (df[0], df[1])

def data_generator(num_sample_per_batch, picpath, tepath):
    lenspath = 'C:\\Users\\wxu\\Documents\\IPL'
    #picpath = lenspath + "\\IPLPictures"
    #tepath = lenspath + "\\IPLTransmissionCurves"
    lenslist = [f for f in listdir(tepath) if not isfile(join(tepath, f))]
    X1data = list()
    X2data = list()
    Ydata = list()
    n = 0
    while 1:
        for lens in lenslist:
            lenspicpath = join(picpath, lens)
            lenstepath = join(tepath, lens)
            #load transmission curve data
            lenstefile = listdir(lenstepath)[0]
            lensDva, lensTE = readCurve(join(lenstepath, lenstefile))
            f = interpolate.interp1d(lensDva, lensTE, kind='quadratic', fill_value='extrapolate')
            tes = [f(dva) for dva in Dvas]
            tes = [int(te*tescale) if te > 0 else 0 for te in tes]
            tes = [te if te <= 100 else 100 for te in tes]
            imgs = [f for f in listdir(lenspicpath) if isImage(join(lenspicpath, f))]
            for img in imgs:
                #load an image
                imgdata = load_img(join(lenspicpath, img), target_size=img_size)
                #convert imge to numpy array
                imgdata = img_to_array(imgdata)
                #reshape data for the model
                imgdata = imgdata.reshape((1, imgdata.shape[0], imgdata.shape[1], imgdata.shape[2]))
                #prepare the image for the VGG model
                imgdata = preprocess_input(imgdata)[0]
                #split one te curve sequence into multiple image te pairs
                for i in range(1, len(tes)):
                    #split into input and output pair
                    in_seq, out_seq = tes[:i], tes[i]
                    #pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=len(tes))[0]
                    #encode oupput sequence
                    out_seq = to_categorical([out_seq], num_classes = numclasses)[0]
                    #append imgdata to xdata
                    X1data.append(imgdata)
                    X2data.append(in_seq)
                    Ydata.append(out_seq)
                    n += 1
                    if n == num_sample_per_batch:
                        yield [[np.array(X1data), np.array(X2data)], np.array(Ydata)]
                        X1data, X2data, Ydata = list(), list(), list()
                        n = 0
            
class PredWorkerSignals(QObject):
    predfinished = Signal(list)
    
class PredWorker(QRunnable):
    def __init__(self, testimgpath):
        super(PredWorker, self).__init__()
        self.testimgpath = testimgpath
        self.predfinished = PredWorkerSignals()

    @Slot() #QtCore.Slot
    def run(self):
        model = load_model('my_model.h5')
        testpath = self.testimgpath
        testimgdata = load_img(testpath, target_size = img_size)
        testimgdata = img_to_array(testimgdata)
        testimgdata = testimgdata.reshape((1, testimgdata.shape[0], testimgdata.shape[1], testimgdata.shape[2]))
        testimgdata = preprocess_input(testimgdata)
        tes = []
        for i in range(len(Dvas)):
            in_seq = [f for f in tes]
            in_seq = pad_sequences([in_seq], maxlen=len(Dvas))
            yhat = model.predict([testimgdata, in_seq])
            yhat = np.argmax(yhat)
            tes.append(yhat)
        #scale back
        tes = [f/tescale for f in tes]
        self.predfinished.predfinished.emit(tes)

        

#define fit generator CallBack
class MyCustomCallback(Callback):    
    
    def __init__(self, worker):
        super(MyCustomCallback, self).__init__()
        self.worker = worker
    
    def on_train_batch_end(self, batch, logs=None):
        #print(batch)
        self.worker.signals.batch_ended.emit(batch)
        
    def on_epoch_end(self, epoch, logs=None):
        #print(epoch)
        self.worker.signals.epoch_ended.emit(epoch)

class WorkerSignals(QObject):

    batch_ended = Signal(int)
    epoch_ended = Signal(int)
                
class Worker(QRunnable):
    '''
    Worker thread for Keras fitting
    '''
    
    def __init__(self, pictpath, curvepath):
        super(Worker, self).__init__()
        self.pictpath = pictpath
        self.curvepath = curvepath
        self.signals = WorkerSignals()
        
    @Slot() #QtCore.Slot
    def run(self):
        # create model and fit'
        model = create_model()
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['acc'])
        myCustomCallback = MyCustomCallback(self)
        model.fit_generator(generator=data_generator(32, self.pictpath, self.curvepath), workers=0,steps_per_epoch=70, epochs=20, verbose=0, callbacks=[myCustomCallback])
        model.save('my_model.h5') 
        print("Saved model to disk")
    
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = Form('lenspred.ui')
    sys.exit(app.exec_())

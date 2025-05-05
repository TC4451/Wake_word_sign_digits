import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # or ""
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from tensorflow.keras.models import load_model
import hls4ml

seed = 0
np.random.seed(seed)
batch_input_shape = [1709, 5, 5, 1]
tf.random.set_seed(seed)
tf.config.set_visible_devices([], 'GPU') #Disable all GPUs.
##os.environ['PATH'] = os.environ['XILINX_VITIS'] + '/bin:' + os.environ['PATH

model = load_model('./cnn_model.h5')
model.summary()
for layer in model.layers:
    print(layer.get_config())


config = hls4ml.utils.config_from_keras_model(model, granularity='model', backend='Vitis')
#config = hls4ml.utils.config_from_keras_model(model, backend='Vitis')a
hls_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=config, backend='Vitis', output_dir='model_2/hls4ml_prj', part='xcu250-figd2104-2L-e'
)
hls_model.compile()
hls_model.build(csim=False) #, synth=False)

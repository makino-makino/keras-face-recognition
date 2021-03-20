import glob

import keras
from keras.models import load_model
import numpy as np
import grpc
from keras.preprocessing.image import img_to_array, load_img
from icecream import ic
import pickle
from keras.utils import to_categorical, np_utils

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from saas.input_client.service.input_client_pb2_grpc import InputClientStub
from saas.input_client.service.input_client_pb2 import NoParam, Tensor
from saas.model_client.service.model_client_pb2_grpc import ModelClientStub
from saas.model_client.service.model_client_pb2 import ModelInfo, ModelBinaryKeras

model_file = './data/model/model.h5'
model = model = load_model(model_file)

test_images_tmpl = './data/divide/{}/b/*.jpg'

murai_score = []

host = "yahoo-hackathon-gateai-1.japaneast.cloudapp.azure.com"
input_client_port = 5001
model_client_port = 5002

# grpc stub to input_client
channel_input_client = grpc.insecure_channel(
    f'{host}:{str(input_client_port)}', options=[])
api_input_client = InputClientStub(channel_input_client)

# grpc stub to model_client
channel_model_client = grpc.insecure_channel(
    f'{host}:{str(model_client_port)}', options=[])
api_model_client = ModelClientStub(channel_model_client)

# call inpuot_client to generate keys
print("call gen_key")
api_input_client.gen_key(NoParam())

def test(img_path):
    img = img_to_array(load_img(img_path, target_size=(50, 50)))
    img_nad = img_to_array(img)/255
    img_nad = img_nad[None, ...]

    label = ['human', 'murai']
    pred = model.predict(img_nad, batch_size=1, verbose=0)
    score = np.max(pred)
    pred_label = label[np.argmax(pred[0])]

    return score, pred_label


murai_test_images = glob.glob(test_images_tmpl.format("murai"))
print(murai_test_images)

"""
for img_path in murai_test_images:
    score, pred_label = test(img_path)

    print(img_path)
    print('\tname:', pred_label)
    print('\tscore:', score)
    murai_score.append(score)
"""

#print("murai average:", np.mean(murai_score))

score, pred_label = test("./data/divide/murai/b/100.jpg")
ic(score, pred_label)








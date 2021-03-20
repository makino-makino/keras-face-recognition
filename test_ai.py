import time
import grpc
import numpy as np
import pickle

import keras
from keras.utils import to_categorical, np_utils

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from saas.input_client.service.input_client_pb2_grpc import InputClientStub
from saas.input_client.service.input_client_pb2 import NoParam, Tensor
from saas.model_client.service.model_client_pb2_grpc import ModelClientStub
from saas.model_client.service.model_client_pb2 import ModelInfo, ModelBinaryKeras


def dispense_data():
  iris = load_iris()

  data_x = iris.data
  data_y = iris.target
  data_y = np_utils.to_categorical(data_y)

  x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)

  return x_train, x_test, y_train, y_test


if __name__ == "__main__":
  host = "yahoo-hackathon-gateai-1.japaneast.cloudapp.azure.com"
  input_client_port = 5001
  model_client_port = 5002

  model_name = 'nn_iris'
  # raw inference for comparison
  model = keras.models.load_model("modelfiles/h5/nn_iris.h5")


  # grpc stub to input_client
  channel_input_client = grpc.insecure_channel(f'{host}:{str(input_client_port)}', options=[])
  api_input_client = InputClientStub(channel_input_client)

  # grpc stub to model_client
  channel_model_client = grpc.insecure_channel(f'{host}:{str(model_client_port)}', options=[])
  api_model_client = ModelClientStub(channel_model_client)

  # call inpuot_client to generate keys
  print("call gen_key")
  api_input_client.gen_key(NoParam())

  # call model_client to upload h5 file from modelfiles/h5/{model_name}.h5
  print("call load_and_compile_model_from_local_h5")
  model_info = ModelBinaryKeras()
  model_info.config = pickle.dumps(model.get_config())
  model_info.weights = pickle.dumps(model.get_weights())
  model_info.type_info = 'sequential'
  model_info.intermediate_output = 'none'
  api_model_client.compile_model_from_binary_keras(model_info)


  # prepare data for inference
  x, y, _, _  = dispense_data()
  test_data = x[0]

  # call input_cleint to do inference
  print("call predict")

  t1 = time.time()

  data_for_cf = test_data.tolist()
  input_data = Tensor()
  input_data.data = pickle.dumps(data_for_cf)

  # call prediction
  result = api_input_client.predict(input_data)
  result = pickle.loads(result.data)

  t2 = time.time()

  #result_raw = model.predict(test_data)
  result_raw = model.predict(test_data.reshape(1,4))

  print("\n===============")
  print("cipher_result")
  print(result)
  print(np.argmax(result))
  print("\n===============")
  print("raw_result")
  print(result_raw)
  print(np.argmax(result_raw))

  print("\n===============")
  print(f'time for cipher prediction >>> {t2-t1}')




import glob

from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

model_file = './data/model/model.h5'
model = model = load_model(model_file)

test_images_tmpl = './data/divide/{}/b/*.jpg'

murai_score = []

murai_test_images = glob.glob(test_images_tmpl.format("murai"))
print(murai_test_images)
for img_path in murai_test_images:
    img = img_to_array(load_img(img_path, target_size=(50, 50)))
    img_nad = img_to_array(img)/255
    img_nad = img_nad[None, ...]

    label = ['human', 'murai']
    pred = model.predict(img_nad, batch_size=1, verbose=0)
    score = np.max(pred)
    pred_label = label[np.argmax(pred[0])]

    print(img_path)
    print('\tname:', pred_label)
    print('\tscore:', score)
    murai_score.append(score)

human_score = []

human_test_images = glob.glob(test_images_tmpl.format("human"))
print(human_test_images)
for img_path in human_test_images:
    img = img_to_array(load_img(img_path, target_size=(50, 50)))
    img_nad = img_to_array(img)/255
    img_nad = img_nad[None, ...]

    label = ['human', 'murai']
    pred = model.predict(img_nad, batch_size=1, verbose=0)
    score = np.max(pred)
    pred_label = label[np.argmax(pred[0])]

    print(img_path)
    print('\tname:', pred_label)
    print('\tscore:', score)
    human_score.append(score)

print("murai average:", np.mean(murai_score))
print("human average:", np.mean(human_score))

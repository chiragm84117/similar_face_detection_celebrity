# import os
# import pickle
# actors = os.listdir('Bollywood_celeb_face_localized')
#
# filename = []
#
# for actor in actors:
#     for file in os.listdir(os.path.join('Bollywood_celeb_face_localized' , actor)):
#         filename.append(os.path.join('Bollywood_celeb_face_localized',actor,file))
#
# # print(filename)
# # print(len(filename))
#
# pickle.dump(filename,open('filename.pkl','wb'))

# ----------------------------------------------------------------------------------------------------------------------------------------------#

# from tensorflow.kreas.layers import Input,Lambda,Dense,Flatten
import tensorflow.python.keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import pickle
from tqdm import tqdm
filename = pickle.load(open('filename.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

model.summary()

def feature_extractor(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expended_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expended_img)

    result = model.predict(preprocessed_img).flatten()
    return result

features = []

for file in tqdm(filename):
    features.append(feature_extractor(file,model))

pickle.dump(features,open('embedding1.pkl','wb'))







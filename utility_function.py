
import argparse
import numpy as np
import argparse
import json 
import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub

image_path="./test_images/"
loaded_model="./model_project.h5"
model_path="./model_project.h5"

model=tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer},compile=False)
def get_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('-i','--image_path',help='./test_images/',default='./test_images/orange_dahlia.jpg')
    parser.add_argument('-m','--model_path',help='model_path',default='./model_project.h5')
    parser.add_argument('-k','--top_k', help='Top k probabilities of image', default=5, type=int)
    parser.add_argument('-c','--category_names',help='Flower Classes, Total: 102 Classes', default='./label_map.json')

    return parser.parse_args()

def process_images(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, [224, 224])
    image /= 255
    return image.numpy()



#ML Libraries
import trax
from trax import layers as tl
from trax.supervised import training
from trax.fastmath import numpy as fastnp
import numpy as np
import pandas as pd
import random as rnd
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import numpy as np
from tqdm import tqdm
from joblib import Parallel

IMG_WIDTH = 100
IMG_HEIGHT = 100

def process_image(img_path_list):
  images = []
  for file_name in tqdm(img_path_list):
    img = image.load_img(file_name, target_size=(IMG_WIDTH,IMG_HEIGHT,1), color_mode="grayscale")
    img = image.img_to_array(img)
    images.append(img)
  return images


def data_generator(image1, image2, batch_size, shuffle=True):
    """Generator function that yields batches of data

    Args:
        image1 (list): List of transformed (to tensor) images.
        image2 (list): List of transformed (to tensor) images.
        batch_size (int): Number of elements per batch.
        shuffle (bool, optional): If the batches should be randomnized or not. Defaults to True.
    Yields:
        tuple: Of the form (input1, input2) with types (numpy.ndarray, numpy.ndarray)
        NOTE: input1: inputs to your model [img1a, img2a, img3a, ...] i.e. (img1a,img1b) are duplicates
              input2: targets to your model [img1b, img2b,img3b, ...] i.e. (img1a,img2i) i!=a are not duplicates
    """

    input1 = []
    input2 = []

    idx = 0
    len_q = len(image1)
    image_indexes = [*range(len_q)]
    
    if shuffle:
        rnd.shuffle(image_indexes)
    
    while True:
        if idx >= len_q:
            # if idx is greater than or equal to len_q, set idx accordingly 
            idx = 0
            # shuffle to get random batches if shuffle is set to True
            if shuffle:
                rnd.shuffle(image_indexes)
        
        # get images at the `images_indexes[idx]` position in image1 and image2
        img1 = image1[image_indexes[idx]]
        img2 = image2[image_indexes[idx]]
        
        # increment idx by 1
        idx += 1
        # append q1
        input1.append(img1)
        # append q2
        input2.append(img2)
        if len(input1) == batch_size:
            b1 = []
            b2 = []
            for i1, i2 in zip(input1, input2):
                # append left image
                b1.append(i1)
                # append right image
                b2.append(i2)
            # use b1 and b2
            yield ([np.array(b1), np.array(b2)])   
            # reset the batches
            input1, input2 = [], []


# Siamese model
def Siamese(input_shape=(IMG_WIDTH, IMG_HEIGHT), embedding_dim=128, mode='train'):
    """Returns a Siamese model.

    Args:
        input_shape (int, int): Shape of Input Image.
        embedding_dim (int, optional): Depth of the model. Defaults to 128.
        mode (str, optional): 'train', 'eval' or 'predict', predict mode is for fast inference. Defaults to 'train'.

    Returns:
        trax.layers.combinators.Parallel: A Siamese model. 
    """

    def normalize(x):  # normalizes the vectors to have L2 norm 1
        return x / fastnp.sqrt(fastnp.sum(x * x, axis=-1, keepdims=True))
    
    img_processor = tl.Serial(
        tl.Conv(16, (3, 3), padding='VALID'),
        tl.MaxPool(pool_size=(3, 3), padding='VALID'),
        tl.Conv(32, (3, 3), padding='VALID'),
        tl.MaxPool(pool_size=(3, 3), padding='VALID'),
        tl.Fn('Flatten', lambda x: fastnp.reshape(x, (x.shape[0], np.prod(x.shape[1:])))),
        tl.Dense(embedding_dim*2),
        tl.Relu(),
        tl.Dense(embedding_dim*3),
        tl.Relu(),
        tl.Dense(embedding_dim),
        tl.Fn('Normalize', lambda x: normalize(x)),
    )
    
    # Run on Q1 and Q2 in parallel.
    model = tl.Parallel(img_processor, img_processor)
    return model

#  predict
def predict(image1, image2, threshold, model, data_generator=data_generator, verbose=False):
    """Function for predicting if two questions are duplicates.

    Args:
        image1 (str): First image.
        image2 (str): Second image.
        threshold (float): Desired threshold.
        model (trax.layers.combinators.Parallel): The Siamese model.
        data_generator (function): Data generator function. Defaults to data_generator.
        verbose (bool, optional): If the results should be printed out. Defaults to False.

    Returns:
        bool: True if the questions are duplicates, False otherwise.
    """
    image1 = process_image([image1])
    image2 = process_image([image2])
    # pass image1 and image2 arguments of the data generator. Set batch size as 1
    img1, img2 = next(data_generator(image1, image2, batch_size=1))
    # Call the model
    v1, v2 = model((img1, img2))
    # take dot product to compute cos similarity of each pair of entries, v1, v2
    d = np.dot(v1[0], v2[0].T)
    # is d greater than the threshold?
    res = d > threshold
    
    ### END CODE HERE ###
    
    if(verbose):
        # print("Image 1  = ", image1, "\nImage 2  = ", image2)
        print("d   = ", d)
        print("res = ", res)

    return res,d



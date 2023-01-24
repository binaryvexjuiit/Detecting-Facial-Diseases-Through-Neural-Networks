

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

import numpy as np
from imutils import paths
import PIL.Image

"""load image from source
"""
dir_name="dataset/6"
imagePaths = list(paths.list_images(dir_name))

i = 0
ij = 2;
for imagePath in imagePaths:
    # extract the class label from the filename
    image = load_img(imagePath,target_size=(200,200))
    #image = np.rot90(image,1)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    

	# construct the image generator for data augmentation then
	# initialize the total number of images generated thus far
    gen = ImageDataGenerator(
        featurewise_center=True,
       featurewise_std_normalization=True,
       brightness_range=[0.4,1.0],
       rotation_range=25,
       width_shift_range=0.1,
       height_shift_range=0.1,
       horizontal_flip=True,
       shear_range=0.05,
       zoom_range=0.05,
   		fill_mode="nearest")
    j = 0
    i+=1
    #print(image)
    # construct the actual Python generator
    if i == 281:
        ij=1
    for batch in gen.flow(image, batch_size=1, save_to_dir=dir_name,
                      save_prefix=str(i)+str(j), save_format="jpg") :

        j+= 1
        """change value"""
        if j == ij:
            break
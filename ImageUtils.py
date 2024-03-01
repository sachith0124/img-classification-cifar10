import numpy as np

""" This script implements the functions for data augmentation and preprocessing.
"""

def parse_record(record, training):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image

def preprocess_image(image, training):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    if training:
        image = np.pad(image, pad_width=((4, 4), (4, 4), (0, 0)), mode='constant')
        # Randomly crop a [32, 32] section of the image.
        upper_left = np.random.randint(9, size=2)
        image = image[upper_left[0]:upper_left[0]+32, upper_left[1]:upper_left[1]+32,:]
        
        # Randomly flip the image horizontally.
        image = np.fliplr(image) if np.random.randint(2) else image

    # Subtract off the mean and divide by the standard deviation of the pixels.
    image = (image - np.mean(image)) / np.std(image)

    return image
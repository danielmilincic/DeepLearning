ORANGE = '#FFAE4A'

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

"""
Before running this script, make sure to create the augmented dataset
by running src/XRaysAugmentation.py with GENERATION = True. 
"""
def create_combined_histogram(data_folder, name, titel):
    """
    Creates a histogram over all pixel values in the given folder.
    """
    all_pixels = []

    for filename in os.listdir(data_folder):
        with Image.open(os.path.join(data_folder, filename)) as image:
                image_arr = (np.asarray(image) / (2 ** 8 + 1)).astype(np.uint8)
                all_pixels.extend(image_arr.ravel())

    plt.hist(all_pixels, bins=256, color=ORANGE, range=(70, 180))
    plt.title(titel, fontsize = 'x-large')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.ylim(0, 10000000)
    plt.savefig(f"{name}.png")
    plt.close()


create_combined_histogram('data/', "org_data", "Original Images")
create_combined_histogram('new_data/', "augmented_data", "Augmented Images")


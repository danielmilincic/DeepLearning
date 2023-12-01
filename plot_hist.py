import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def create_combined_histogram(data_folder, name):
    all_pixels = []

    # Durchlaufen Sie alle .tiff-Dateien im Datenordner
    for filename in os.listdir(data_folder):
        # Bild öffnen und in ein Numpy-Array konvertieren
        with Image.open(os.path.join(data_folder, filename)) as image:
                image_arr = (np.asarray(image) / (2 ** 8 + 1)).astype(np.uint8)
                all_pixels.extend(image_arr.ravel())

    # Histogramm erstellen
    plt.hist(all_pixels, bins=256, color='blue', alpha=0.7, edgecolor='black')
    # plt.title('Combined Histogram for All Images')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{name}.png")

# Verwenden Sie die Funktion create_combined_histogram mit dem Pfad zu Ihrem Datenordner
# create_combined_histogram('data/', "org_data")
create_combined_histogram('new_data/', "augmented_data")


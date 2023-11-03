# DeepLearning
In an era of rapid advancements in X-ray physics and the growing capabilities of X-ray synchrotron sources, the analysis of tomographic X-ray datasets has become increasingly critical in various scientific, medical, and industrial applications. However, once the raw data are collected, manually segmenting these images is a time-consuming and error-prone process, often plagued by uncertainties and subjectivity. Automating the segmentation process becomes imperative to keep pace with data acquisition rates and to ensure timely scientific discoveries and industrial insights. To address this challenge, in this project, we plan to leverage the power of deep neural networks to automate the segmentation of ptychographic X-ray images, removing the need for human intervention and significantly expediting the analysis process. The primary objective is developing and training a deep neural network based on existing architectures commonly used for other computer vision tasks (e.g., UNet, VGGnet, etc.) The training dataset consists of real-world X-ray images (raw and segmented), and the results will be benchmarked against manually labeled datasets. The project will be supervised by Salvatore De Angelis (sdea@dtu.dk ) and Peter Stanley Jørgensen (psjq@dtu.dk)

## About the dataset
This dataset was obtained using Ptychographic X-ray Tomography on a sample extracted from Solid Oxide Cells.

The microstructure presents three phases (nickel, yttria-stabilized zirconia, and pores). 
Accurately segmenting the three phases is crucial to extract quantitative morphological properties. 

So, the grayscale values need to be reduced to a set of labels, one for each phase. 

In the training dataset, you have:

- data: this folder contains the RAW data (gray-scale) in the TIFF format (16-bit) 
- labels: this folder contains the segmented data (manual segmentation) in the TIFF format (8-bit) 

The project aims to teach a neural network to perform this segmentation, eliminating the manual intervention. 
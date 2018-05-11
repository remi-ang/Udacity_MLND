# Flower Recognition

Capstone project for Udacity Machine Learning Nanodegree

![](/images_samples/813445367_187ecf080a_n.jpg)

## Proposal

Is it possible to identify the variety of a flower from a picture?

The aim of this project is to build a deep convolutional network and take advantage of pre-trained reference networks (Transfer Learning).

The model is then be implemented in an small application returning the variety of the flower from a picture.  

* [validated proposal](proposal.pdf)

## Dataset

The original [Kaggle Flowers recognition dataset](https://www.kaggle.com/alxmamaev/flowers-recognition/data) (zip, 230MB) is a collection of 4326 images of flowers divided in 5 classes:

* Daisy (769 pictures)
* Dandelion (1055 pictures)
* Rose (784 pictures)
* Sunflower (734 pictures)
* Tulip (984 pictures)

## Software requirements

1. Clone the repository

		git clone https://github.com/remi-ang/Udacity_MLND/tree/master/capstone-project

2. Download the [Kaggle Flowers Recogition dataset](https://www.kaggle.com/alxmamaev/flowers-recognition/downloads/flowers.zip/1) and save it in the location `/path/to/capstone-project/flowers`

4. This project requires the following librairies:

	* Python3.6
	* NumPy
	* Pandas
	* Scikit-learn
	* Keras
	* TensorFlow
	* OpenCV

5. Run the Jupyter Notebook

		jupyter notebook flower_recognition.ipynb


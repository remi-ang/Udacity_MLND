## 1. imports

print("importing packages...")
from time import time
start = time()

import logging
from datetime import datetime

import sys
import os

from batch_utilities.batch_utils import plotTrianingHistory, timeElapsed, loadimg

import numpy as np

from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

# set TensorFlow as Keras backend
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras import __version__ as keras_version
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
import cv2

print("All imports done ({})".format(timeElapsed(start)))

## LOG config
logdir = 'log'
if not os.path.exists(logdir):
    os.mkdir(logdir)
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
logfile = os.path.join(logdir, 'batchlog_' + timestamp + '.txt')
print("log : {}".format(logfile))
logging.basicConfig(filename=logfile, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logging.info("Flower Recognition CNN training - BATCH - START")

try:
    ## 1. GENERAL CONFIGURATION
    # -------------------------
    logging.info("1. GENERAL CONFIGURATION")

    # base_model_name = 'VGG16'
    # base_model_name = 'VGG19'
    # base_model_name = 'ResNet50'    # size: 99MB Top-5 Accuracy: 0.929
    # base_model_name = 'InceptionV3' # size: 92MB Top-5 Accuracy: 0.944
    base_model_name = 'Xception'  # size: 88MB Top-5 Accuracy: 0.945

    # random seed
    random_seed = 6
    logging.info("Global random seed: {}".format(random_seed))

    # paths
    data_path = 'flowers'
    plot_path = 'plot'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    saved_models_path = 'saved_models'
    if not os.path.exists(saved_models_path):
        os.makedirs(saved_models_path)

    ## Packages version

    logging.info("Packages versions:")
    logging.info("------------------")
    logging.info("Python: {}".format(sys.version))
    # logging.info("Matplotlib : {}".format(mlp.__version__))
    # logging.info("Seaborn : {}".format(sns.__version__))
    logging.info("Numpy : {}".format(np.__version__))
    logging.info("Scikit-learn : {}".format(sklearn_version))
    logging.info("Keras : {}".format(keras_version))
    logging.info("Keras backend: {}".format(os.environ['KERAS_BACKEND']))
    logging.info("Tensor Flow : {}".format(tf.__version__))
    logging.info("openCV : {}".format(cv2.__version__))

    ## 2. Scan the flowers pictures dataset
    # -------------------------------------
    title = "2. SCAN THE FLOWERS PICTURES DATASET"
    print(title)
    logging.info(title)

    if not os.path.exists(data_path):
        raise Exception("The Flowers Dataset is missing!")

    blacklist_file = "blacklist.txt"  # list of files to be removed drm the dataset

    # read blacklist
    logging.info("reading blacklist: {}...".format(blacklist_file))
    with open(blacklist_file, 'r') as f:
        blacklist = f.readlines()
    blacklist = [data_path + s.strip() for s in blacklist]
    logging.info("{} files to discard defined in blacklist".format(len(blacklist)))

    logging.info("Collecting files from data folder : {}".format(data_path))
    data = load_files(data_path, load_content=False)
    all_flowers_files = np.array([s.replace('\\', '/') for s in data["filenames"]])
    logging.info("{} flower pictures found".format(all_flowers_files.shape[0]))

    # identify blacklist indexes
    isvalid_file = [True if f not in blacklist else False for f in all_flowers_files]
    flowers_files = all_flowers_files[isvalid_file]
    logging.info("{} flower pictures kept after black list filtering".format(flowers_files.shape[0]))

    # flowers_targets = np_utils.to_categorical(data["target"],5)
    flowers_targets = data["target"][isvalid_file]
    flowers_target_names = data["target_names"]

    logging.info('types of flower found: {}'.format(flowers_target_names))

    ## Stratified Train/Valid/test split of the dataset
    title = "3.STRATIFIED TRAIN/TEST/SPLIT OF THE DATASET"
    print(title)
    logging.info(title)

    np.random.seed(random_seed)

    test_size = .15

    logging.info("Splitting train+valid/test flower files (split ratio: {} / {} %)...".format((1-test_size) * 100, test_size * 100))
    flowers_files_train_valid, flowers_files_test, flowers_targets_train_valid, flowers_targets_test = train_test_split(
        flowers_files, flowers_targets,
        test_size=test_size,
        stratify=flowers_targets,
        random_state=random_seed)


    valid_size = .1
    logging.info("Splitting train/valid flower files (split ratio: {} / {} %)...".format((1-valid_size) * 100, valid_size * 100))
    flowers_files_train, flowers_files_valid, flowers_targets_train, flowers_targets_valid = train_test_split(
        flowers_files_train_valid, flowers_targets_train_valid,
        test_size=valid_size,
        stratify=flowers_targets_train_valid,
        random_state=random_seed)

    logging.info("{} training flowers files.".format(flowers_files_train.shape[0]))
    logging.info("{} validation flowers files.".format(flowers_files_valid.shape[0]))
    logging.info("{} testing flowers files.".format(flowers_files_test.shape[0]))

    ## 4. Load dataset in memory

    title = "4. LOAD THE DATASET IN MEMORY"
    print(title)
    logging.info(title)

    x_train = np.array([loadimg(f) for f in flowers_files_train]).reshape(-1, 299, 299, 3)
    y_train = np_utils.to_categorical(flowers_targets_train)
    logging.info("training set shape : {}".format(x_train.shape))

    x_valid = np.array([loadimg(f) for f in flowers_files_valid]).reshape(-1, 299, 299, 3)
    y_valid = np_utils.to_categorical(flowers_targets_valid)
    logging.info("training set shape : {}".format(x_train.shape))

    x_test = np.array([loadimg(f) for f in flowers_files_test]).reshape(-1, 299, 299, 3)
    y_test = np_utils.to_categorical(flowers_targets_test)
    logging.info("testing set shape : {}".format(x_test.shape))

    ## 5. Data augmentation - Create images generators

    title = "5. DATA AUGMENTATION - CREATE IMAGES GENERATORS"
    print(title)
    logging.info(title)

    batch_size = 32 # default: 32

    # training data generator - scaling and data augmentation
    logging.info("Initializing training data generator...")
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    train_datagen.fit(x_train)
    # train_generator = train_datagen.flow_from_directory(
    #         os.path.join('.', data_root, 'train'),
    #         target_size=(150, 150),  # all images will be resized to 150x150
    #         batch_size=batch_size,
    #         class_mode='categorical')

    # validation data generator - scaling and data augmentation
    logging.info("Initializing validation data generator...")
    valid_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    valid_datagen.fit(x_valid)

    # test data generator - scaling only
    logging.info("Initializing testing data generator...")
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen.fit(x_test)

    # test_generator = test_datagen.flow_from_directory(
    #        os.path.join('.', data_root, 'test'),
    #         target_size=(150,150),
    #         batch_size=batch_size,
    #         class_mode='categorical')

    ## 7. Convolutional Neural Network conception, Transfer Learning from Xception

    title = "7. CONVOLUTIONAL NEURAL NETWORK CONCEPTION, TRANSFER LEARNING"
    print(title)
    logging.info(title)

    logging.info("base model: {}".format(base_model_name))
    if base_model_name == 'Xception':
        from keras.applications.xception import Xception
        base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    elif base_model_name == 'VGG16':
        from keras.applications.vgg16 import VGG16
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    elif base_model_name == 'VGG19':
        from keras.applications.vgg19 import VGG19
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    elif base_model_name == 'ResNet50':
        from keras.applications.resnet50 import ResNet50
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    elif base_model_name == 'InceptionV3':
        from keras.applications.inception_v3 import InceptionV3
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    for layer in base_model.layers:
        layer.trainable = False

    # new layers:
    # -----------

    x = Flatten()(base_model.output)
    x = Dense(500, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(5, activation='softmax', name='fc2')(x)

    # Model class API, take tensor in input and tensor in output
    model = Model(inputs=base_model.input, outputs=x)
    opt = Adam(lr=1e-3, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    ## 8. Train the model

    title = "8. TRAIN THE MODEL"
    print(title)
    logging.info(title)

    batch_size = 16
    epochs = 100

    best_weights_h5 = os.path.join(saved_models_path, 'weights.best.flowers_recognition.{}.hdf5'.format(base_model_name))

    checkpointer = ModelCheckpoint(filepath=best_weights_h5,
                                   verbose=1, save_best_only=True)

    training_start = time()
    logging.info("Train the model...")
    training_hist = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size),
                                        steps_per_epoch=x_train.shape[0] // batch_size,
                                        epochs=epochs,
                                        verbose=1,
                                        callbacks=[checkpointer],
                                        validation_data=valid_datagen.flow(x_valid, y_valid, batch_size=batch_size),
                                        validation_steps=x_valid.shape[0] // batch_size)

    logging.info("Training done ({}).".format(timeElapsed(training_start)))

    plot_file = os.join.path(plot_path, 'training_history_{}_{}.png'.format(base_model_name, timestamp))
    plotTrianingHistory(plot_file, training_hist)

    ## 9. Load the Model with the Best Validation Accuracy
    title = "9. LOAD THE MODEL WITH THE BEST VALIDATION ACCURACY"
    print(title)
    logging.info(title)

    model.load_weights(best_weights_h5)

    ## 10. Calculate Classification Accuracy on Test Set
    title = "10. CALCULATE CLASSIFICATION ACCURACY ON TEST SET"
    print(title)
    logging.info(title)

    print("evaluate model accuracy...")
    score = model.evaluate(x_test, y_test, verbose=1)

    logging.info('\n', 'Test accuracy:', score[1])

    ## END

    logging.info("Flower Recognition CNN training - BATCH - END - {}".format(timeElapsed(start)))

except Exception as e:

    logging.error(e)

print("done.")

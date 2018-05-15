"""
custum utilities functions
"""
import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from shutil import copyfile

def print_list(in_list, items_name="items", max_disp = 10):
    """
    print a title with number of items in a list
    print bullet pointed maximum amount of items  
    """
    list_size = len(in_list)
    if max_disp < list_size:
        max_disp = list_size
        
    print("{} {}:".format(list_size, items_name))
    print("\n".join(["\to {}".format(x) for x in in_list]))
    
    if list_size > max_disp:
        print("\to ... (and {} more)".format(len(list_size) - max_disp))

def plot_images(img_paths, titles=None):
    """
    plot images in subplots form a list of paths and optionally titles 
    """
    
    # check that titles length match number of images
    show_titles = checkTitlesLength(titles, img_paths)

    # define subplots settings
    n_rows, max_img_per_row = defineNrows(img_paths)
        
    plt.figure(figsize=(20, n_rows * 5))
    
    for i, img_path in enumerate(img_paths):
        plt.subplot(n_rows, max_img_per_row, i+1)
        img = cv2.imread(img_path)
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(cv_rgb)
        if show_titles:
            plt.title(titles[i])     
            
    plt.show()

def flowersDistBarplot(flowerTypesSets, class_names, titles=None):
    """
    create flowers types distirbution barplots
    """
    
    flowerTypesSets = castAsList(flowerTypesSets)
    titles = castAsList(titles)
    
    # check that tiltes length match datasets length
    show_titles = checkTitlesLength(titles, flowerTypesSets)
    
    # define subplots settings
    n_rows, max_img_per_row = defineNrows(flowerTypesSets)
    
    plt.figure(figsize=(15,n_rows * 5))
 
    for i, flowerTypesSet in enumerate(flowerTypesSets):
        _, counts = np.unique(flowerTypesSet, return_counts=True)
        plt.subplot(n_rows, max_img_per_row, i+1)
        p = sns.barplot(x=class_names, y=counts)
        if show_titles:
            plt.title(titles[i], fontsize=16)
        plt.xlabel("Variety", fontsize=14)
        plt.ylabel("Count", fontsize=14)

        txt_buff = round(max(counts) * 0.01) # just to leave some space over the column tip
        for i, count in enumerate(counts):
            p.text(i, count+txt_buff, count)  
    
    plt.show()  
	
def copy_pictures(img_paths, train_or_test, root='data', ):
    
    assert train_or_test in ['train', 'test']
    print("copying {}ing images in ./data/{} subdirectories...".format(train_or_test, train_or_test))

    for img_path in tqdm(img_paths):
        target_path = os.path.join(root, train_or_test, *img_path.rsplit('/',2)[-2:])
        copyfile(img_path, target_path)
		
# Support functions 

def castAsList(X):
    if not isinstance(X, list):
        X = [X]
    return X

def defineNrows(X):
    if len(X) > 1:
        max_img_per_row = 5
        if len(X) < max_img_per_row:
            max_img_per_row = len(X)
        
        n_rows = np.ceil(1.0 * len(X) / max_img_per_row)
    else:
        # if only 1 picture
        max_img_per_row = 1
        n_rows = 1  
    
    return n_rows, max_img_per_row

def checkTitlesLength(titles, ref_list):
    show_titles = False
    if titles is not None:
        if len(titles) == len(ref_list):
            show_titles = True
        else:
            print("Number of titles ({}) does not match number of files ({})".format(
                len(titles), len(ref_list)))
    return show_titles
 
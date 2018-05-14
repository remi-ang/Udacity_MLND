"""
custum utilities functions
"""
import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt

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
    
    show_titles = False
    if titles is not None:
        if len(titles) == len(img_paths):
            show_titles = True
        else:
            print("Number of titles ({}) does not match number of files ({})".format(
                len(titles), len(img_paths)))

    # define subplots settings
    if len(img_paths) > 1:
        max_img_per_row = 5
        n_rows = np.ceil(1.0 * len(img_paths) / max_img_per_row)
    else:
        # if only 1 picture
        max_img_per_row = 1
        n_rows = 1
        
    plt.figure(figsize=(20, n_rows * 5))
    
    for i, img_path in enumerate(img_paths):
        plt.subplot(n_rows, max_img_per_row, i+1)
        img = cv2.imread(img_path)
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(cv_rgb)
        if show_titles:
            plt.title(titles[i])     
            
    plt.show()  
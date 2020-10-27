# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 19:41:59 2020

@author: paolo
"""

import hashlib
# from scipy.misc import imread, imresize, imshow
# from  imageio import imread
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# %matplotlib inline
import time
import numpy as np
import os
import imagehash
from PIL import Image
import warnings




# def file_hash(filepath):
#     with open(filepath, 'rb') as f:
#         return md5(f.read()).hexdigest()

#%%
hash_collection={}
hashlist=[]
for entry in os.scandir('C:/projects/find_object/dataset/train'):
    if entry.is_dir() :
        print(entry.path)
        
        file_list = os.listdir(entry.path)
        
        duplicates = []        
        hash_keys = dict()
        for index, filename in  enumerate(os.listdir(entry.path)):  #listdir('.') = current directory
            filepath = os.path.join(entry.path,filename)
            if os.path.isfile(filepath):
                # with open(filepath, 'rb') as f:
                    # filehash = hashlib.md5(f.read()).hexdigest()
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", category=UserWarning)
                    im = Image.open(filepath)
                    filehash = imagehash.phash(im)
                    im.close()
                    if len(w) > 0: print(filepath + " creating a warning")
                        
                
                if filehash not in hash_keys: 
                    hash_keys[filehash] = index
                else:
                    duplicates.append((index,hash_keys[filehash]))
                
                if filehash in hash_collection:
                    hash_collection[filehash].append(filepath)
                else:
                    hash_collection[filehash]=[]                    
                    hash_collection[filehash].append(filepath)
                    
    hashlist.append({entry.path : hash_keys})
#%%
for filepath_per_hash in hash_collection.values():
    if (len(filepath_per_hash)) > 1:
        path_and_size = {}
        for file_path in filepath_per_hash:
            print(file_path)
            im = Image.open(file_path)
            width, height = im.size
            path_and_size[file_path] = width * height
            im.close()
        
        bigger_img = max( path_and_size.values())
        for file_path_to_remove in filepath_per_hash:
            if path_and_size[file_path_to_remove] != bigger_img :
                 print( "removing: " +  file_path_to_remove)
                 os.remove(file_path_to_remove)
    
    # if (len(filepath_per_hash)) > 1:
    #     for file_path in filepath_per_hash:
    #         print(file_path)
    #         im = Image.open(file_path)
    #         width, height = im.size
    #     to_remove = filepath_per_hash.pop()
    #     print( "Removein: " +  to_remove)
    #     # os.remove(to_remove)
import pandas as pd
from scipy.misc import imread, imresize
from os import listdir
from os.path import splitext
from random import shuffle, seed

def load_image(path,shape=(299,299)):
    readed_img = imread(path)
    return imresize(readed_img,shape)

def extend_children(path,ftype=False):
    allpaths = [path+'/'+child for child in listdir(path)]
    if ftype != False:
# remember to include the period in ftype (ie .jpg)
# pass '' to include only folders
        ret = []
        for v in allpaths:
            if splitext(v)[1] == ftype:
                ret.append(v)
    else:
        ret=allpaths
    return ret

def extend_generation(paths,ftype=False):
    ret = []
    for path in paths:
        ret += extend_children(path,ftype)
    return ret

def load_txt_as_df(fpath):
# open txt file and read lines
    with open(fpath) as f:
        lines = [line.rstrip('\n').split('\t') for line in f]
# convert list to dict
    data_dict = {}
    for col, heading in enumerate(lines[0]):
        data_dict[heading] = [lines[r+1][col] for r in range(len(lines)-1)]
# make pandas dataframe from the dict
    df = pd.DataFrame(data=data_dict)
    return df

def format_image_path(dpath,user_id,face_id,original_image,aligned=True):
    if aligned:
        subfolder = '/aligned/'
        prefix = 'landmark_aligned_face.'
    else:
        subfolder = '/faces/'
        prefix = 'coarse_tilt_aligned_face.'
    fname = prefix+face_id+'.'+original_image
    fpath = dpath+subfolder+user_id+'/'+fname
    return fpath

def format_from_index(df,index,aligned=True):
    ui = list(df['user_id'])[index]
    fi = list(df['face_id'])[index]
    oi = list(df['original_image'])[index]
    fpath = format_image_path('data',ui,fi,oi,aligned)
    return fpath

def shuffle_xy(x,y,shuffleseed=False):
    if shuffleseed:
        seed(shuffleseed)
    shuffler = list(range(len(x)))
    shuffle(shuffler)
    new_x = [x[i] for i in shuffler]
    new_y = [y[i] for i in shuffler]
    return new_x,new_y

def one_hot(index,cols):
    one_hot_vector = [0 for _ in range(cols)]
    one_hot_vector[index] = 1
    return one_hot_vector
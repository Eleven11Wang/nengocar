
import os
import cv2
import time

st = time.time()
import nengo
import random
import nengo_dl
import numpy as np
#import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import rescale, resize, downscale_local_mean

import dlModel
import utilsFunction
import functionList
import warnings
import car_movement
warnings.filterwarnings("ignore")
end = time.time()
print("all package imported time :{}".format(end-st))


name = "test007.jpg"
h = 480
w = 640

if __name__ == '__main__':
    print("image_processing_begain")
    st = time.time()
    minibatch_size = 30
    parm_path = "./face_mnist_params20"

    sim = nengo_dl.Simulator(dlModel.net, minibatch_size=minibatch_size)
    sim.load_params(parm_path)
    out_p_filt=dlModel.out_p_filt
    ed = time.time()
    print("load model time ={:.2f} second".format(ed - st))


    total_location_ls = []
    for round_num in range(3):
        print("round_number : {}".format(round_num))

        img_gray, processed_image = functionList.processing_image(name, h, w)
        new_processed_image = functionList.re_process_processed_img(processed_image,round_num)
        if (sum(sum(processed_image))) >  (round_num+1)**(round_num+1)*28*28:
            #print(sum(sum(processed_image)))
            pass
        else:

            temp_pos_ls, temp_image_ls = functionList.crop_sub_image(img_gray, new_processed_image, round_num)
            location_ls = functionList.dealing_layer_data(temp_image_ls, temp_pos_ls,sim,out_p_filt)
            #print(location_ls)
            if location_ls:
                car_movement.car_act_found()
                #utilsFunction.add_rectangle(img_gray, location_ls)
            else:
                car_movement.car_act_not_found()
            total_location_ls.append(location_ls)
        h = h // 2
        w = w // 2
    total_location_ls = [loc for loc_ls in total_location_ls for loc in loc_ls ]
    print(total_location_ls)
    ed2 = time.time()
    print("model running time ={:.2f} second".format(ed2 - ed))


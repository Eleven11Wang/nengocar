print("programe start")
import os
import cv2
import time
import nengo
import random
import nengo_dl
import numpy as np
import seaborn as sns
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
import movement_maker_new2 as mk
import HiwonderSDK.Board as Board 
import camera_test
import utilsFunction
cv2.namedWindow("test")
def acquire(num):
    while True: 
        frame = MyCamera.getframe()
        if frame is not None:
            cv2.imshow("test",frame)
            key = cv2.waitKey(1)
            cv2.imwrite("acquired_frame_{}.png".format(num),frame)
            break
    return frame 

name = "acquired_frame_2.png"
h = 480
w = 640
MyCamera = camera_test.USBCamera((480,360))
if __name__ == '__main__':
    print("image_processing_begain")
    st = time.time()
    minibatch_size = 30
    acquire(-1)
    parm_path = "./face_mnist_params20"

    sim = nengo_dl.Simulator(dlModel.net, minibatch_size=minibatch_size)
    sim.load_params(parm_path)
    out_p_filt=dlModel.out_p_filt
    ed = time.time()
    print("load model time ={:.2f} second".format(ed - st))

    
    image = acquire(0)
    total_location_ls = []
    for round_num in range(3):
        print("round_number : {}".format(round_num))
        img_gray, processed_image = functionList.processing_image(image, h, w)
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
                time.sleep(2)
                xl,yl = zip(*location_ls)
                rtx= np.mean(xl)
                rt_val = utilsFunction.to_binary(rtx)
                #utilsFunction.add_rectangle(img_gray, location_ls)
                mk.make_movement(rt_val)
                Board.setMotor(2,0)
                Board.setMotor(1,0)
            else:
                car_movement.car_act_not_found()
                time.sleep(2)
            total_location_ls.append(location_ls)
        h = h // 2
        w = w // 2
    #total_location_ls = [loc for loc_ls in total_location_ls for loc in loc_ls ]
    #xl,yl = zip(*total_location_ls)
    #rtx = np.mean(xl)
    #rt_val = utilsFunction.to_binary(rtx)
    #print(rt_val)
    #print(total_location_ls)
    ed2 = time.time()
    print("model running time ={:.2f} second".format(ed2 - ed))

    #mk.make_movement(rt_val)
    #Board.setMotor(1,0)
    #Board.setMotor(2,0)


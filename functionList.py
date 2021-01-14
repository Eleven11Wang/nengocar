
import os
import cv2
import time
import nengo
import random
import nengo_dl
import numpy as np
#import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import rescale, resize, downscale_local_mean

import utilsFunction as func
import dlModel


def processing_image(image, h, w):
    #image = func.read_img(name, h, w)
    img_gray = func.rgb2gray(image)
    img_gray_localization = func.rgb2gray(image)
    YUV = func.RGB2YUV(image)
    Cr = YUV[:, :, 1]
    Cb = YUV[:, :, 2]
    Cr = cv2.blur(Cr, (7, 7))
    Cb = cv2.blur(Cb, (7, 7))
    Cr = Cr.astype('uint8')
    Cb = Cb.astype('uint8')

    #     Cr[Cr>120]=0
    #     Cr[Cr<100]=0
    #     Cb[Cb<140]=0
    #     Cb[Cb>170]=0
    _, Cr = cv2.threshold(Cr, 0, 255, cv2.THRESH_BINARY +
                          cv2.THRESH_OTSU)  # OTSU 二值化
    _, Cb = cv2.threshold(Cb, 0, 255, cv2.THRESH_BINARY +
                          cv2.THRESH_OTSU)  # OTSU 二值化

    # erosion_Cb = morphology(Cb)
    # erosion_Cr = morphology(Cr)
    erosion_Cb = Cb
    erosion_Cr = Cr

    img_gray_localization[erosion_Cr != 0] = 0
    img_gray_localization[erosion_Cb == 0] = 0
    # img_gray_localization[erosion_Cr==0]=0

    #fig, axs = plt.subplots(2, 2, figsize=(10, 10), facecolor='w', edgecolor='k')
    #axs = axs.ravel()
    #sns.heatmap(erosion_Cr,ax=axs[0])
    #sns.heatmap(erosion_Cb,ax=axs[1])
    #sns.heatmap(img_gray_localization, ax=axs[2])
    #sns.heatmap(img_gray,ax=axs[3])
    #plt.tight_layout()
    #plt.savefig("test_region_{}.{}.png".format(h,w))
    #plt.close()
    return img_gray, img_gray_localization


def find_link_area(arr):
    arr_dict = {} # x.y
    arr_pos_dict={}
    h, w = arr.shape
    i, j = 0, 0
    cnt = 0
    while i < h:
        j = 0
        while j < w:
            if arr[i, j] == 1:
                if not arr_dict:
                    pos_to_add = (i / h, j / w)
                    arr_dict[pos_to_add] = 1
                    arr_pos_dict[pos_to_add]=[(i,j)]

                else:
                    for pos, v in arr_dict.items():
                        if abs(pos[0] * h - i) < (0.2 * h) or abs(pos[1] * w - j) < (0.2 * w):
                            pos_to_add = pos
                        else:
                            pos_to_add = (i / h, j / w)
                if pos_to_add not in arr_dict.keys():
                    arr_dict[pos_to_add] = 1
                    arr_pos_dict[pos_to_add]=[(int(pos_to_add[0]*h),int(pos_to_add[1]*w))]

            while i< h and j < w and (arr[i, j] == 1):
                arr_dict[pos_to_add] += 1
                arr_pos_dict[pos_to_add].append((i,j))
                cnt += 1
                j += 1
            j += 1
            cnt += 1
        i += 1
        if cnt > h * w:
            break
        # print(arr_dict)
        # for k,v in arr_pos_dict.items():
        #     print(k,len(v))
    return arr_dict,arr_pos_dict


def filter_processed_area(arr_dict, arr_pos_dict, processed_image, round_num):
    new_processed_image = processed_image[:, :]
    round_num = round_num + 1
    small_remove = 28 * 28 * round_num
    big_remove = 28 * 28 * round_num ** round_num
    #print(round_num-1,small_remove,big_remove)
    def remove_area(r_pos):
        pos_ls = arr_pos_dict[r_pos]
        #print(pos_ls[:10])

        for pos in pos_ls:
            new_processed_image[pos[0], pos[1]] = 0

    for r_pos, v in arr_dict.items():
        x_pos, y_pos = r_pos
        flag=  0
        if x_pos < 0.1 or y_pos < 0.1:
            if v < small_remove :
                remove_area(r_pos)
                flag = 1
        if v > big_remove:
            remove_area(r_pos)
            flag = 1
        # print(r_pos)
        # if flag ==1:
        #     print("removed")
        # else:
        #     print("saved")
    return new_processed_image

def re_process_processed_img(processed_image,round_num):
    processed_image[processed_image > 1] = 1
    arr_dict, arr_pos_dict = find_link_area(processed_image)
    #print(arr_dict)
    new_processed_image = filter_processed_area(arr_dict, arr_pos_dict, processed_image, round_num)
    #fig = plt.figure()
    #sns.heatmap(new_processed_image)
    #plt.savefig("processed_area_{}.png".format(round_num))
    #plt.close()
    return new_processed_image


def crop_sub_image(img_gray, processed_image, round_number):
    x_min = 0
    y_min = 0
    x_min = 0
    y_min = 0
    temp_pos_ls = []
    temp_image_ls = []
    h, w = img_gray.shape

    while (x_min + 28) < w:
        y_min = 0
        while y_min + 28 < h:
            flag = 0
            croped_processed_image = processed_image[y_min:y_min + 28, x_min:x_min + 28]

            center_pos = ((x_min + 14) / w, (y_min + 14) / h)

            if sum(sum(croped_processed_image)) > 100 :
                croped_img = img_gray[y_min:y_min + 28, x_min:x_min + 28]
                croped_img = croped_img.reshape(28 * 28)
                if np.std(croped_img) < 30:
                    pass
                else:
                    temp_image_ls.append(croped_img)
                    temp_pos_ls.append(center_pos)
            y_min += 7
        x_min += 7
    return temp_pos_ls, temp_image_ls


def prepare_data_for_nengo_format(image_array, label_array):
    n_steps = 30
    while (len(image_array) < 30):
        image_array = np.concatenate([image_array, image_array], axis=0)
        label_array = label_array + label_array
    #print(image_array.shape)
    test_images_nengo = np.tile(image_array[:, None, :], (1, n_steps, 1))
    test_labels_nengo = np.tile(label_array[:, None, None], (1, n_steps, 1))
    return test_images_nengo, test_labels_nengo


def filter_prediction(rt):
    rt_numpy= rt.numpy()
    rt_last = rt_numpy[-5:,:]
    mean_val = rt_last.mean(axis =0)
    top_idx = (-mean_val).argsort()[:2]
    top = top_idx[0]
    second = top_idx[1]
    wanted= False
    if mean_val[top] > 0.8 and mean_val[second] < 0.3 *mean_val[top]:
        wanted = True
    return top,mean_val[top],wanted



def make_prediction_ls(test_images,sim,out_p_filt):
    data = sim.predict(test_images)
    out_p_filt_data = data[out_p_filt]
    prediction_ls =[]
    wanted_ls = []
    for idx in range(len(out_p_filt_data)):
        rt=tf.nn.softmax(out_p_filt_data[idx])
        which,val,wanted = filter_prediction(rt)
        prediction_ls.append((which,val))
        wanted_ls.append(wanted)
    return prediction_ls,data,wanted_ls

def dealing_layer_data(pyramid_ls, pos_ls, sim,out_p_filt):
    if not pyramid_ls:
        return []
    test_data = np.concatenate(pyramid_ls).reshape(-1, 28 * 28)
    test_images_nengo, test_labels_nengo = prepare_data_for_nengo_format(test_data, np.array([4] * len(test_data)))
    prediction_ls, data, wanted_ls = make_prediction_ls(test_images_nengo, sim,out_p_filt)
    which_ls, val_ls = zip(*prediction_ls)
    location_ls = []
    #     print(which_ls)
    #     print(wanted_ls)
    #     print(pos_ls)
    for idx, label in enumerate(which_ls):
        if idx >= len(pos_ls):
            break

        if wanted_ls[idx] and label <= 4:
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(test_data[idx].reshape(28, 28))
            plt.title(np.std(test_data[idx]))
            plt.subplot(1, 2, 2)
            plt.plot(tf.nn.softmax(data[out_p_filt][idx]))
            plt.legend([str(i) for i in range(5)], loc="upper left")
            plt.show()
        if label == 4 and wanted_ls[idx]:
            location_ls.append(pos_ls[idx])
    return location_ls

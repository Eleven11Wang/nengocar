import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
import tensorflow as tf

import cv2

def RGB2YUV(rgb):
    m = np.array([[0.29900, -0.16874, 0.50000],
                  [0.58700, -0.33126, -0.41869],
                  [0.11400, 0.50000, -0.08131]])

    yuv = np.dot(rgb, m)
    yuv[:, :, 1:] += 128.0
    return yuv

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def read_img(name, h, w):
    path = name
    img = plt.imread(path)
    # img = irgb2gray(img)

    img = resize(img, (h, w))
    img = img * 255
    img = img.astype(int)
    # img = img.astype('uint8')

    return img


def morphology(img):
    open_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opend = cv2.morphologyEx(img, cv2.MORPH_OPEN, open_element)
    # 腐蚀
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(opend, kernel, iterations=3)
    return erosion

def processing_image(name,h,w):
    image = read_img(name,h,w)
    img_gray = rgb2gray(image)
    img_gray_localization = rgb2gray(image)
    YUV = RGB2YUV(image)
    Cr= YUV[:,:,1]
    Cb= YUV[:,:,2]
    Cr = cv2.blur(Cr, (7, 7))
    Cb = cv2.blur(Cb, (7, 7))
    Cr=Cr.astype('uint8')
    Cb=Cb.astype('uint8')
    #plt.figure()
    #sns.heatmap(Cr)
    #plt.figure()
    #sns.heatmap(Cb)
    Cr[Cr>120]=0
    Cr[Cr<100]=0
    Cb[Cb<140]=0
    Cb[Cb>170]=0
#     _, Cr = cv2.threshold(Cr, 0, 255, cv2.THRESH_BINARY +
#                           cv2.THRESH_OTSU)  # OTSU 二值化
#     _, Cb = cv2.threshold(Cb, 0, 255, cv2.THRESH_BINARY +
#                           cv2.THRESH_OTSU)  # OTSU 二值化

    erosion_Cb = morphology(Cb)
    erosion_Cr = morphology(Cr)
    img_gray_localization[erosion_Cb==0]=0
    img_gray_localization[erosion_Cr==0]=0
    #img_gray[erosion_Cr!=0]=0
    #plt.figure()
    #plt.imshow(img_gray_localization,cmap ="gray")
    return img_gray,img_gray_localization


def crop_sub_image(img_gray,processed_image):
    x_min =0
    y_min =0
    x_min =0
    y_min =0
    temp_pos_ls = []
    temp_image_ls= []
    h,w = img_gray.shape
    while (x_min+28) < w:
        y_min = 0
        while y_min+28 < h:
            croped_processed_image = processed_image[y_min:y_min+28,x_min:x_min+28]
            if sum(sum(croped_processed_image)) != 0:
                croped_img = img_gray[y_min:y_min+28,x_min:x_min+28]
                croped_img = croped_img.reshape(28*28)
                if np.std(croped_img)< 30 :
                    # > x>
                        pass
                else:
                    temp_image_ls.append(croped_img)
                    temp_pos_ls.append(((x_min+14)/w,(y_min+14)/h))
            y_min+=7
        x_min+=7
    return temp_pos_ls,temp_image_ls


def prepare_data_for_nengo_format(image_array, label_array):
    n_steps = 30
    while (len(image_array) < 30):
        image_array = np.concatenate([image_array, image_array], axis=0)
        label_array = label_array + label_array
    print(image_array.shape)
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
    if mean_val[top] > 0.4 and mean_val[second] < 0.5 *mean_val[top]:
        wanted = True
    return top,mean_val[top],wanted

def make_prediction_ls(test_images,sim):
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


def dealing_layer_data(pyramid_ls, pos_ls, sim):
    if not pyramid_ls:
        return []
    test_data = np.concatenate(pyramid_ls).reshape(-1, 28 * 28)
    test_images_nengo, test_labels_nengo = prepare_data_for_nengo_format(test_data, np.array([4] * len(test_data)))
    prediction_ls, data, wanted_ls = make_prediction_ls(test_images_nengo, sim)
    which_ls, val_ls = zip(*prediction_ls)
    location_ls = []
    #     print(which_ls)
    #     print(wanted_ls)
    #     print(pos_ls)
    for idx, label in enumerate(which_ls):
        if idx >= len(pos_ls):
            break

        # if wanted_ls[idx] and label <= 4:
        #     plt.figure(figsize=(8, 4))
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(test_data[idx].reshape(28, 28))
        #     plt.title(np.std(test_data[idx]))
        #     plt.subplot(1, 2, 2)
        #     plt.plot(tf.nn.softmax(data[out_p_filt][idx]))
        #     plt.legend([str(i) for i in range(5)], loc="upper left")
        if label == 4 and wanted_ls[idx]:
            location_ls.append(pos_ls[idx])
    return location_ls
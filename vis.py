import cv2
import numpy as np
import pickle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def vis_image_pair(image_pair, name):
    #print(image_pair.shape)
    im1 = image_pair[:,:,0:3]
    print(im1.shape)
    im2 = image_pair[:,:,3:]

    #im1 = im1[...,::-1]
    #im2 = im2[...,::-1]
    #print(im2.shape)
    #cv2.imshow(name + 'image 1', im1)
    #cv2.imshow(name + 'image 2', im2)
    #cv2.waitKey(0)
    #cv2.destroyWindow(name)

    plt.imshow(im1)
    plt.show()
    plt.imshow(im2)
    plt.show()

def vis_depth(depth_pred, depth_gt, name):
    pred = depth_pred[0,:,:,0]
    print(pred.shape)

    gt = depth_gt[0,:,:,0]
    print(gt.shape)
    print(gt)

    fig = plt.figure()
    plt.imshow(pred, cmap='gray')
    fig.savefig('training_4/output/depth_pred.png')

    fig = plt.figure()
    plt.imshow(gt, cmap='gray')
    fig.savefig('training_4/output/depth_gt.png')



def vis_flow(flow, shape, name):
    hsv = np.zeros(shape, dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow(name, bgr)
    cv2.waitKey(0)
    #cv2.destroyWindow(name)

def vis_conf(conf, name):
    im1 = np.array(conf[:,:,0]*255, dtype=np.uint8)
    im2 = np.array(conf[:,:,1]*255, dtype=np.uint8)

    cv2.imshow(name + ' x', im1)
    cv2.imshow(name + ' y', im2)
    cv2.waitKey(0)
    #cv2.destroyWindow(name)


def show_train_history_from_pickle(pickle_file_path):
    with open(pickle_file_path, 'rb') as file_pi:
        history = pickle.load(file_pi)
        fig = plt.figure()

        plt.plot(history['loss'])
        # plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        #plt.show()
        fig.savefig('training_4/history.png')


if __name__ == '__main__':
    history = 'training_4/train_history_bootstrap_depth_motion.pickle'

    show_train_history_from_pickle(history)

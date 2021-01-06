import tensorflow as tf
import numpy as np
import cv2
class HDRnet:
    def __init__(self):
        self.tf = tf
        self.session = tf.Session()

    def wightVarible(self,shape):
        initial = self.tf.truncated_normal(shape,stddev=0.1)
        return self.tf.Variable(initial)

    def biasVarible(self,shape):
        initial = self.tf.constant(0.1,shape = shape)
        return self.tf.Variable(initial)

    def conv2d(self,x,w,padding="SAME",stride = [1,1,1,1]):
        return self.tf.nn.conv2d(x,w,strides = stride,padding=padding)

    def inference(self):
        #input
        lowResInput = self.tf.placeholder(dtype=self.tf.float32,shape=[None,256,256,3],name = "lowResInput")
        #low-level feature S
        with self.tf.name_scope("Slayer1"):
            wConv1 = self.wightVarible(shape = [3,3,3,8])
            bConv1 = self.biasVarible([8])
            hConv1 = self.conv2d(lowResInput,wConv1,padding="SAME",stride=[1,2,2,1]) + bConv1
            hConv1 = self.tf.nn.relu(hConv1)
        with self.tf.name_scope("Slayer2"):
            wConv2 = self.wightVarible(shape = [3,3,8,16])
            bConv2 = self.biasVarible([16])
            hConv2 = self.conv2d(hConv1,wConv2,padding="SAME",stride=[1,2,2,1]) + bConv2
            hConv2 = self.tf.nn.relu(hConv2)
        with self.tf.name_scope("Slayer3"):
            wConv3 = self.wightVarible(shape = [3,3,16,32])
            bConv3 = self.biasVarible([32])
            hConv3 = self.conv2d(hConv2,wConv3,padding="SAME",stride=[1,2,2,1]) + bConv3
            hConv3 = self.tf.nn.relu(hConv3)
        with self.tf.name_scope("Slayer4"):
            wConv4 = self.wightVarible(shape = [3,3,32,64])
            bConv4 = self.biasVarible([64])
            hConv4 = self.conv2d(hConv3,wConv4,padding="SAME",stride=[1,2,2,1]) + bConv4
            hConv4 = self.tf.nn.relu(hConv4)




        self.session.run(self.tf.global_variables_initializer())
        image = cv2.imread(r"E:\Py3.6_Proje\myHdrNet\data\straight_lines1_x.png")
        image = cv2.resize(image,dsize=(256,256))
        image = np.reshape(image,newshape=(1,256,256,3))
        res = self.session.run(hConv1, feed_dict={lowResInput: image})
        print("low-res feature layer1 shape :", res.shape)
        res = self.session.run(hConv2, feed_dict={lowResInput: image})
        print("low-res feature layer2 shape :", res.shape)
        res = self.session.run(hConv3, feed_dict={lowResInput: image})
        print("low-res feature layer3 shape :", res.shape)
        res = self.session.run(hConv4,feed_dict={lowResInput:image})
        print("low-res feature layer4 shape :",res.shape)



if __name__ == '__main__':
    HDRnet().inference()
import tensorflow as tf
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

    def inference(self,batchSize=1,isTraining = True):
        #input
        lowResInput = self.tf.placeholder(dtype=self.tf.float32,shape=[None,256,256,3],name = "lowResInput")
        #low-level feature S
        #input 256 * 256 * 3
        with self.tf.name_scope("Slayer1"):
            wConv1 = self.wightVarible(shape = [3,3,3,8])
            bConv1 = self.biasVarible([8])
            hConv1 = self.conv2d(lowResInput,wConv1,padding="SAME",stride=[1,2,2,1]) + bConv1
            hConv1 = self.tf.nn.relu(hConv1)#N * 128 * 128 * 8
            print("low-res feature layer1 shape :", hConv1.get_shape())
        with self.tf.name_scope("Slayer2"):
            wConv2 = self.wightVarible(shape = [3,3,8,16])
            bConv2 = self.biasVarible([16])
            hConv2 = self.conv2d(hConv1,wConv2,padding="SAME",stride=[1,2,2,1]) + bConv2
            hConv2 = tf.contrib.layers.batch_norm(inputs=hConv2, decay=0.9, is_training=isTraining,updates_collections=None)
            hConv2 = self.tf.nn.relu(hConv2)#N * 64 * 64 * 16
            print("low-res feature layer2 shape :", hConv2.get_shape())
        with self.tf.name_scope("Slayer3"):
            wConv3 = self.wightVarible(shape = [3,3,16,32])
            bConv3 = self.biasVarible([32])
            hConv3 = self.conv2d(hConv2,wConv3,padding="SAME",stride=[1,2,2,1]) + bConv3
            hConv3 = tf.contrib.layers.batch_norm(inputs=hConv3, decay=0.9, is_training=isTraining,updates_collections=None)
            hConv3 = self.tf.nn.relu(hConv3)#N * 32 * 32 * 32
            print("low-res feature layer3 shape :", hConv3.get_shape())
        with self.tf.name_scope("Slayer4"):
            wConv4 = self.wightVarible(shape = [3,3,32,64])
            bConv4 = self.biasVarible([64])
            hConv4 = self.conv2d(hConv3,wConv4,padding="SAME",stride=[1,2,2,1]) + bConv4
            hConv4 = tf.contrib.layers.batch_norm(inputs=hConv4, decay=0.9, is_training=isTraining,updates_collections=None)
            Sout = self.tf.nn.relu(hConv4)# N * 16 * 16 * 64
            print("low-res feature layer4 shape :", hConv4.get_shape())
        #local features
        with self.tf.name_scope("local_layer1"):
            wConv1_local = self.wightVarible(shape = [3,3,64,64])
            bConv1_local = self.biasVarible([64])
            hConv1_local = self.conv2d(Sout,wConv1_local,padding="SAME",stride=[1,1,1,1]) + bConv1_local
            hConv1_local = tf.contrib.layers.batch_norm(inputs=hConv1_local, decay=0.9, is_training=isTraining,updates_collections=None)
            hConv1_local = self.tf.nn.relu(hConv1_local)# N * 16 * 16 * 64
            print("local feature layer1 shape :", hConv1_local.get_shape())
        with self.tf.name_scope("local_layer2"):
            wConv2_local = self.wightVarible(shape = [3,3,64,64])
            bConv2_loca1 = self.biasVarible([64])
            hConv2_local = self.conv2d(hConv1_local,wConv2_local,padding="SAME",stride=[1,1,1,1]) + bConv2_loca1
            Lout = self.tf.nn.relu(hConv2_local)# N * 16 * 16 * 64
            print("local feature layer2 shape :", Lout.get_shape())
        #global features
        with self.tf.name_scope("global_layer1_conv"):#N * 16 * 16 * 64
            wConv1_global = self.wightVarible(shape = [3,3,64,64])
            bConv1_global = self.biasVarible([64])
            hConv1_global = self.conv2d(Sout,wConv1_global,padding="SAME",stride=[1,2,2,1]) + bConv1_global
            hConv1_global = tf.contrib.layers.batch_norm(inputs=hConv1_global, decay=0.9, is_training=isTraining,updates_collections=None)
            hConv1_global = self.tf.nn.relu(hConv1_global)# N * 8 * 8 * 64
            print("global feature layer1 conv shape :", hConv1_global.get_shape())
        with self.tf.name_scope("global_layer2_conv"):
            wConv2_global = self.wightVarible(shape = [3,3,64,64])
            bConv2_global = self.biasVarible([64])
            hConv2_global = self.conv2d(hConv1_global,wConv2_global,padding="SAME",stride=[1,2,2,1]) + bConv2_global
            hConv2_global = tf.contrib.layers.batch_norm(inputs=hConv2_global, decay=0.9, is_training=isTraining,updates_collections=None)
            hConv2_global = self.tf.nn.relu(hConv2_global)# N * 4 * 4 * 64
            print("global feature layer2 conv shape :", hConv2_global.get_shape())
        with self.tf.name_scope("global_layer3_fc"):
            hConv2_global = self.tf.reshape(hConv2_global,shape=[-1,4*4*64])
            wConv3_global = self.wightVarible(shape=[4*4*64,256])
            bConv3_global = self.biasVarible(shape=[256])
            hConv3_global = tf.matmul(hConv2_global, wConv3_global) + bConv3_global
            hConv3_global = tf.nn.relu(hConv3_global)#N * 256
            print("global feature layer3 fc shape :", bConv3_global.get_shape())
        with self.tf.name_scope("global_layer4_fc"):
            wConv4_global = self.wightVarible(shape=[256,128])
            bConv4_global = self.biasVarible(shape=[128])
            hConv4_global = tf.matmul(hConv3_global, wConv4_global) + bConv4_global
            hConv4_global = tf.nn.relu(hConv4_global)#N * 128
            print("global feature layer4 fc shape :", bConv4_global.get_shape())
        with self.tf.name_scope("global_layer5_fc"):
            wConv5_global = self.wightVarible(shape=[128,64])
            bConv5_global = self.biasVarible(shape=[64])
            hConv5_global = tf.matmul(hConv4_global, wConv5_global) + bConv5_global
            hConv5_global = tf.nn.relu(hConv5_global)#N * 64
            print("global feature layer5 fc shape :", hConv5_global.get_shape())



        self.session.run(self.tf.global_variables_initializer())
        # image = cv2.imread(r"E:\Py3.6_Proje\myHdrNet\data\straight_lines1_x.png")
        # image = cv2.resize(image,dsize=(256,256))
        # image = np.reshape(image,newshape=(1,256,256,3))
        # res = self.session.run(hConv5_global, feed_dict={lowResInput: image})




if __name__ == '__main__':
    HDRnet().inference()
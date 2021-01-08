import tensorflow as tf
import numpy as np
import math
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
class HDRnet:
    def __init__(self):
        self.tf = tf
        self.session = tf.Session()
        self.GRID_X = 16
        self.GRID_Y = 16
        self.GRID_Z = 7
        self.imageBit = (1 << 8)
    def wightVarible(self,shape):
        initial = self.tf.truncated_normal(shape,stddev=0.1)
        return self.tf.Variable(initial)

    def biasVarible(self,shape):
        initial = self.tf.constant(0.1,shape = shape)
        return self.tf.Variable(initial)

    def clipBit(self,input,min,max):
        if input > max:
            output = max
            # print(input)
        elif input < min:
            output = min
        else:
            output = input
        return output

    def slice(self,Grid,guidMap,batchSize):
        #Grid  N * 16 * 16 * 96
        #guidMap N * w * h
        coeffs = np.zeros(shape=(batchSize,guidMap.shape[1],guidMap.shape[2],12))
        # print("slice gridshape,guidmapshape:",Grid.shape,guidMap.shape)
        Grid = np.reshape(Grid,[batchSize,16,16,8,12])
        #interp gray image to buid gray mask
        rows = guidMap.shape[1]
        cols = guidMap.shape[2]
        idx_width = rows / self.GRID_X
        idy_width = cols / self.GRID_Y
        idx,idy,idz = 0.0,0.0,0.0
        idxLow,idxHigh = 0.0,0.0
        idyLow,idyHigh = 0.0,0.0
        idzLow,idzHigh = 0.0,0.0
        dx,dy,dz = 0.0,0.0,0.0

        for row in range(rows):
            # print(row)
            idx = row / idx_width
            idxLow = math.floor(idx)
            # print(row,idx,idxLow,math.ceil(idx))
            idxHigh = self.clipBit(math.ceil(idx),0,self.GRID_X - 1)
            xd = 0 if idxLow == idxHigh else ((idx - idxLow) / (idxHigh - idxLow))

            sum = 0
            for col in range(cols):
                idy = col / idy_width
                idyLow = math.floor(idy)
                idyHigh = self.clipBit(math.ceil(idy),0,self.GRID_Y - 1)

                yd = 0 if idyLow == idyHigh else ((idy - idyLow) / (idyHigh - idyLow))

                for bs in range(batchSize):
                    idz = guidMap[0][row][col] / (1 / 7)
                    idzLow = math.floor(idz)
                    idzHigh = self.clipBit(math.ceil(idz),0, 6)
                    zd = 0 if idzLow == idzHigh else ((idz - idzLow) / (idzHigh - idzLow))
                    sum = Grid[bs][idxLow][idyLow][idzLow] * (1-xd) * (1-yd) * (1-zd) + \
                          Grid[bs][idxLow][idyHigh][idzLow] * (1-xd) * (yd) * (1-zd) + \
                          Grid[bs][idxHigh][idyLow][idzLow] * (xd) * (1-yd) * (1-zd) + \
                          Grid[bs][idxHigh][idyHigh][idzLow] * (xd) * (yd) * (1-zd) + \
                          Grid[bs][idxLow][idyLow][idzHigh] * (1-xd) * (1-yd) * (zd) + \
                          Grid[bs][idxLow][idyHigh][idzHigh] * (1-xd) * (yd) * (zd) + \
                          Grid[bs][idxHigh][idyLow][idzHigh] * (xd) * (1-yd) *(zd) + \
                          Grid[bs][idxHigh][idyHigh][idzHigh] * (xd) * (yd) * (zd)
                    coeffs[bs][row][col][:] = sum[:]
        return coeffs
    def applyCoeffs(self,fullImage,Coeffs,batchSize):
        """
        :param fullImage: n * 512 * 512 * 3
        :param Coeffs:n * 512 * 512 * 12
        :param batchSize:
        """
        # print("applycoeffs fulliamge shape :",fullImage.shape)
        # print("applycoeffs Coeffs shape :",Coeffs.shape)
        resImage = np.zeros_like(fullImage)
        for bs in range(fullImage.shape[0]):
            for row in range(fullImage.shape[1]):
                for col in range(fullImage.shape[2]):
                    R = fullImage[bs][row][col][0]
                    G = fullImage[bs][row][col][1]
                    B = fullImage[bs][row][col][2]
                    ccm = Coeffs[bs][row][col][:]
                    resR = R * ccm[0] + G * ccm[1] + B * ccm[2] + ccm[3]
                    resG = R * ccm[4] + G * ccm[5] + B * ccm[6] + ccm[7]
                    resB = R * ccm[8] + G * ccm[9] + B * ccm[10] + ccm[11]
                    resImage[bs][row][col][0] = resR
                    resImage[bs][row][col][1] = resG
                    resImage[bs][row][col][2] = resB
        # print("apply coeff resImage shape: ",resImage.shape)
        return resImage



    def conv2d(self,x,w,padding="SAME",stride = [1,1,1,1]):
        return self.tf.nn.conv2d(x,w,strides = stride,padding=padding)

    def inference(self,batchSize=1,isTraining = True,epochs = 1):
        #input
        lowResInput = self.tf.placeholder(dtype=self.tf.float32,shape=[batchSize,256,256,3],name = "lowResInput")
        # fullResInput = self.tf.placeholder(dtype=self.tf.float32,shape=[batchSize,1024,1024,3],name = "lowResInput")
        # guidMap = (fullResInput[:,:,:,0] + fullResInput[:,:,:,1] + fullResInput[:,:,:,2]) / 3
        # guidMap = self.tf.reshape(guidMap,shape=[batchSize,1024,1024])
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
            Gout = tf.nn.relu(hConv5_global)#N * 64
            print("global feature layer5 fc shape :", Gout.get_shape())

        #Fusion global and local feature
        with self.tf.name_scope("fusion_layer"):
            Gout = self.tf.reshape(Gout,shape=[batchSize,1,1,64])
            fusionFeature = tf.nn.relu(Gout + Lout)# N * 16 * 16 * 64
            print("fusion feature layer shape :",fusionFeature.get_shape())
        #Bilteral Grid layer
        with self.tf.name_scope("grid_layer"):
            wConv1_grid = self.wightVarible(shape=[1,1,64,96])
            bConv1_grid = self.biasVarible(shape=[96])
            hConv1_grid = self.conv2d(fusionFeature,wConv1_grid,padding="SAME",stride=[1,1,1,1]) + bConv1_grid
            hconv1_grid = self.tf.contrib.layers.batch_norm(inputs=hConv1_grid, decay=0.9, is_training=isTraining,updates_collections=None)
            Grid = tf.nn.relu(hConv1_grid)#N * 16 * 16 *96
            print("grid feature layer shape :",Grid.get_shape())


        # self.slice(Grid, guidMap, batchSize)
        self.session.run(tf.global_variables_initializer())
        for epo in range(100):
            imageRGB = cv2.imread(r"E:\Py3.6_Proje\myHdrNet\data\straight_lines1_x.png")
            imageRGB = cv2.resize(imageRGB,dsize=(256,256)) / 255.0
            image = cv2.resize(imageRGB,dsize=(256,256))
            image = np.reshape(image,newshape=(1,256,256,3))
            Grid_res = self.session.run(Grid,feed_dict={lowResInput: image})
            guidMap = (imageRGB[ :, :, 0] + imageRGB[ :, :, 1] + imageRGB[ :, :, 2]) / 3
            guidMap = np.reshape(guidMap, (batchSize, 256, 256))
            coeffs = self.slice(Grid_res, guidMap, batchSize)
            # print("coeffs shape :",coeffs.shape)
            imageRGB = np.reshape(imageRGB,(batchSize,imageRGB.shape[0],imageRGB.shape[1],imageRGB.shape[2]))
            preImage = self.applyCoeffs(imageRGB,coeffs,batchSize)
            loss = self.tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.tf.Variable(imageRGB,dtype=self.tf.float32),\
                                                                                            logits=self.tf.Variable(preImage,dtype=self.tf.float32), pos_weight=50))
            optimis = tf.train.AdamOptimizer().minimize(loss)
            lossVal = np.sum(np.abs(imageRGB - preImage))
            print("epoch loss :",epo,"  loss val :",lossVal)






        # self.session.run(self.tf.global_variables_initializer())
        # image = cv2.imread(r"E:\Py3.6_Proje\myHdrNet\data\straight_lines1_x.png")
        # image = cv2.resize(image,dsize=(256,256))
        # image = np.reshape(image,newshape=(1,256,256,3))
        # res = self.session.run(hConv5_global, feed_dict={lowResInput: image})




if __name__ == '__main__':
    HDRnet().inference()
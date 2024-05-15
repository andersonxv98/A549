import numpy as np
from PIL import Image
class Segmentacao:
    def __init__(self, im):
        self.im = im
        self.gImg = np.zeros(self.im.shape, dtype=np.uint8)
    def sobel(self):
        GxMask = [[-1,0,1],[-2,0,2],[-1,0,1]] 
        GyMask = [[-1,-2,-1],[0,0,0],[1,2,1]]

        for i in range(len(self.im)):
            for j in range(len(self.im[i])):
                sumGy = 0
                sumGx = 0
                vec_neigth_img = self.getVizinhanca(i , j, self.im)
                print("Intensidade do pixel central: ", self.im[i][j])
                print("Vizinhnaça: ", vec_neigth_img)
                vec_neigth_Gx = self.getVizinhanca(1 , 1, GxMask)
                vec_neigth_Gy = self.getVizinhanca(1, 1 , GyMask)
                for i in range(len(vec_neigth_img)):
                    if(vec_neigth_img[i] != 0):
                        val_img = vec_neigth_img[i]
                        sumGx += vec_neigth_Gy[i] * val_img
                        sumGy += vec_neigth_Gx[i] * val_img

                sumGx = sumGx if sumGx > 0 else sumGx * -1
                sumGy = sumGy if sumGy > 0 else sumGy * -1

                self.gImg[i][j] = pow((pow(sumGx, 2) + pow(sumGy, 2)), 1/2) #+ self.im[i][j]
        Image.fromarray(self.gImg).show()
            

    def getVizinhanca(self,i , j, matriz):
        topLeft = matriz[i-1][ j -1] if i >0 and j > 0 else 0
        topCenter = matriz[i-1][ j] if  i > 0 else 0
        topRight = matriz[i-1][ j+1] if i >0 and j < len(self.im)-1 else 0
        Left =  matriz[i][j -1]  if j > 0 else 0
        Center = matriz[i][j]
        Right = matriz[i][ j +1] if  j < len(self.im)-1  else 0
        botLeft = matriz[i+1][ j-1] if i < len(self.im)-1 and j > 0 else 0
        botRight= matriz[i+1][j+1] if i < len(self.im)-1 and j < len(self.im)-1 else 0

        vector_pos = [topLeft,topCenter, topRight, Left, Center,Right, botLeft, botRight]
        return vector_pos


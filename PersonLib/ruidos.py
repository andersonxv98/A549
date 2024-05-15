import math
import random

import numpy as np
from PIL import Image
from decimal import Decimal, getcontext
import cv2
from matplotlib import pyplot as plt


class Ruidos:
    def __init__(self, img):
        self.img = img
        gauss_noise = np.zeros(self.img.shape, dtype=np.uint8)
        cv2.randn(gauss_noise, 128, 20)
        self.gauss_noise = (gauss_noise * 0.5).astype(np.uint8)
    def calculohistograma(self):
        hI = []
        print(self.img)
        for i  in range(self.img.max()+1):
            hI.append(0)

        for x in range(len(self.img)):
            for y in range(len(self.img[x])):
                pos = (self.img[x,y])
                hI[pos] += 1

        return hI

    def calculohistogramagaussiano(self):
        hI = []

        for i in range(self.gauss_noise.max() + 1):
            hI.append(0)

        for x in range(len(self.gauss_noise)):
            for y in range(len(self.gauss_noise[x])):
                pos = (self.gauss_noise[x, y])
                hI[pos] += 1

        return hI

    def plotHistograma(self):
        hI = self.calculohistograma()
        print("inwsidde plotHistogram", hI)
        intensidades = [i for i in range(len(hI))]
        plt.plot( intensidades, hI)
        plt.xlabel('intensidade')
        plt.ylabel('quantidade')

        plt.show()

    def saltandpeper(self):
        # Getting the dimensions of the image
        row, col = self.img.shape

        # Randomly pick some pixels in the
        # image for coloring them white
        # Pick a random number between 300 and 10000
        number_of_pixels = random.randint(300, 10000)
        for i in range(number_of_pixels):
            # Pick a random y coordinate
            y_coord = random.randint(0, row - 1)

            # Pick a random x coordinate
            x_coord = random.randint(0, col - 1)

            # Color that pixel to white
            self.img[y_coord][x_coord] = 255

        # Randomly pick some pixels in
        # the image for coloring them black
        # Pick a random number between 300 and 10000
        number_of_pixels = random.randint(300, 10000)
        for i in range(number_of_pixels):
            # Pick a random y coordinate
            y_coord = random.randint(0, row - 1)

            # Pick a random x coordinate
            x_coord = random.randint(0, col - 1)

            # Color that pixel to black
            self.img[y_coord][x_coord] = 0

        return self.img



    def plotGaussian(self):

        gauss_noise = self.calculohistogramagaussiano()
        intensidades = [i for i in range(len(gauss_noise))]
        plt.plot(intensidades, gauss_noise)
        plt.xlabel('intensidade')
        plt.ylabel('quantidade')

        plt.show()
        n_image = self.img + self.gauss_noise
        Image.fromarray(n_image).show()
        return n_image

    def somaDaDiferencaDeCadaPontoModulada(self, n_img):
        soma = 0
        for i in range(len(self.img)):
            for j in range(len(self.img[i])):
                px = int(self.img[i][j])
                pg = int(n_img[i][j])
                p_result = (px - pg)
                aux =  p_result if p_result > 0 else p_result * -1
                soma += aux

        return int(soma)


    def somaDaDiferencaDeCadaPonto(self, n_img):
        soma = 0
        for i in range(len(self.img)):
            for j in range(len(self.img[i])):
                pf = int(self.img[i][j])
                pg =  int(n_img[i][j])
                p_result = pf  - pg
                soma += p_result

        return int(soma)
    def getErroMaximoByPixels(self, n_img):
        max_dif = 0
        for i in range(len(self.img)):
            for j in range(len(self.img[i])):
                val =  int(self.img[i, j]) - int(n_img[i, j])
                aux = val if val > 0 else val * -1
                if aux > max_dif:
                    max_dif = aux
        return max_dif

    def erromaximo(self, n_img):
        maximo_ = self.getErroMaximoByPixels(n_img)
        Max_produto_img =  maximo_ if maximo_  > 0 else maximo_ * -1
        return Max_produto_img

    def errmomedioabsoluto(self, n_img):
        somaDaDif = self.somaDaDiferencaDeCadaPontoModulada(n_img)
        errMdAbs = somaDaDif/ (len(self.img) * len(self.img[0]))
        return errMdAbs

    def erromedioquadratico(self, n_img):
        somaDaDif = (self.somaDaDiferencaDeCadaPonto(n_img))

        somaquadrad = Decimal(pow(int(somaDaDif), 2))

        errMdQuad = somaquadrad / (len(self.img) * len(self.img[0]))
        return errMdQuad

    def raizdoerromedioquadratico(self, n_img):
        part1 = self.erromedioquadratico(n_img)
        raizErrMdQuadratico =  part1 ** 1/2
        return raizErrMdQuadratico

    def somaElementosDeFx(self):
        soma = 0
        for i in range(len(self.img)):
            for j in range(len(self.img[i])):
                soma += self.img[i][j]
        return int(soma)
    def erromedioquadraticonormalizado(self, n_img):
        somaDaDif = self.somaDaDiferencaDeCadaPonto(n_img)
        somaquadrad = Decimal(pow(int(somaDaDif), 2))
        somaFxyquadrado = self.somaElementosDeFx() ** 2
        erroMdQuadratico =  somaquadrad / somaFxyquadrado
        return erroMdQuadratico


    def coeficientedeJaccard(self, n_img):
        soma = 0
        for i in range(len(self.img)):
            for j in range(len(self.img[i])):
                if (self.img[i][j] == n_img[i][j]):
                    soma +=1


        coeficiente = soma  / (len(self.img)  * len(self.img[0]))

        return coeficiente

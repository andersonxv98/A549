import cv2

import matplotlib.pyplot as plt
from PIL import Image
from skimage import filters as Filters
import numpy as np
import statistics
import os

class Realce:
    def __init__(self, im, name):
        self.im = im
        self.name = name

    def show(self):
        Image.fromarray(self.im).show()
    def calculohistograma(self):
        hI = []
        print(self.im)
        for i  in range(self.im.max()+1):
            hI.append(0)

        for x in range(len(self.im)):
            for y in range(len(self.im[x])):
                pos = (self.im[x,y])
                if pos != 0:
                    hI[pos] += 1



        return hI

    def getMedia(self):
        soma = 0
        qtd= 0
        for x in range(len(self.im)):
            for y in range(len(self.im[x])):
                pos = (self.im[x,y])
                if pos != 0:
                   soma = self.im[x][y]
                   qtd +=1 
        return (soma/qtd)
   

    def plotHistograma(self):
        hI = self.calculohistograma()
        #intensidades = [i for i in range(len(hI))]
        '''plt.plot( intensidades, hI)
        plt.xlabel('intensidade')
        plt.ylabel('quantidade')
        plt.title(self.name)
        plt.show()
        iten = max(intensidades[1:])
        '''
        #media = sum(intensidades)/ len(intensidades)
        mediana = statistics.median(hI)
        #moda = statistics.mode(intensidades)
        #print("media histograma", media)
        variancia = statistics.variance(hI)

        # Calculando o desvio padrão
        desvio_padrao = statistics.stdev(hI)
        media = sum(hI)/ len(hI)
        print("SOMA: ", sum(hI))
        print("A variância dos números é:", variancia)
        print("O desvio padrão dos números é:", desvio_padrao)
        print("mediana histograma", mediana)
        print("media histograma", media)
        print("pico histograma", hI.index(max(hI)))
        return 

    def equalizacaoPf(self):
        histogramaOriginal = self.calculohistograma()
        pfAcumulada= []
        lastPAcumulado = 0
        for k in histogramaOriginal: #para cade nivel de cinza no histograma
            p= k/sum(histogramaOriginal)
            lastPAcumulado += p
            pfAcumulada.append(round(lastPAcumulado * self.im.max()))

        return pfAcumulada

    def plotHistogramaEqualizado(self):
        pfAcumuladaEqual = self.equalizacaoPf()
        print("inwsidde plotHistogram", pfAcumuladaEqual)
        intensidades = [i for i in range(len(pfAcumuladaEqual))]
        plt.plot(intensidades, pfAcumuladaEqual)
        plt.xlabel('intensidade')
        plt.ylabel('quantidade')
        plt.title(self.name+'|equalizado')
        plt.show()
        
        return
    
    def equalizeImg(self):
        histoEq = self.equalizacaoPf()
        for i in range(len(self.im)):
            for j in range(len(self.im[i])):
                self.im[i][j] = histoEq[self.im[i][j]]
        
        im = Image.fromarray(self.im)
        im.show()
        return
        if not os.path.exists("./imgs_equalizada/"):
            os.makedirs("./imgs_equalizada/")
            print("Pasta criada com sucesso:")
            im.save("./imgs_equalizada/"+self.name)
        else:
            im.save("./imgs_equalizada/"+self.name)
       

    def correcaoGamma(self, gamma, c):
        c = c
        compensacao= 1


        gx = pow((c * (self.im+ compensacao)), gamma)
        Image.fromarray(gx).show()

    def negativo(self):
        max_intensit  = self.im.max()
        for i in range(len(self.im)):
            for j in range(len(self.im[i])):
                self.im[i][j] = max_intensit - self.im[i][j]


        Image.fromarray(self.im).show()


    def getMaxIntensidadeFromHisto(self):
        intensidades  = self.calculohistograma()
        indice_maximo = intensidades.index(max(intensidades[1:]))
        return indice_maximo

    def binariazacao(self,intensidade_):
        max_intensit = self.im.max()
        #intensidade_limite = self.getMaxIntensidadeFromHisto()
        intensidade_limite = intensidade_
        for i in range(len(self.im)):
            for j in range(len(self.im[i])):
                if(self.im[i][j] < intensidade_limite):
                    self.im[i][j] = 0
                else:
                    self.im[i][j] = max_intensit
        Image.fromarray(self.im).show()
        #return
        im = Image.fromarray(self.im)
        intensidade_str = str(intensidade_)
        if not os.path.exists("./binarizadas_pico/"+intensidade_str):
            os.makedirs("./binarizadas_pico/"+intensidade_str)
            print("Pasta criada com sucesso:")
            im.save("./binarizadas_pico/"+intensidade_str+"/"+self.name)
        else:
            im.save("./binarizadas_pico/"+intensidade_str+"/"+self.name)
        return self.im
    def espacamento(self):
        max_intensit = self.im.max()
        intensidade_limite = self.getMaxIntensidadeFromHisto()
        for i in range(len(self.im)):
            for j in range(len(self.im[i])):
                val =self.im[i][j]
                if(val < 128):
                    self.im[i][j] = 0
                elif(self.im[i][j] < intensidade_limite):
                    self.im[i][j] = 128
                else:
                    self.im[i][j] = max_intensit

        im = Image.fromarray(self.im)
        im.save("./images_espacamento/"+self.name)

    def limiarizacaoOtsu(self):
        img_array = np.array(self.im)
        limiar_otsu = Filters.threshold_otsu(img_array)
        
        img_bin = img_array > limiar_otsu
        img_otsu = (img_bin * 255).astype(np.uint8)
        im = Image.fromarray(img_otsu)
        print("Valor trashold: ", limiar_otsu)
        #im.save("./imgs_equalizada_otsu/"+self.name)
        im.show()
        return img_otsu
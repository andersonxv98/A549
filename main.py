import os
import csv
import numpy as np

from PersonLib import  *
import cv2
from PIL import Image
from PersonLib.fundo import Fundo
from PersonLib.realce import Realce
from PersonLib.segmentacao import Segmentacao
from PersonLib.filtros import  Filtros
from PersonLib.graphclusters import GaphClusters
from exercicio6e10 import HoshenKopelmanAlgoritm, CalculoDeDistancia
from skimage import filters

def main():
    '''exercC = imgMaker.criarImagemExercicioB(255 / 2, 255, 2, 1)
    arr_matriz_rotacao = np.rot90(np.rot90(exercC))
    arr_matriz_rotacao = imgMaker.criarImagemExercicioC(arr_matriz_rotacao, 6, 9)
    C = Image.fromarray(arr_matriz_rotacao)
    C.show()
    HKalgoritmD = HoshenKopelmanAlgoritm(exercC)
    CalculoDeDistancia(exercC, HKalgoritmD.largest_label)
    '''
    #Primeira abordagem: caso a iluminação das imagens estejam corretas;
    #nome_arquivo = 'clusters_eq_otsu.csv'
    #with open(nome_arquivo, mode='w', newline='') as arquivo_csv:
    #    pass
    #nome_arquivo = 'clusters_eq_otsu.csv'
    grC = GaphClusters()
    grC.graficoClusterBinarizado()
    print("END")
    return

    for imagem in os.listdir("./imgs_equalizada/"):
        #param = 10
        #intensidades = [120, 178, 176, 160, 152, 168, 160]
        #for intensidade in intensidades:
        img = cv2.imread('./imgs_equalizada/' + imagem, cv2.IMREAD_GRAYSCALE)
        #contorno = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        #print(imagem,len(contorno)-1)
        realce = Realce(img, imagem)
        print("IMAGEM: ", imagem)
        realce.limiarizacaoOtsu()
        #img = cv2.resize(img, (256, 256)
        #return
        #im = Image.fromarray(img)
        #im.show(imagem)
        #HKalgoritmD = HoshenKopelmanAlgoritm(img)
        #CalculoDeDistancia(img, HKalgoritmD.largest_label)

        #image = cv2.imread('./Normal.png', cv2.IMREAD_GRAYSCALE)
        
            #img_binarizada = Realce(img, imagem).binariazacao(200)
        #val = param * intensidade
        #val = 150
        #realce = Realce(img, str(val)+imagem)
        #realce.plotHistograma()
        #result_bin = realce.binariazacao(val)
        #segment = Segmentacao(img)
        
        #img_binarizada = realce.binariazacao(val)
        #img_sobel =  segment.sobel()
        '''vorda_horizontal =  cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        borda_vertical = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        magnitude_gradiente = np.uint8(np.sqrt(vorda_horizontal**2 + borda_vertical**2))
        cv2.imshow("Celula ", magnitude_gradiente)
        cv2.waitKey(0)
        '''
        #img_limiarizada = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
        #cv2.imshow('Imagem Limiarizada', img_limiarizada)
        #cv2.waitKey(0)
        #realce = Realce(img, imagem)
        #realce.plotHistograma()
        #realce.plotHistograma()
        #print("MEDIA: ", realce.getMedia())
        #realce.plotHistograma()
        #print("Mediana", )_
        #realce.equalizeImg()
        #img_binarizada = realce.binariazacao(intensidade)
        #Image.fromarray(img_sobel).show()
        #return
        #realce.espacamento()
        #realce.plotHistogramaEqualizado()
        #img_equalizada = realce.equalizeImg()
        #img_filtrada = Filtros(realce.im).filtroMedia(3)
        


        '''img_otsu = realce.limiarizacaoOtsu()
        val_kernel= [3, 9, 21, 63]
        for val in val_kernel:
            kernel = np.ones((val,val), np.uint8)
            imagem_abertura = cv2.morphologyEx(img_otsu, cv2.MORPH_OPEN, kernel)
            #cv2.imwrite('./img_equalizada_otsu_abertura/'+str(val)+'kernel'+imagem+'.png', imagem_abertura)
            #cv2.imshow(imagem_abertura)
            #cv2.waitKey(7000)
            #realce.binariazacao(240)
            HKalgoritmD = HoshenKopelmanAlgoritm(imagem_abertura, val, imagem)
            #CalculoDeDistancia(img, HKalgoritmD.largest_label)
        '''

main()
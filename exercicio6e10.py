import cv2

#import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import os
import csv

class CenterCluster:
    def __init__(self, x, y):
        self.x = x
        self.y = y


    def getI(self):
        return self.x

    def getJ(self):
        return self.y


class CalculoDeDistancia:
    def __init__(self, img, largest_lb):
        self.img = img
        self.largest_lb = largest_lb
        self.doTheThing()

    def getCenterPixel(self, cluster_value, qtd_cluster):
        center_position = (0,0)
        inicial_position = (0, 0)
        qtd_cluster = math.log2(qtd_cluster) * 2
        print("QTD DIV: ", qtd_cluster)
        flag_ini = True
        qtd_x = len(self.img)
        qtd_y = len(self.img[0])
        for i in range(qtd_x):
            for j in range(qtd_y):
                cluster_value_img = self.img[i][j]
                if cluster_value_img == cluster_value:
                    ini_X = inicial_position[0]
                    ini_Y = inicial_position[1]
                    center_position = (ini_X+(qtd_x/qtd_cluster),(ini_Y+(qtd_y/qtd_cluster)))
                    if(flag_ini):
                        inicial_position = (i, j)
                        flag_ini   = False

        return center_position

    def getClustersAleatorio(self):
        cluster1, cluster2 = (random.randint(1, self.largest_lb), random.randint(1, self.largest_lb))
        #cluster1, cluster2 = (1,2)
        #cluster1, cluster2 = (1,3)
        #cluster1, cluster2 = (4,1)
        print("CENTRO DE: ---------------------------------------")
        print("CHOICE 1: ", cluster1  ,", : Choice2: ", cluster2)
        center_cluster1 = self.getCenterPixel(cluster1, self.largest_lb)
        center_cluster2 = self.getCenterPixel(cluster2, self.largest_lb)

        return center_cluster1, center_cluster2


    def calDistanciaEuclidiana(self, f1, f2):
        d1 = pow((f1.getI() - f2.getI()), 2)
        d2 = pow((f1.getJ() - f2.getJ()), 2)
        re = math.sqrt(d1+d2)
        return re

    def calDistanciaD4CityBlock(self, f1, f2):
        d1 = (f1.getI() - f2.getI()) if (f1.getI() > f2.getI()) else ((f1.getI() - f2.getI())* -1)
        d2 = f1.getJ() - f2.getJ() if f1.getJ() > f2.getJ() else ((f1.getJ() - f2.getJ())* -1)
        re = d1 + d2
        return re

    def calDistanciaChessBoard(self, f1, f2):
        d1 = (f1.getI() - f2.getI()) if (f1.getI() > f2.getI()) else ((f1.getI() - f2.getI())* -1)
        d2 = f1.getJ() - f2.getJ() if f1.getJ() > f2.getJ() else ((f1.getJ() - f2.getJ())* -1)
        re = max(d1, d2)
        return re

    def doTheThing(self):
        cl1, cl2 = self.getClustersAleatorio()
        cl1 = CenterCluster(cl1[0], cl1[1])
        cl2 = CenterCluster(cl2[0], cl2[1])

        De = self.calDistanciaEuclidiana(cl1, cl2)
        D4 = self.calDistanciaD4CityBlock(cl1, cl2)
        D8 = self.calDistanciaChessBoard(cl1, cl2)

        print("VALORES DE DISTANCIA f(i,j):-------------")
        print("DISTANCIA EUCLIDIANA: CL1 (",cl1.getI(),",", cl1.getJ(),") : CL2 (",
              cl2.getI(),",", cl2.getJ(),") :", De)
        print("DISTANCIA CITYBLOCK: CL1 (",cl1.getI(),",", cl1.getJ(),") : CL2 (",
              cl2.getI(),",", cl2.getJ(),") :", D4)
        print("DISTANCIA CHESSBOARD: CL1 (", cl1.getI(),",", cl1.getJ(),") : CL2 (",
              cl2.getI(),",", cl2.getJ(),") :", D8)


class Rotulo:
    def __init__(self, classification, value):
        self.classification = classification
        self.value = value

    def get_value(self):
        return self.value

    def getClassificacao(self):
        return  self.classification

class HoshenKopelmanAlgoritm:
    def __init__(self, image, intensidade_corte, name):
        print("Construtor Hoshen")
        self.ultimo_processado = None
        self.intensidade_corte = intensidade_corte
        self.largest_label = 0
        self.labels = [i for i in range(image.size)]
        self.classificacoes = np.full((len(image), len(image[0])), Rotulo(None, None))
        self.image = image
        self.nome = name
        self.algorithm()

    def escreverDadosCsv(self,dados):
        
        # Nome do arquivo CSV que queremos editar
        nome_arquivo = 'clusters_eq_otsu.csv'

        # Abrindo o arquivo CSV em modo de leitura e escrita
        with open(nome_arquivo, mode='r+', newline='') as arquivo_csv:
            leitor_csv = csv.reader(arquivo_csv)
            linhas = list(leitor_csv)  # Lendo todas as linhas do arquivo

            # Modificando os dados conforme necessário
            linhas.append(dados)

            # Movendo o cursor para o início do arquivo
            arquivo_csv.seek(0)

            # Escrevendo as linhas modificadas de volta ao arquivo
            escritor_csv = csv.writer(arquivo_csv)
            for linha in linhas:
                escritor_csv.writerow(linha)

            # Truncando o restante do conteúdo do arquivo, caso o novo conteúdo seja menor que o anterior
            arquivo_csv.truncate()

        print("Arquivo CSV editado com sucesso!")
        return

    def algorithmLogic(self, i, j):

        if (self.image[i][j] != 0):

            left = self.image[i, j -1] if j > 0 else 0
            above = self.image[i-1,  j] if i > 0 else 0
            topleft = self.image[i-1, j -1] if (i> 0  and j >0) else 0

            if left == 0 and above == 0:  # Neither a label above nor to the left
                self.largest_label += 1  # Make a new, as-yet-unused cluster label

                pRotulo = Rotulo(  self.largest_label,  self.image[i, j])
                self.classificacoes[i, j] = pRotulo
                #self.ultimo_processado = self.image[i, j]

                self.image[i, j] = self.largest_label

            elif left == 0 and above == 0 and topleft != 0:
                class_inAValue = self.classificacoes[i-1, j - 1].get_value()
                imgValue = self.image[i, j]
                if (class_inAValue != imgValue):
                    self.largest_label += 1
                    pRotulo = Rotulo(self.largest_label, self.image[i, j])
                    self.classificacoes[i, j] = pRotulo
                    self.image[i, j] = self.largest_label
                else:
                    pRotulo = Rotulo(class_inAValue, self.image[i, j])
                    self.classificacoes[i, j] = pRotulo
                    self.image[i, j] = self.find(class_inAValue)


            elif left != 0 and above == 0:  # One neighbor, to the left
                #aux  = self.image[i , j]
                class_inDValue = self.classificacoes[i, j -1].get_value()
                imgValue = self.image[i,j]
                if(class_inDValue != imgValue):
                    self.largest_label += 1
                    pRotulo = Rotulo( self.largest_label, self.image[i, j])
                    self.classificacoes[i, j] = pRotulo
                    self.image[i, j] = self.largest_label
                else:
                    pRotulo = Rotulo(self.largest_label, self.image[i, j])
                    self.classificacoes[i, j] = pRotulo
                    self.image[i, j] = self.find(self.classificacoes[i, j-1].getClassificacao())





            elif left == 0 and above != 0:  # One neighbor, above
                val_classsValue = self.classificacoes[i-1, j].get_value()
                valuer= self.image[i, j]
                if(val_classsValue != valuer):
                    #print("ENTROU CONDIÇÂO")
                    self.largest_label += 1
                    pRotulo = Rotulo( self.largest_label, self.image[i, j])
                    self.classificacoes[i, j] = pRotulo
                    self.image[i, j] = self.largest_label
                else:
                    pRotulo = Rotulo(above, self.image[i, j])
                    self.classificacoes[i, j] = pRotulo
                    self.image[i, j] = self.find(self.classificacoes[i-1, j].getClassificacao())



            else:  # Neighbors BOTH to the left and above
                value =  self.image[i, j]
                classificLeft = self.classificacoes[i, j - 1].getClassificacao()
                classificAbove = self.classificacoes[i - 1, j].getClassificacao()
                valAbove = self.classificacoes[i-1, j].get_value()
                valLeft = self.classificacoes[i, j-1].get_value()

                if (valLeft != value and value !=  valAbove):
                    #print("DIFERENTE:: ", "esquerda:  ", self.image[i-1, j], "Cima: ", self.image[i,j-1] , "atual: ",self.image[i, j])
                    self.largest_label += 1
                    pRotulo = Rotulo(self.largest_label, self.image[i, j])
                    self.classificacoes[i,j] = pRotulo
                    self.image[i, j] = self.find(self.largest_label)

                else:


                    if (valLeft != (value)):
                        if(value == valAbove):
                            pRotulo = Rotulo(above, self.image[i, j])
                            self.classificacoes[i, j] = pRotulo
                            self.image[i, j] = self.find(above)

                            #self.union(aux_left, aux_above)  # Link the left and above clusters
                    else:
                        pRotulo = Rotulo(left, self.image[i, j])
                        self.classificacoes[i, j] = pRotulo
                        self.image[i, j] = self.find(left)

    def algorithm(self):
        for i in range(len(self.image)):
            for j in range(len(self.image[0])):
                self.algorithmLogic(i, j)


        print(self.image)
        print("QUantidade de Label (cluster): ", self.largest_label)
        dados = [self.nome, self.intensidade_corte, self.largest_label]
        self.escreverDadosCsv(dados)
        c =Image.fromarray(self.image)

        '''r,g, b =c.convert("RGB").split()
        g = Image.fromarray(np.full(c.size,0,dtype=np.uint8))
        b = Image.fromarray(np.full(c.size,50,dtype=np.uint8))

        img_color = Image.merge('RGB', (r,g, b))
        img_color.show()'''
        c.show()
        if not os.path.exists("./hoshenbinarizada/"+str(self.intensidade_corte)):
            os.makedirs("./hoshenbinarizada/"+str(self.intensidade_corte))
            print("Pasta criada com sucesso:")
            c.save("./hoshenbinarizada/"+str(self.intensidade_corte)+"/"+self.nome)
        else:
            c.save("./hoshenbinarizada/"+str(self.intensidade_corte)+"/"+self.nome)
        #print(c)

        return self.image
        #wc.show()

        #print("LABELS: ", self.labels)

    def union(self, x, y):
        self.labels[self.find(x)] = self.find(y)

    def find(self, x):
        y = x
        while (self.labels[y] != y):
            y = self.labels[y]

        while(self.labels[x] != x):
            z = self.labels[x]
            self.labels[x] = y
            x = z
        return y
class ImagemMaker:
    def __init__(self, size):
        self.mode = 'RGB'
        self.size = (size, size)

    def getProfundidade(self, numero_d_niveisDCinza):
        taxa_profundidade = math.log2(numero_d_niveisDCinza)
        return taxa_profundidade
    def getAmostra(self, size_w, size_h):
        return (size_w * size_h)

    def getSliceOFImage(self, imagem, i, j,size_i, size_j, val):
        aux_j = j
        while( i < size_i):
            j= aux_j
            while( j < size_j):
                imagem[i][j] = val
               # print("i: ", i, "j: ", j)
                j += 1
            #print("saiu do while J")
            i += 1
        #print("SAIU DE TODOS OS LOOPS")
        return imagem

    def criarImagemExercicioC(self, imagem, range_x, range_y):
        n_cinzas = (imagem[0][0],imagem[255][255])
        img= imagem
        val_min = int(n_cinzas[0])
        inicio_x = 0
        final_size_x = int(self.size[0] / range_x)
        final_size_y = int(self.size[1] / range_y)
        amostra = self.getAmostra(final_size_x, final_size_y)
        vezes_de_entrada = 0
        for i in range(range_x):
            inicio_y = 0
            for j in range(range_y):
                if (i > 0 and i < range_x-1 and j > 0 and j < range_y-1):
                    if(j != int(range_y / 2)):
                        if(i > range_x/2):
                            val_min = n_cinzas[0]
                            vezes_de_entrada +=1
                            img = self.getSliceOFImage(img, inicio_x, inicio_y, final_size_x* (i+1), final_size_y* (j+1), int(val_min))
                            print("Posição inicial:", [inicio_x, inicio_y], "posição Finall",
                                  [final_size_x * (i + 1), final_size_y * (j + 1)], "Amostragem: ", amostra,
                                  "Nivel de profundidade: ",
                                  val_min)
                        else:
                            if(i < (range_x/2)-1):
                                val_min = n_cinzas[1]
                                vezes_de_entrada += 1
                                img = self.getSliceOFImage(img, inicio_x, inicio_y, final_size_x * (i + 1),
                                                           final_size_y * (j + 1), int(val_min))
                                print("Posição inicial:", [inicio_x, inicio_y], "posição Finall",
                                      [final_size_x*(i+1), final_size_y* (j+1)], "Amostragem: ", amostra, "Nivel de profundidade: ",
                                      val_min)
                inicio_y = int((final_size_y) * (j + 1))
                val_min = val_min
            inicio_x = int((final_size_x) * (i + 1))

        print("VEZES QUE SATISFEZ A ENTRADA: ", vezes_de_entrada)
        return img
                #imagem[i][j] =
    def criarImagemExercicioB(self, val_min, val_max, qtd_divisoes_x, qtd_divisoes_y):
        val_min = int(val_min)
        val_max = int(val_max)
        img =  np.full(self.size,0,dtype=np.uint8)

        val_increment = (val_max- val_min) / (qtd_divisoes_x * qtd_divisoes_y)
        inicio_x = 0
        inicio_y = 0
        final_size_x= self.size[0]/ qtd_divisoes_x
        final_size_y = self.size[1] / qtd_divisoes_y

        amostra = self.getAmostra(final_size_x, final_size_y)
        n_nivel_de_cinza = 0
        for i in range(qtd_divisoes_x):
            inicio_y = 0

            for j in range(qtd_divisoes_y):
          #      print("Nivel de Cinza: ", val_min)
                img = self.getSliceOFImage(img, inicio_x, inicio_y, final_size_x* (i+1), final_size_y* (j+1), int(val_min))
         #       print(img)
                #inicio_x = 0

                print("Amostragem: ", amostra, "Nivel de Profundidade: ", val_min)
                inicio_y = int((final_size_y)* (j+1))
                val_min = val_min+ (val_increment)
                n_nivel_de_cinza += 1
            inicio_x = int((final_size_x)* (i+1))
        #print(img)
        print("Taxa de Profundidade: ", self.getProfundidade(n_nivel_de_cinza))
        return img

    def criarImagemUnicoValorExercicioA(self, val):
        val = int(val)
        img = np.full(self.size,val,dtype=np.uint8)
        taxa_prof = self.getProfundidade(1)
        amostra = self.getAmostra(self.size[0], self.size[1])
        nvl_prof = val
        print("Amostragem: ", amostra, "Nivel de Profundidade: ", nvl_prof, "taxa de profundidade: ", taxa_prof)
        Img =Image.fromarray(img)
        Img.show()
        return img

##img_a = Image.new(mode='RGB', size=(256,256), color= (int(val_),int(val_),int(val_)))


#imgMaker = ImagemMaker(256)

"""
print('SIZE: ',imgMaker.size[0])
print("Imagem Exercicio A):")
ImgclustersA = imgMaker.criarImagemUnicoValorExercicioA(255/2)
HKalgoritmA = HoshenKopelmanAlgoritm(ImgclustersA)
"""

"""print("Imagem Ezercicio B):")
min_B = 255/2
exercB = imgMaker.criarImagemExercicioB(min_B, 255, 2, 1)
arr_matriz_rotacao = np.rot90(np.rot90(exercB))
B = Image.fromarray(arr_matriz_rotacao)
B.show()

HKalgoritmB = HoshenKopelmanAlgoritm(arr_matriz_rotacao)


print("Imagem Ezercicio C):")
exercC = imgMaker.criarImagemExercicioB(255/2, 255, 2, 1)
arr_matriz_rotacao = np.rot90(np.rot90(exercC))
arr_matriz_rotacao = imgMaker.criarImagemExercicioC(arr_matriz_rotacao, 6, 9)
C = Image.fromarray(arr_matriz_rotacao)
C.show()

HKalgoritmC =HoshenKopelmanAlgoritm(arr_matriz_rotacao)
CalculoDeDistancia(exercC, HKalgoritmC.largest_label)
"""



print("Imagem Exercicio D):")
#exercicioD = imgMaker.criarImagemExercicioB(255/2, 255, 2, 2)

'''image = cv2.imread('./Normal.png', cv2.IMREAD_GRAYSCALE)
image =  cv2.resize(image, (256, 256))

D = Image.fromarray(image)
D.show()
HKalgoritmD = HoshenKopelmanAlgoritm(image)
CalculoDeDistancia(image, HKalgoritmD.largest_label)
'''
"""
print("Imagem Exercicio e):")
exercicioE = imgMaker.criarImagemExercicioB(255/2, 255, 4, 4)
E = Image.fromarray(exercicioE)
E.show()
hKalgoritmE = HoshenKopelmanAlgoritm(exercicioE)
print("LARGEST LABEL: ", hKalgoritmE.largest_label)
CalculoDeDistancia(exercicioE, hKalgoritmE.largest_label)


"""
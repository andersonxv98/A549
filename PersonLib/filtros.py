import numpy as np
from PIL import Image, ImageFilter
import math
import statistics
class Filtros:
    def __init__(self, img):
        self.img = img
        self.size = (len(self.img) , len(self.img))
    def criarFiltroGaussian(self, tamanho, ro):
        filtro = []
        for i in range(tamanho):
            col = []
            for j in range(tamanho):
                col.append(self.binomioNewton(tamanho,i,j))
            filtro.append(col)

        return filtro

    def filtroGaussiano(self, n):

        fat = self.generate_pascal_triangle(n)

        fat = (fat[n - 1])
        n_sum = sum(fat)
        filter = []
        for ile in fat:
            row = []
            for ele in fat:
                row.append(int(ile)*int(ele) /  (n_sum * n_sum))
            filter.append(row)
        print("FILTRO: ", filter)
        print("DESVIO PADRÂO: ", self.desvioPadrao(n))

        return filter

    def fatorial(self, n):
        if n == 1:
            return 1
        else:
            return n * self.fatorial(n - 1)


    def desvioPadrao(self, n):
        result = math.sqrt((n-1)/ 2)
        return result
    def generate_pascal_triangle(self, num_rows):
        triangle = []
        for i in range(num_rows):
            row = [1]  # O primeiro elemento de cada linha é sempre 1
            if i > 0:
                # Calcula os coeficientes binomiais para o restante da linha
                for j in range(1, i):
                    row.append(triangle[i - 1][j - 1] + triangle[i - 1][j])
                row.append(1)  # O último elemento de cada linha também é sempre 1
            triangle.append(row)
        return triangle

    def print_pascal_triangle(self, triangle):
        max_width = len(' '.join(map(str, triangle[-1])))
        for row in triangle:
            print(' '.join(map(str, row)).center(max_width))

    def filtroGaussianoIMG(self, dimensao):
        filtro = (self.filtroGaussiano(dimensao))
        img_filtrada  = self.aplicarFiltro(filtro  )
        im = Image.fromarray(img_filtrada)
        im.show()
        return

    def filtroDaMediana(self):
        img_filtrada = np.full(self.size, 0, dtype=np.uint8)
        for i in range(len(self.img)):
            for j in range(len(self.img[i])):
                mediana =  self.getMedianaByVizinhanca(i, j)
                img_filtrada[i, j] =  mediana


        im = Image.fromarray(img_filtrada)

        im.show()
        return

    def filtroModa(self):
        img_filtrada = np.full(self.size, 0, dtype=np.uint8)
        for i in range(len(self.img)):
            for j in range(len(self.img[i])):
                mediana = self.getModaByVizinhanca(i, j)
                img_filtrada[i, j] = mediana

        im = Image.fromarray(img_filtrada)

        im.show()

        return

    def getMedianaByVizinhanca(self, i , j ):
        vector = []
        TopLeft = self.img[i - 1, j - 1] if i > 0 and j > 0 else 0
        TopCenter = self.img[i - 1, j] if i > 0 else 0
        TopRight = self.img[i - 1, j + 1] if i > 0 and j < len(self.img) - 1 else 0
        CenterLeft = self.img[i, j - 1] if j > 0 else 0
        CenterRight = self.img[i, j + 1] if j < len(self.img) - 1 else 0
        BottomLeft = self.img[i + 1, j - 1] if i < len(self.img) - 1 and j > 0 else 0
        BottomCenter = self.img[i + 1, j] if i < len(self.img) - 1 else 0
        BottomRight = self.img[i + 1, j + 1] if i < len(self.img) - 1 and j < len(self.img) - 1 else 0

        vector.append(TopCenter)
        vector.append(BottomCenter)
        vector.append(TopRight)
        vector.append(CenterRight)
        vector.append(CenterLeft)
        vector.append(BottomLeft)
        vector.append(BottomRight)
        vector.append(TopLeft)
        vector.append(self.img[i, j])
        sorted_vector = sorted(vector)
        #print("VECTOR: ", sorted_vector)
        return statistics.median(sorted_vector)
    def aplicarFiltro(self, filtro ):
        img_filtrada = np.full(self.size, 0, dtype=np.uint8)
        for i in range(len(self.img)):
            for j in range(len(self.img[i])):
                TopLeft = self.img[i - 1, j - 1] if i > 0 and j > 0 else 0
                TopCenter = self.img[i - 1, j] if i > 0 else 0
                TopRight = self.img[i - 1, j + 1] if i > 0 and j < len(self.img) - 1 else 0
                CenterLeft = self.img[i, j - 1] if j > 0 else 0
                CenterRight = self.img[i, j + 1] if j < len(self.img) - 1 else 0
                BottomLeft = self.img[i + 1, j - 1] if i < len(self.img) - 1 and j > 0 else 0
                BottomCenter = self.img[i + 1, j] if i < len(self.img) - 1 else 0
                BottomRight = self.img[i + 1, j + 1] if i < len(self.img) - 1 and j < len(self.img) - 1 else 0

                slice_img = [[TopLeft, TopCenter, TopRight],
                             [CenterLeft, self.img[i, j], CenterRight],
                             [BottomLeft, BottomCenter, BottomRight]]
                rm1 = 0

                for row in range(len(filtro)):
                    for col in range(len(filtro[0])):
                        rm1 += slice_img[row][col] * filtro[row][col]


                img_filtrada[i, j] = (rm1)

        return img_filtrada
    def divisorMacasra(self, mask, n):

        for row in range(len(mask)):

            for col in range(len(mask[0])):
                mask[row][col] = mask[row][col] / n

        return mask
    def filtroPassaAlta3p3(self):
        m1 = [[0,-1,0],[-1,4,-1],[0,-1,0]] #vizinhança 4
        m2 = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]] #vizinhança8
        m1 = self.divisorMacasra(m1,9)
        m2 = self.divisorMacasra(m2, 9)



        n_m1 = self.aplicarFiltro(m1)
        n_m2 = self.aplicarFiltro(m2)


        #print(n_m1)
        #print(n_m2)
        image_1 = Image.fromarray(n_m1)
        image_2 = Image.fromarray(n_m2)

        image_1.show()
        image_2.show()

        return image_1, image_2

    def filtroMedia(self, n):
        filtro = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(1 * 1/(n*n))
            filtro.append(row)

        img_filtrada  = self.aplicarFiltro(filtro )

        image_1 = Image.fromarray(img_filtrada)


        image_1.show()

        return img_filtrada

    def getModaByVizinhanca(self, i , j):
        vector = []
        TopLeft = self.img[i - 1, j - 1] if i > 0 and j > 0 else 0
        TopCenter = self.img[i - 1, j] if i > 0 else 0
        TopRight = self.img[i - 1, j + 1] if i > 0 and j < len(self.img) - 1 else 0
        CenterLeft = self.img[i, j - 1] if j > 0 else 0
        CenterRight = self.img[i, j + 1] if j < len(self.img) - 1 else 0
        BottomLeft = self.img[i + 1, j - 1] if i < len(self.img) - 1 and j > 0 else 0
        BottomCenter = self.img[i + 1, j] if i < len(self.img) - 1 else 0
        BottomRight = self.img[i + 1, j + 1] if i < len(self.img) - 1 and j < len(self.img) - 1 else 0

        vector.append(TopCenter)
        vector.append(BottomCenter)
        vector.append(TopRight)
        vector.append(CenterRight)
        vector.append(CenterLeft)
        vector.append(BottomLeft)
        vector.append(BottomRight)
        vector.append(TopLeft)
        vector.append(self.img[i, j])
        sorted_vector = sorted(vector)

        return statistics.mode(sorted_vector)
import numpy as np
from PIL import Image


class Fundo:
    def __init__(self, img):
        self.img = img
    def removerFundo(self):

        for i in range(len(self.img)):
            for j in range(len(self.img[i])):
                if self.img[i][j] == 255:
                    self.img[i][j] = 0
        print(self.img)
        Image.fromarray(self.img).show()

        return self.img


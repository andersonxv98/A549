from matplotlib import pyplot as plt
import csv
class GaphClusters:
    def __init__(self):
        pass
    def graficoClusterBinarizado(self):
        with open("clusters.csv", mode='r') as arquivo_csv:
            leitor_csv = csv.reader(arquivo_csv)
            linhas = list(leitor_csv)  # Lendo todas as linhas do arquivo
        dados = linhas
        dados = sorted(dados, key=lambda x: int(x[1]))
        # Ordenar os dados pela intensidade
        

        # Organizar os dados em listas separadas para cada imagem
        _dados = {}
        for row in dados:
            
            # Se a chave existir, adicione os valores
            if row[0] in _dados:
                _dados[row[0]][0].append(int(row[1]))
                _dados[row[0]][1].append(int(row[2]))
                    
            else:
                vet = [int(row[1])],[int(row[2])]
                _dados[row[0]] = vet   


        for data  in _dados:
           
            values = _dados.get(data)

            plt.plot(values[0], values[1], label=data)    
            
            
           
            
            
        plt.legend(loc='upper left')
        plt.xlabel('dimensão')
        plt.ylabel('clusters')
        plt.title('Relação entre dimensões do kernel de abertura e Clusters: ')
            #plt.legend()

            # Mostrar o gráfico
        plt.show()
        # Plotar os gráficos para cada imagem


        # Adicionar legendas e título
        
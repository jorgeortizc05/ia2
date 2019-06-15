import re #Expresiones regulares
import itertools

class Utilities:
    
    def __init__(self, path = 'corpus/digits-database.data'):
        self.path = path
        self.regex = re.compile('(0|1){2,}') # Patrones pares de 0 y unos
        self.regexno = re.compile('(\s)+[0-9]{1}') # Busca un unico numero el cual tenga un espacio o tabulacion antes del mismo.
        
    
    def generate_indices(self):
        _dict = []
        with open(self.path, 'r') as _f: #abre el archivo corpus
            pivote = 0
            flag = False
            lineno = 0
            for line in _f:
                if self.regex.match(line)!=None and not flag:
                    pivote = lineno
                    flag = True
                if self.regexno.match(line)!=None and flag:
                    _dict.append((int(line.replace(' ','')),pivote,lineno))
                    flag = False
                lineno += 1
            _f.close()
            
        return _dict

    def get_digit(self,_slice, _end):
        data = []
        with open(self.path, 'r') as _f:
            for line in itertools.islice(_f, _slice, _end):
                data.append([int(i) for i in line.lstrip().rstrip()])
            
            _f.close()
        return data

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tkinter import *
from tkinter import ttk
import tkinter.messagebox as msg
import numpy as np


utilities = Utilities()

class Ventana(Frame):

    def __init__(self, master = None):
        super().__init__(root)
        self.master = master
        self.coordenadas = [] #Almacena la matriz que se recupera de la interfaz
        self.utilities = Utilities()
        self.indices = self.utilities.generate_indices()
        self.n = []
        self.entrada = []
        self.datos = []
        self.delta = []
        self.init() #llama al init para entrenar la red.

    def normalizador(self):
        for j, k, l in self.indices:
            self.n.append(j)
            self.entrada.append((k,l))

        for i in range(0, len(self.indices)):
            inicio, fin = self.entrada[i]
            fila = np.ravel(np.matrix(self.utilities.get_digit(inicio, fin)))
            self.datos.append(fila)
            self.delta.append(self.n[i])

    def init(self):
        self.master.resizable(0, 0)
        self.grid(row = 0,column = 0)
        self.matriz()
        self.normalizador()
        self.train() #Llama al metodo para entrenar la red

        btnReiniciar = Button(self, text="Reiniciar", height=3, command=self.reiniciar) #Limpia la grilla
        btnReiniciar.grid(columnspan = 16, sticky = W + E + N + S,row = 32, column = 0)

        btnPredecir = Button(self, text="Predecir", height=3, command=self.decode)
        btnPredecir.grid(columnspan = 16, sticky = W + E + N + S,row = 32, column = 16)

    def train(self): #entrena la red
        self.label_encoder = LabelEncoder()
        salida = self.label_encoder.fit_transform(self.delta)
        onehot_encoder = OneHotEncoder(sparse=False)
        salida = salida.reshape(len(salida), 1)
        self.onehot_encoded = onehot_encoder.fit_transform(salida)
        x_train, x_test, d_train, d_test = train_test_split(self.datos, self.onehot_encoded, test_size=0.80, random_state=0)
        self.mlp = MLPClassifier(solver = 'lbfgs', activation='logistic', verbose=True, alpha=1e-4, tol=1e-15, max_iter=500, \
        hidden_layer_sizes=(1024, 800, 400, 200, 10))
        self.mlp.fit(self.datos, self.onehot_encoded)

        prediccion = (np.argmax(self.mlp.predict(x_test), axis = 1) + 1).reshape(-1, 1)
        matriz = confusion_matrix((np.argmax(d_test, axis = 1) + 1).reshape(-1, 1), prediccion)
        print(matriz)

    def decode(self):
        entrada = self.normaliza(32, self.coordenadas)
        numero = np.ravel(np.matrix(entrada))
        res = self.mlp.predict(numero.reshape(1, -1)) #Red ya entrenada
        num = (np.argmax(res, axis=1)+1).reshape(-1, 1)
        aux = []
        matriz = []
        resultado = int(num[0] - 1)
        print(resultado)
        return resultado

    def matriz(self):
        self.btn = [[0 for x in range(32)] for x in range(32)] 
        for x in range(32):
            for y in range(32):
                self.btn[x][y] = Button(self, command=lambda x1=x, y1=y: self.dibujar(x1,y1))
                self.btn[x][y].grid(column = x, row = y)

    def normaliza(self, n, coordenadas): #Transforma la interfaz de botones en una matriz
        matriz = []
        for i in range(n):
            matriz.append([0 for j in range(n)])

        for i in range(len(coordenadas)):
            x, y = coordenadas[i]
            matriz[y][x] = 1
        return matriz

    def dibujar(self, x, y):
        self.btn[x][y].config(bg = "black")
        self.coordenadas.append((x, y))
        
    def reiniciar(self):
        self.matriz()
        self.coordenadas = [] #vacia la matriz

if __name__ == '__main__':
    root = Tk()
    ventana = Ventana(root)
    root.mainloop()
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tkinter import *
from tkinter import ttk
import tkinter.messagebox as msg
import numpy as np
from Utilities import Utilities

class Ventana(Frame):

	def __init__(self, master = None):
		super().__init__(root)
		self.master = master
		self.coordenadas = []
		self.utilities = Utilities()
		self.indices = self.utilities.generate_indices()
		self.n = []
		self.entrada = []
		self.datos = []
		self.delta = []
		self.init()

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
		self.train()
		
		btnReiniciar = Button(self, text="Reiniciar", height=3, command=self.reiniciar)
		btnReiniciar.grid(columnspan = 16, sticky = W + E + N + S,row = 32, column = 0)
		
		btnPredecir = Button(self, text="Predecir", height=3, command=self.decode)
		btnPredecir.grid(columnspan = 16, sticky = W + E + N + S,row = 32, column = 16)		
		
	
	def train(self):
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
		res = self.mlp.predict(numero.reshape(1, -1))
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

	def normaliza(self, n, coordenadas):
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
		self.coordenadas = []
			

if __name__ == '__main__':
	root = Tk()
	ventana = Ventana(root)
	root.mainloop()

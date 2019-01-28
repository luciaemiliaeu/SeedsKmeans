from sklearn.cluster import KMeans
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from copy import deepcopy

class SEEDS(object):
	def __init__ (self, base, num_grupos):
		self.num_grupos = num_grupos
		self.X_train, self.X_test, self.y_train, self.y_test = self.preparacao(base)
		self.sementes = self.calSementes()
		self.cluster = self.kmeans()
		self.test = self.agrupar()
		self.acuracia = accuracy_score(self.y_test, self.cluster)
		self.t = accuracy_score(self.y_test, self.test)

	def preparacao(self, data):		
		Y = data.loc[:,'classe'].get_values()
		X = data.drop(['classe'],axis = 1).get_values()
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.9)
		
		scaler = MinMaxScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.fit_transform(X_test)
		return X_train, X_test, y_train, y_test

	def calSementes(self):
		data = pd.DataFrame(self.X_train)
		data.loc[:,'classe'] = pd.Series(self.y_train, index=data.index)
		grouped = data.groupby(['classe']).mean()
		return grouped.get_values()
	
	def agrupar(self):
		kmeans = KMeans(n_clusters = self.num_grupos, init = self.sementes, n_init= 1).fit_predict(self.X_test)
		for i in range(0,len(kmeans)): kmeans[i]+=1
		return kmeans
	
	def kmeans(self):
		data = self.X_test		
		C = self.sementes
		C_old = np.zeros(C.shape)
		clusters = np.zeros(len(data))
		erro = self.euclidian_distance(C, C_old, None)
		
		while erro != 0:
			for i in range(len(data)):
				distancia = self.euclidian_distance(data[i], C)
				cluster = np.argmin(distancia)
				clusters[i] = cluster
			C_old = deepcopy(C)
			for i in range(self.num_grupos):
				points = [data[j] for j in range(len(data)) if clusters[j] == i]
				C[i] = np.mean(points, axis=0)
			erro = self.euclidian_distance(C, C_old, None)
		
		for i in range(len(clusters)): clusters[i] += 1
		return clusters

	def euclidian_distance (self, a, b, ax=1):
		return np.linalg.norm(a - b, axis=ax)

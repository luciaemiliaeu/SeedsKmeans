import Semente as seed
import pandas as pd 

caminhos = [] 
caminhos.append(('C:\\Users\\LuciaEmilia\\Desktop\\Bases de dados\\iris.csv',3))
caminhos.append(('C:\\Users\\LuciaEmilia\\Desktop\\Bases de dados\\sementes.csv',3))
caminhos.append(('C:\\Users\\LuciaEmilia\\Desktop\\Bases de dados\\vinhos.csv',3))

implementado = []

for i in caminhos:
	soma=0
	summ = 0
	base = pd.read_csv(i[0],sep=',',parse_dates=True)
	for j in range(0, 10):		
		soma += seed.SEEDROTULATOR(base, i[1]).acuracia
	implementado.append(soma/10)

print(implementado)

#Resultado 
#[0.9584126984126984, 0.9612121212121213, 0.8846560846560847, 0.958385093167702]
'''
procurar dos dados cada uma das sementes, salvar num array as que encontrar, verificar a classe da maioria
'''
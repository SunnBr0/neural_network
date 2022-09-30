from array_number import arraynNumber
from neutron import Neuron
import random

class Network:
    def __init__(self,count_neiron,lens_weight):
        self.neirons = []
        for _ in range(count_neiron):
            self.neirons.append(Neuron(lens_weight))
    
    # вывод сети 
    def prediction(self,on_data):
        result = []
        for i in range(len(self.neirons)):
            result.append(float(self.neirons[i].prediction(on_data)))

        return result

    # процесс обучения сети
    def learning(self,on_data,result_data,goal,alpha = 0.01):
        for i in range(len(self.neirons)):
            self.neirons[i].learning(on_data,result_data[i],goal[i],alpha)

    # обучение
    def train(self,array_data,array_goal,epoch = 100):
        for _ in range(epoch):
            for i in range(len(array_data)):
                result = self.prediction(array_data[i])
                self.learning(array_data[i],result,array_goal[i])




network = Network(10,35)


# дата сет данных
array_data = []
# дата сет ожидаемых результатов
array_goal = []
# массив угаданных чисел
result_number = []

for _ in range(100):
    i = random.randint(0,9)
    array_data.append(list(arraynNumber[i]))
    goal = []
    for j in range(10):
        if j == i:
            goal.append(1)
        
        else:
            goal.append(0)

    array_goal.append(goal)

# обучаем нейросеть , выбрав кол-во эпох
epoch = 100
network.train(array_data,array_goal,epoch)

arr = list(range(10))
random.shuffle(arr)


for i in range(len(arr)):
    pred = network.prediction(arraynNumber[arr[i]])
    print("\n",network.prediction(arraynNumber[arr[i]]))
    print("\nНейросеть узнала цифру: ",pred.index(max(pred)))
    result_number.append(pred.index(max(pred)))


print ("Массив ожидаемых цифр:\n",arr)
print("Массив выдаваемых неросетью цифр \n",result_number)






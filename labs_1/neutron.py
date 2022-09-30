import numpy as np
import random
class Neuron:
    def __init__(self,length):
        self.weigths = []
        for _ in range(length):
            self.weigths.append(random.randint(0,1))

    def signoid(self, sum):
        return 1 / (1 + np.exp(-sum))
    
    # для процесс обучения 
    def learning(self, data, result, goal, alpha = 0.01):
        error = goal - result
        for i in range(len(self.weigths)):
            newWeigths = self.weigths[i] + error * alpha * data[i]
            self.weigths[i] = newWeigths

    # вывод нейрона 
    def prediction(self,data):
        net = np.dot(self.weigths,data)
        prediction = self.signoid(net)
        return prediction
    # метод тренировки нейрона
    def train(self,dataset,goalset,step_study = 100):
        for _ in range(step_study):
            for i in len(self.weigths):
                step =  self.prediction(dataset[i])
                self.learning(dataset[i],step,goalset[i])


import numpy as np
from keras.datasets import mnist
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Flatten

#ПОДГОТОВКА ДАННЫХ ДЛЯ НЕРОННОЙ МОДЕЛИ
#====================================================================#
# загружаем изображения
(x_training, y_training), (x_testing, y_testing) = mnist.load_data()
# каждый пискель представляем в  значении от 0 до 255(оттенки серого)
x_train = x_training / 255.0
x_test = x_testing / 255.0
# преобразуем десятичные числа в вектор типа 3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] 
y_train = to_categorical(y_training, 10)
y_test = to_categorical(y_testing, 10)
#====================================================================#





#ПАРАМЕТРЫ НЕЙРОННОЙ МОДЕЛИ
#====================================================================#
# Создает модель
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dropout(0.6),
    Dense(10, activation='softmax')
])
# определяем все параметры нейронной сети (какой используем оптимизатор, 
# какую используем функцию потерь, список метрик, с помощью которых оценим качество работы НС)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# функция для обучения
# epochs - количество эпох 
model.fit(x_train, y_train, batch_size=32, epochs=10)
#====================================================================#



#ПРОВЕРКА НЕЙРОНКИ
#====================================================================#
# задаём изображение на котором будем проверять
test_number = 10
print(f"Точность в процентах: {round(model.evaluate(x_test, y_test)[1], 3) * 100}%")
print(f"Картинка из dataset mnist № {test_number} на ней цифра {y_testing[test_number]}")
x = np.expand_dims(x_test[test_number], axis=0)
res = model.predict(x)
print(res)
print(f"Нейросеть на картинке #{test_number} с цифрой {y_testing[test_number]} узнала, определила: {np.argmax(res)}")
#====================================================================#
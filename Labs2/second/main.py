import json
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten

# Читает данные с json
def load_dataJson(name_file):
    with open(name_file, 'r') as f:
        data = json.load(f)
    x_data = []
    y_data = []
    for key, value in data.items():
        x_data.append(value)
        y_data.append(int(key))
    f.close()
    return np.array(x_data), np.array(y_data)



#ПОДГОТОВКА ДАННЫХ ДЛЯ НЕРОННОЙ МОДЕЛИ
#====================================================================#
# массивы с данными для выборки и  для тренировки 
x_training, y_training = load_dataJson("./trainstudy.json")

x_testprop, y_testprop = load_dataJson("./testing.json")

numbers = y_training
# Переделывает числа 0, 1, 2... в массив из 0 и 1
y_training = to_categorical(y_training, 10)

y_testprop = to_categorical(y_testprop, 10)
#====================================================================#


#ПАРАМЕТРЫ НЕЙРОННОЙ МОДЕЛИ
#====================================================================#
#Sequential модель подходит для простого стека слоев , 
# где каждый слой имеет ровно один тензор входной и один выходной тензор.
model = Sequential([
    #сетка 5 на 7
    Flatten(input_shape=(7, 5)),
    Dense(10, activation='relu'),
    Dense(10, activation='softmax')
])
# Компилирует модель с определенными характеристиками
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Обучает модель с определенным колличеством итераций
# функция для обучения
# epochs - количество эпох 
model.fit(x_training, y_training, batch_size=32, epochs=200)
#====================================================================#




#ПРОВЕРКА НЕЙРОНКИ
#====================================================================#
test_number = 5
print(f"Точность в процентах =  {round(model.evaluate(x_testprop, y_testprop)[1], 3) * 100}%")
print(f"Картинка из dataset testing № {test_number} на ней цифра {numbers[test_number]}")
x = np.expand_dims(x_testprop[test_number], axis=0)
res = model.predict(x)
print(f"Нейросеть на картинке № {test_number} с цифрой {numbers[test_number]} узнала, определила: {np.argmax(res)}")
#====================================================================#
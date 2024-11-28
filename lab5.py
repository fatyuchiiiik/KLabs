import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Загрузка набора данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model = Sequential()
model.add(Flatten(input_shape=(28, 28))) # Преобразование изображений в одномерный вектор
model.add(Dense(128, activation='relu')) # Скрытый слой с 128 нейронами и функцией активации ReLU
model.add(Dense(10, activation='softmax')) # Выходной слой с 10 нейронами (для 10 цифр) и функцией активации softmax

model.compile(optimizer='adam',
       loss='categorical_crossentropy',
       metrics=['accuracy'])

# обучение модели и сохранение истории обучения
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# оцениваб модель
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# model.save("mnist_model.h5")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Точность модели')
plt.ylabel('Точность')
plt.xlabel('эпоха')
plt.legend(['обучение', 'тест'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Потери модели')
plt.ylabel('Потери')
plt.xlabel('эпоха')
plt.legend(['обучение', 'тест'], loc='upper left')
plt.show()

random_index = np.random.randint(0, len(x_test))
random_image = x_test[random_index]
random_label = y_test[random_index]

predictions = model.predict(np.expand_dims(random_image, axis=0))
predicted_class = np.argmax(predictions)

plt.figure()
plt.imshow(random_image, cmap='gray')
plt.title(f"случайная: {np.argmax(random_label)}, предсказание: {predicted_class}")
plt.show()
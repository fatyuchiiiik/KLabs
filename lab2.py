import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import Counter

class KNNapp:
  def __init__(self, master):
    self.master = master
    master.title("Метод К-ближайших соседей")

    # Фрейм для выбора файла
    file_frame = tk.Frame(master)
    file_frame.pack(pady=10)

    self.file_path = tk.StringVar(value="")
    tk.Label(file_frame, text="Выберите файл данных:").pack(side="left")
    tk.Entry(file_frame, textvariable=self.file_path, width=40).pack(side="left")
    tk.Button(file_frame, text="Обзор", command=self.open_file).pack(side="left")

    # Фрейм для значения K
    split_frame_k = tk.Frame(master)
    split_frame_k.pack(pady=10)

    tk.Label(split_frame_k, text="Введите значение K:").pack(side="left")
    self.k = tk.IntVar(value=3)
    tk.Entry(split_frame_k, textvariable=self.k, width=5).pack(side="left")

    # Фрейм для разделения данных
    split_frame = tk.Frame(master)
    split_frame.pack(pady=10)

    tk.Label(split_frame, text="Пропорция разделения (обучение/тест):").pack(side="left")
    self.test_size = tk.DoubleVar(value=0.8)
    tk.Entry(split_frame, textvariable=self.test_size, width=5).pack(side="left")

    # Фрейм для графика
    graph_frame = tk.Frame(master)
    graph_frame.pack(pady=10)
    self.figure = plt.Figure(figsize=(6, 4))
    self.ax = self.figure.add_subplot(111)
    self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
    self.canvas.draw()
    self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Фрейм для кнопок
    button_frame = tk.Frame(master)
    button_frame.pack(pady=10)
    tk.Button(button_frame, text="Построить график", command=self.split_data).pack(side="left")
    tk.Button(master, text="Показать ближайших соседей", command=self.show_neighbors).pack()
    tk.Button(master, text="Показать график погрешности", command=self.show_error_graph).pack()

    # Фрейм для вывода результатов
    result_frame = tk.Frame(master)
    result_frame.pack(pady=10)
    self.error_label = tk.Label(result_frame, text="Погрешность модели: ")
    # self.neighbors_label = tk.Label(result_frame, text="Соседи и их классы: ")
    self.error_label.pack()
    # self.neighbors_label.pack()

    self.data = None
    self.X_train = None
    self.X_test = None
    self.y_train = None
    self.y_test = None
    self.model = None

  def open_file(self):
        file_path = filedialog.askopenfilename(
            initialdir="/", title="Выбрать файл",
            filetypes=(("CSV файлы", "*.csv"), ("Все файлы", "*.*"))
        )
        if file_path:
            self.file_path.set(file_path)
            self.load_data()

  def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path.get(), sep=";", encoding='utf-8')
            self.X = self.data[['X1', 'X2']].to_numpy()  # Замените на ваши признаки
            self.y = self.data['Y'].to_numpy()
        except FileNotFoundError:
            tk.messagebox.showerror("Ошибка", "Файл не найден!")
            return

  def split_data(self):
        try:
            test_size = self.test_size.get()
            if 0 < test_size < 1:
                # Разделение данных без sklearn
                n = len(self.X)
                split_index = int(test_size * n)
                self.X_train = self.X[:split_index]
                self.X_test = self.X[split_index:]
                self.y_train = self.y[:split_index]
                self.y_test = self.y[split_index:]
                self.train_model()
                self.plot_data()
            else:
                tk.messagebox.showerror("Ошибка", "Неверный размер тестовой выборки!")
        except ValueError:
            tk.messagebox.showerror("Ошибка", "Неверный формат размера тестовой выборки!")

  def train_model(self):
        # Функция для обучения модели KNN (вызов функции KNN)
        self.model = KNN(k=self.k.get())
        self.model.fit(self.X_train, self.y_train)

  def plot_data(self):
        self.ax.clear()
        self.ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, label="Обучающие данные")
        self.ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, marker="x", label="Тестовые данные")
        self.ax.legend()
        self.canvas.draw()

  def show_neighbors(self):
        try:
            if self.model is None:
                tk.messagebox.showerror("Ошибка", "Модель не обучена!")
                return
                        # Перебираем точки из тестовой выборки
            all_neighbors = []
            all_neighbor_classes = []
            all_test_points = []
            all_test_classes = []
            for i in range(len(self.X_test)):
                test_point = self.X_test[i]
                test_class = self.y_test[i]
                # Получаем индексы ближайших соседей
                neighbors_indices = self.model.find_nearest_neighbors(test_point, self.k.get())
                neighbors = self.X_train[neighbors_indices]
                neighbor_classes = self.y_train[neighbors_indices]
                
                # Собираем данные для отрисовки
                all_neighbors.extend(neighbors)
                all_neighbor_classes.extend(neighbor_classes)
                all_test_points.append(test_point)
                all_test_classes.append(test_class)

            # Вычисляем прогноз для тестовой точки
            predictions = [self.model.predict(test_point) for test_point in self.X_test]

            # Вычисляем погрешность модели
            error = 1 - self.calculate_accuracy(self.y_test, predictions)
            self.error_label.config(text=f"Погрешность модели: {error:.2f}")

            # Выводим информацию о ближайших соседях
            print("Соседи и их классы:")
            for i in range(len(self.X_test)):
                print(f"Тестовая точка: {all_test_points[i]}, Класс: {all_test_classes[i]}, Соседи: {all_neighbors[i*self.k.get():(i+1)*self.k.get()]}, Классы: {all_neighbor_classes[i*self.k.get():(i+1)*self.k.get()]}")

            # Выводим график с ближайшими соседями
            self.ax.clear()
            self.ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, label="Обучающие данные")
            self.ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, marker="x", label="Тестовые данные")
            self.ax.scatter(np.array(all_neighbors)[:, 0], np.array(all_neighbors)[:, 1], c=all_neighbor_classes, marker="o", s=80, label="Ближайшие соседи")
            for i in range(len(self.X_test)):
                self.ax.scatter(all_test_points[i][0], all_test_points[i][1], c="red", marker="*", s=20)  # Выделим тестовую точку
            self.ax.legend()
            self.canvas.draw()
        except IndexError:
            tk.messagebox.showerror("Ошибка", "Неверный индекс тестовой точки!")

  def calculate_accuracy(self, y_true, y_pred):
        correct_predictions = sum(y_true == y_pred)
        return correct_predictions / len(y_true)
    
  def show_error_graph(self):
        if self.X is None or self.y is None:
            tk.messagebox.showerror("Ошибка", "Сначала выберите файл данных!")
            return

        # Создаем список K для графика
        k_values = [3, 5, 7, 9, 11, 13, 15]  # Значения K от 2 до 10

        # Вычисляем погрешность для каждого K
        errors = []
        for k in k_values:
            self.model = KNN(k=k)
            self.model.fit(self.X_train, self.y_train)
            predictions = [self.model.predict(test_point) for test_point in self.X_test]
            error = 1 - self.calculate_accuracy(self.y_test, predictions)
            errors.append(error)

        # Создаем новый график
        plt.figure(figsize=(6, 4))
        plt.plot(k_values, errors)
        plt.title("Зависимость погрешности от K")
        plt.xlabel("K")
        plt.ylabel("Погрешность")
        plt.show()
    

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def find_nearest_neighbors(self, test_point, k):
        distances = np.linalg.norm(self.X_train - test_point, axis=1)
        nearest_neighbors_indices = np.argsort(distances)[:k]
        return nearest_neighbors_indices

    def predict(self, test_point):
        neighbors_indices = self.find_nearest_neighbors(test_point, self.k)
        neighbor_classes = self.y_train[neighbors_indices]
        # Выбрать класс, который встречается чаще всего
        prediction = Counter(neighbor_classes).most_common(1)[0][0]
        return prediction

if __name__ == "__main__":
    root = tk.Tk()
    app = KNNapp(root)
    root.mainloop()





import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import Counter, defaultdict
from math import sqrt

def euklid_distance(a, b):
  return sqrt(sum([(a[i] - b[i]) ** 2 for i in range(len(a))]))

def k_means(X, k, max_iterations=100):
  # Инициализация центров кластеров случайным образом
  n = len(X)
  centroids = X[np.random.choice(n, k, replace=False)]

  for _ in range(max_iterations):
    # Создание словаря для хранения точек, принадлежащих каждому кластеру
    clusters = defaultdict(list)
    
    # Присваивание каждой точки кластеру с ближайшим центроидом
    for point in X:
      distances = [euklid_distance(point, centroid) for centroid in centroids]
      cluster_id = distances.index(min(distances))
      clusters[cluster_id].append(point)

    # Обновление центроидов как среднего значения точек в каждом кластере
    new_centroids = [np.mean(cluster, axis=0) for cluster in clusters.values()]

    # Проверка, изменились ли центроиды
    if all([np.array_equal(centroids[i], new_centroids[i]) for i in range(k)]):
      break

    centroids = new_centroids

  # Возврат кластеров и центроидов
  return clusters, centroids

def calculate_accuracy(clusters, true_labels):
  """Рассчитывает точность классификации."""
  # Создание словаря для хранения истинного класса для каждого кластера
  cluster_to_true_class = defaultdict(list)
  for cluster_id, points in clusters.items():
    for i, point in enumerate(points):
      cluster_to_true_class[cluster_id].append(true_labels[i])

  # Подсчет точности для каждого кластера
  correct_predictions = 0
  for cluster_id, true_classes in cluster_to_true_class.items():
    most_common_class = Counter(true_classes).most_common(1)[0][0]
    correct_predictions += true_classes.count(most_common_class)

  # Возврат точности
  return correct_predictions / len(true_labels)

class KNNapp:
  def __init__(self, master):
    self.master = master
    master.title("Метод K-средних")

    # Фрейм для выбора файла
    file_frame = tk.Frame(master)
    file_frame.pack(pady=10)
    self.file_path = tk.StringVar(value="")
    tk.Label(file_frame, text="Выберите файл данных:").pack(side="left")
    tk.Entry(file_frame, textvariable=self.file_path, width=40).pack(side="left")
    tk.Button(file_frame, text="Обзор", command=self.open_file).pack(side="left")

    # Фрейм для значения K
    k_frame = tk.Frame(master)
    k_frame.pack(pady=10)

    tk.Label(k_frame, text="Количество кластеров (K):").pack(side="left")
    self.k = tk.IntVar(value=3)
    tk.Entry(k_frame, textvariable=self.k, width=5).pack(side="left")

    # Фрейм для графика
    graph_frame = tk.Frame(master)
    graph_frame.pack(pady=10)
    self.figure = plt.Figure(figsize=(6, 4))
    self.ax = self.figure.add_subplot(111)
    self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
    self.canvas.draw()
    self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Фрейм для кнопки
    button_frame = tk.Frame(master)
    button_frame.pack(pady=10)
    tk.Button(button_frame, text="Кластеризовать", command=self.cluster_data).pack()

    # Фрейм для вывода результатов
    result_frame = tk.Frame(master)
    result_frame.pack(pady=10)
    self.error_label = tk.Label(result_frame, text="Погрешность модели: ")
    self.error_label.pack()

    self.data = None
    self.X = None
    self.y = None
    self.clusters = None
    self.centroids = None

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
      self.X = self.data[['X1', 'X2']].to_numpy() # Замените на ваши признаки
      self.y = self.data['Y'].to_numpy()
    except FileNotFoundError:
      tk.messagebox.showerror("Ошибка", "Файл не найден!")
      return

  def cluster_data(self):
    if self.X is None or self.y is None:
      tk.messagebox.showerror("Ошибка", "Сначала выберите файл данных!")
      return

    k = self.k.get()
    self.clusters, self.centroids = k_means(self.X, k)

    # Вычисляем точность классификации
    accuracy = calculate_accuracy(self.clusters, self.y)
    self.error_label.config(text=f"Погрешность модели: {1-accuracy:.2f}")
    
    print("Изначальные классы и кластеры:")
    for cluster_id, points in self.clusters.items():
      for i, point in enumerate(points):
        print(f"Точка: {point}, Изначальный класс: {self.y[i]}, Кластер: {cluster_id + 1}")

    # Отображаем кластеры на графике
    self.plot_clusters()

  def plot_clusters(self):
    self.ax.clear()
    for cluster_id, points in self.clusters.items():
      self.ax.scatter(np.array(points)[:, 0], np.array(points)[:, 1], label=f"Кластер {cluster_id+1}")
  
  # Преобразуем self.centroids в массив NumPy
    self.centroids = np.array(self.centroids)
    self.ax.scatter(self.centroids[:, 0], self.centroids[:, 1], marker="x", s=40, c="black", label="Центроиды")
    self.ax.set_title("Кластеризация методом K-средних")
    self.ax.set_xlabel("X1")
    self.ax.set_ylabel("X2")
    self.ax.legend()
    self.canvas.draw()

if __name__ == "__main__":
  root = tk.Tk()
  app = KNNapp(root)
  root.mainloop()
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RegressionApp:
  def __init__(self, master):
    self.master = master
    master.title("Однофакторная линейная регрессия")

    # Фрейм для выбора файла
    file_frame = tk.Frame(master)
    file_frame.pack(pady=10)

    self.file_path = tk.StringVar(value="")
    tk.Label(file_frame, text="Выберите файл данных:").pack(side="left")
    tk.Entry(file_frame, textvariable=self.file_path, width=40).pack(side="left")
    tk.Button(file_frame, text="Обзор", command=self.open_file).pack(side="left")

    # Фрейм для разделения данных
    split_frame = tk.Frame(master)
    split_frame.pack(pady=10)

    tk.Label(split_frame, text="Пропорция разделения (обучение/тест):").pack(side="left")
    self.test_size = tk.DoubleVar(value=0.2)
    tk.Entry(split_frame, textvariable=self.test_size, width=5).pack(side="left")

    # Фрейм для отображения целевой функции
    func_frame = tk.Frame(master)
    func_frame.pack(pady=10)
    self.func_label = tk.Label(func_frame, text="Целевая функция Y (Стоимость аренды): ")
    self.func_label.pack()

    # Фрейм для выбора независимой переменной
    var_frame = tk.Frame(master)
    var_frame.pack(pady=10)
    self.var_label = tk.Label(var_frame, text="Выберите параметр (X1- площадь в м2, X2- количество комнат, X3- этаж):")
    self.var_label.pack(side="left")
    # params = ["1", "2", "3"]
    self.var_choice = tk.StringVar(value="")
    self.var_dropdown = ttk.Combobox(var_frame, textvariable=self.var_choice, width=10, state="readonly")
    self.var_dropdown.pack(side="left")

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
    tk.Button(button_frame, text="Построить график", command=self.plot_data).pack(side="left")
    tk.Button(button_frame, text="Обучить модель", command=self.train_model).pack(side="left")

    # Фрейм для вывода погрешности
    error_frame = tk.Frame(master)
    error_frame.pack(pady=10)
    self.error_label = tk.Label(error_frame, text="Погрешность модели: ")
    self.error_label.pack()
    

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
      print(self.data)
      self.func_label.config(text=f"Целевая функция: {self.data.columns[-1]}")
      self.var_dropdown["values"] = ["X1","X2","X3"]
      self.plot_data()
    except FileNotFoundError:
      tk.messagebox.showerror("Ошибка", "Файл не найден!")
      return

  def split_data(self):
    independent_var = self.var_choice.get()
    self.X_train = self.data[independent_var].values.reshape(-1, 1)
    self.y_train = self.data[self.data.columns[-1]].values
    self.X_test = self.X_train[int(len(self.X_train) * (1 - self.test_size.get())):]
    self.y_test = self.y_train[int(len(self.y_train) * (1 - self.test_size.get())):]
    self.X_train = self.X_train[:int(len(self.X_train) * (1 - self.test_size.get()))]
    self.y_train = self.y_train[:int(len(self.y_train) * (1 - self.test_size.get()))]

  def train_model(self):
    self.split_data()
    self.model = LinearRegression()
    self.model.fit(self.X_train, self.y_train)
    self.calculate_error()
    self.plot_data()

  def calculate_error(self):
    if self.model is not None:
      y_pred = self.model.predict(self.X_test)
      error = np.mean((self.y_test - y_pred) ** 2)
      self.error_label.config(text=f"Погрешность модели: {error:.2f}")

  def plot_data(self):
    self.ax.clear()
    if self.data is not None:
      independent_var = self.var_choice.get()
      self.ax.scatter(self.data[independent_var], self.data[self.data.columns[-1]], label="Данные")
      if self.model is not None:
        x_range = np.linspace(self.X_train.min(), self.X_train.max(), 100)
        y_pred = self.model.predict(x_range.reshape(-1, 1))
        self.ax.plot(x_range, y_pred, color="red", label="Линейная регрессия")
        self.ax.legend()
      self.ax.set_xlabel(independent_var)
      self.ax.set_ylabel(self.data.columns[-1])
      self.ax.set_title("ГРАФИК")
      self.canvas.draw()

class LinearRegression:
  def __init__(self):
    self.w = None

  def fit(self, X, y):
    X = np.c_[np.ones(X.shape[0]), X]
    self.w = np.linalg.solve(X.T @ X, X.T @ y)

  def predict(self, X):
    X = np.c_[np.ones(X.shape[0]), X]
    return X @ self.w 

root = tk.Tk()
app = RegressionApp(root)
root.mainloop()
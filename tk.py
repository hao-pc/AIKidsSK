import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# Путь к локальной базе данных
db_path = "db"

# Создаем или подключаемся к существующей базе данных
chroma_client = chromadb.PersistentClient(path=db_path)

# Инициализация функции эмбеддинга для мультимодальных данных
multimodal_ef = OpenCLIPEmbeddingFunction()

# Инициализация загрузчика изображений
image_loader = ImageLoader()

# Создаем коллекцию или используем существующую с загрузчиком данных
multimodal_db = chroma_client.get_or_create_collection(
    name="multimodal_db",
    embedding_function=multimodal_ef,
    data_loader=image_loader
)


# Функция для добавления новых изображений и описаний в базу данных
def add_images_to_db(file_paths):
    ids = []
    uris = []
    metadatas = []

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        ids.append(file_name)  # Используем имя файла как уникальный ID
        uris.append(file_path)  # Путь к изображению
        metadatas.append({'description': file_name})  # Добавляем описание

    # Добавляем новые записи в базу данных
    multimodal_db.add(
        ids=ids,
        uris=uris,
        metadatas=metadatas
    )


# Функция для выполнения запроса к базе данных и получения результатов
def query_db(query_texts, n_results=1):
    query_results = multimodal_db.query(
        query_texts=query_texts,
        n_results=n_results,
        include=['documents', 'distances', 'metadatas', 'data', 'uris'],
    )
    return query_results


# GUI для ввода запроса и отображения результатов
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Поиск животных")

        # Поле для ввода текста запроса
        self.query_label = tk.Label(root, text="Введите запрос:")
        self.query_label.pack(pady=5)

        self.query_entry = tk.Entry(root, width=40)
        self.query_entry.pack(pady=5)

        # Кнопка для выполнения запроса
        self.query_button = tk.Button(root, text="Искать", command=self.execute_query)
        self.query_button.pack(pady=10)

        # Область для отображения результатов
        self.results_frame = tk.Frame(root)
        self.results_frame.pack(pady=20)

    def execute_query(self):
        # Получаем текст запроса из поля ввода
        query_text = self.query_entry.get().strip()
        if not query_text:
            messagebox.showwarning("Ошибка", "Введите запрос!")
            return

        # Выполняем запрос и выводим результаты
        query_results = query_db([query_text])
        self.display_results(query_text, query_results)

    def display_results(self, query_text, query_results):
        # Очищаем предыдущие результаты
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        for i, (id, distance, metadata, uri) in enumerate(zip(query_results['ids'][0],
                                                              query_results['distances'][0],
                                                              query_results['metadatas'][0],
                                                              query_results['uris'][0])):
            # Отображаем текстовые результаты
            result_text = f'ID: {id}\nDistance: {distance:.4f}\nDescription: {metadata.get("description", "Нет описания")}'
            result_label = tk.Label(self.results_frame, text=result_text, wraplength=400, justify='left')
            result_label.pack(pady=5)

            # Загружаем и отображаем изображение
            image = plt.imread(uri)  # uri теперь будет локальным путем
            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.axis("off")

            # Вставляем изображение в tkinter через Canvas
            canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=5)


# Запуск приложения
root = tk.Tk()
app = App(root)
root.mainloop()

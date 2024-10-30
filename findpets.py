import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from matplotlib import pyplot as plt
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


# Функция для выполнения запроса и отображения результатов
def print_query_results(query_list, query_results):
    for i, query in enumerate(query_list):
        print(f'Results for query: {query}')

        for id, distance, metadata, uri in zip(query_results['ids'][i], query_results['distances'][i],
                                               query_results['metadatas'][i], query_results['uris'][i]):
            print(f'id: {id}, distance: {distance}, metadata: {metadata}')
            print(f'description: {metadata.get("description", "No description available")}')

            # Загрузка изображения и его отображение
            image = plt.imread(uri)  # uri теперь будет локальным путем
            plt.imshow(image)
            plt.axis("off")
            plt.show()


# Пример добавления новых изображений и метаданных в базу данных
images_folder = 'saved_images'
file_paths = [os.path.join(images_folder, filename) for filename in os.listdir(images_folder) if
              filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
add_images_to_db(file_paths)


# Функция для выполнения запроса к базе данных и получения результатов
def query_db(query_texts, n_results=1):
    query_results = multimodal_db.query(
        query_texts=query_texts,
        n_results=n_results,
        include=['documents', 'distances', 'metadatas', 'data', 'uris'],
    )
    return query_results


# Пример выполнения запроса
query_texts = ['dog']  # Замените на ваш запрос
query_results = query_db(query_texts)

# Печать и отображение результатов
print_query_results(query_texts, query_results)


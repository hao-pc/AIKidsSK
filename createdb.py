import asyncio
import aiohttp
import uuid
import json
import os
import numpy as np
import torch
import clip
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from chromadb import PersistentClient

# Инициализация подключения к Chromadb
client = PersistentClient(path="db") # укажите путь к директории для сохранения
collection = client.create_collection("pet911_catalog")

# Загрузка модели CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Асинхронная функция для запроса страницы
async def fetch_page(session, url):
    async with session.get(url) as response:
        return await response.text()

# Асинхронная загрузка изображения
async def fetch_image(session, url):
    async with session.get(url) as response:
        if response.status == 200:
            image_data = await response.read()
            return Image.open(BytesIO(image_data))
        return None

# Извлечение URL-ов элементов каталога с главной страницы
def extract_item_urls(page_content):
    """Извлекает ссылки на элементы из страницы каталога."""
    soup = BeautifulSoup(page_content, 'html.parser')
    items = soup.select('div.catalog__items a.catalog-item__thumb')
    return ['https://pet911.ru' + item['href'] for item in items]

# Извлечение URL фото и описания с страницы элемента
def extract_photo_and_description(page_content):
    """Извлекает URL фото и описание из страницы элемента."""
    soup = BeautifulSoup(page_content, 'html.parser')
    photo_url = soup.select_one('img.img-crop')['src']
    if not photo_url.startswith('http'):
        photo_url = 'https://pet911.ru' + photo_url
    description = soup.select_one('div.card__descr').get_text(separator=" ").strip()
    return photo_url, description

# Разделение длинного текста на блоки по допустимому числу токенов
def split_text_to_chunks(text, max_tokens=70):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(clip.tokenize([" ".join(current_chunk)])[0]) > max_tokens:
            current_chunk.pop()
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Функция создания эмбеддинга для изображения
def get_image_embedding(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image).cpu().numpy().flatten()
    return image_embedding

# Функция создания эмбеддинга для текста
def get_text_embedding(text):
    chunks = split_text_to_chunks(text)
    chunk_embeddings = []

    for chunk in chunks:
        text_tokens = clip.tokenize([chunk]).to(device)
        with torch.no_grad():
            chunk_embedding = model.encode_text(text_tokens).cpu().numpy().flatten()
        chunk_embeddings.append(chunk_embedding)

    # Усреднение эмбеддингов для всех частей текста
    combined_embedding = np.mean(chunk_embeddings, axis=0)
    return combined_embedding

# Нормализация вектора
def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

# Папка для сохранения изображений
images_folder = "saved_images"
os.makedirs(images_folder, exist_ok=True)

# Асинхронный парсинг элемента каталога
async def parse_item(session, item_url):
    page_content = await fetch_page(session, item_url)
    photo_url, description = extract_photo_and_description(page_content)

    # Загрузка изображения по URL
    image = await fetch_image(session, photo_url)
    if image is None:
        print(f"Не удалось загрузить изображение по URL: {photo_url}")
        return None

    # Сохранение изображения в локальную папку
    image_filename = os.path.join(images_folder, os.path.basename(photo_url))
    image.save(image_filename)

    # Получение эмбеддингов для фото и описания
    image_vector = get_image_embedding(image)
    text_vector = get_text_embedding(description)
    combined_vector = np.mean([image_vector, text_vector], axis=0)
    normalized_vector = normalize_vector(combined_vector)

    # Генерация уникального ID и формирование JSON-документа
    document = json.dumps({"photo_url": photo_url, "description": description})
    unique_id = str(uuid.uuid4())

    return unique_id, document, normalized_vector, image_filename  # Возвращаем путь к изображению

# Основная функция запуска парсинга и записи данных в базу
async def main():
    base_url = 'https://pet911.ru/catalog'
    async with aiohttp.ClientSession() as session:
        # Получаем главную страницу каталога и извлекаем ссылки на элементы
        page_content = await fetch_page(session, base_url)
        item_urls = extract_item_urls(page_content)

        # Создаём задачи для парсинга каждого элемента
        tasks = [parse_item(session, item_url) for item_url in item_urls]
        results = await asyncio.gather(*tasks)

        # Фильтруем успешные результаты
        results = [result for result in results if result is not None]

        if results:
            # Пакетное добавление данных в базу
            ids, documents, embeddings, image_paths = zip(*results)
            collection.add(
                ids=list(ids),
                documents=list(documents),
                embeddings=list(embeddings),
                uris=list(image_paths)  # Сохраняем пути к изображениям
            )
            print(f"Добавлено {len(results)} элементов в базу данных.")
        else:
            print("Не удалось добавить элементы в базу данных.")

if __name__ == "__main__":
    asyncio.run(main())
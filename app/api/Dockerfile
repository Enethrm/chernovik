FROM python:3.12

WORKDIR /api
COPY requirements.txt /api/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Копируем файлы проекта
COPY . /api
EXPOSE 8000
# Указываем команду для запуска api
CMD ["python3", "app.py"]
FROM python:3.12

WORKDIR /bot
COPY requirements.txt /bot/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем файлы проекта
COPY . /bot

# Указываем команду для запуска бота
CMD ["python3", "main.py"]
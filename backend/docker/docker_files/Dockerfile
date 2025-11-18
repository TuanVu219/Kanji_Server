# Dockerfile for Django API
FROM python:3.12

# Không ghi cache bytecode, output không buffer
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Cài đặt các dependency cơ bản
RUN apt-get update && \
    apt-get install -y netcat-openbsd gettext && \
    rm -rf /var/lib/apt/lists/*

# Tạo thư mục code trong container
RUN mkdir /code
WORKDIR /code

# Copy toàn bộ code vào container
COPY . /code/

# Cài pip và các requirements
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# Mở port 8080 cho Daphne
EXPOSE 8080

# ENTRYPOINT: chạy Daphne trực tiếp
CMD ["daphne", "-b", "0.0.0.0", "-p", "8080", "drf_course_main.asgi:application"]

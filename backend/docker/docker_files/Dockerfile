# Dockerfile for Django API + EfficientNet
FROM python:3.12-slim

# Không buffer output, không ghi bytecode
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Cài các dependency cơ bản
RUN apt-get update && \
    apt-get install -y --no-install-recommends netcat-openbsd gettext git && \
    rm -rf /var/lib/apt/lists/*

# Tạo thư mục code
WORKDIR /code

# Copy chỉ file requirements trước (tận dụng Docker cache)
COPY requirements.txt /code/

# Cài pip + requirements
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code
COPY . /code/

# Expose port cho Daphne
EXPOSE 8080

# Đặt entrypoint chạy Daphne
CMD ["daphne", "-b", "0.0.0.0", "-p", "8080", "drf_course_main.asgi:application"]

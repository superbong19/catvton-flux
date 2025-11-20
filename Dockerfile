# ================================
# 🚀 Dockerfile for catvton-flux
# GPU, HuggingFace, Google GenAI
# ================================

# 1. Base image: PyTorch + CUDA
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn9-devel

# 2. Tạo thư mục app
WORKDIR /app

# 3. Copy source code
COPY . /app

# 4. Upgrade pip
RUN pip install --upgrade pip

# 5. Cài đặt dependencies từ requirements.txt
RUN pip install -r requirements.txt

# 6. Cài Google GenAI SDK
RUN pip install -q -U google-genai

# 7. Login HuggingFace bằng token từ ENV
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN huggingface-cli login --token $HF_TOKEN

# 8. Expose port cho API
EXPOSE 8000

# 9. Command chạy API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

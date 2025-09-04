# MNIST 手写数字识别

## 项目结构
```text
mnist-app/
├── app.py              # 您的 Streamlit 应用
├── train.py            # 训练脚本
├── requirements.txt    # Python 依赖
├── Dockerfile          # Docker 构建文件
├── .dockerignore       # Docker 忽略文件
└── mnist.h5            # 训练好的模型（运行 train.py 后生成）
```

## 安装依赖

```bash
pip install --no-cache-dir -r requirements.txt
```

## 构建和运行容器

```bash
docker compose up -d --build
```
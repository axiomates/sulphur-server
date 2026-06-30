# Sulphur-2 Video Generation Server

类似 llama-server，加载 GGUF 到显存，常驻等待请求生成视频。

## 从头开始

**前提**：Windows，NVIDIA 显卡 ≥12 GB 显存，Python 3.10+。

打开终端（PowerShell 或 bash），一步一步来：

### 1. 装 PyTorch（如果没装）

不需要全局安装 CUDA toolkit，只需 NVIDIA 驱动 + 下面这条命令：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

验证 GPU 可用：

```bash
python -c "import torch; print(torch.cuda.is_available())"
# 必须输出 True
```

### 2. 进项目装依赖

```bash
cd sulphur-server
pip install -r requirements.txt
```

### 3. 下载 GGUF 文件

浏览器打开：

```
https://huggingface.co/Abiray/Sulphur-2-base-GGUF/tree/main
```

下载 `sulphur_dev-Q3_K_M.gguf`（11.1 GB），放到项目目录下。

### 4. 下载 base pipeline（只需一次）

```bash
python prepare_base.py
```

下载到 `./LTX-2.3-Diffusers/`，约 50 GB。等着就行。

### 5. 启动

```bash
python server.py --model ./LTX-2.3-Diffusers --gguf ./sulphur_dev-Q3_K_M.gguf
```

看到 `Server ready` 就成功了。

### 6. 测试生成

```bash
curl -X POST http://localhost:8080/v1/video/generate -H "Content-Type: application/json" -d "{\"prompt\": \"A cat walking on a sunny street\", \"num_frames\": 121, \"seed\": 42}"
```

返回 `{"task_id": "xxx", "status": "queued"}`。

```bash
# 查状态（出来 done 才能下载）
curl http://localhost:8080/v1/video/status/xxx

# 下载视频
curl http://localhost:8080/v1/video/result/xxx -o output.mp4
```

## 资源

| | 需要 | 说明 |
|---|---|---|
| 硬盘 | ~61 GB | GGUF 11 GB + base pipeline 50 GB |
| 显存 | ~12 GB | GGUF 常驻 GPU，其余组件放 CPU |

## 离线部署

联网机器上做完步骤 3 和 4，把整个目录拷到离线机器，然后直接步骤 2 + 5。

## API 速查

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/video/generate` | 提交任务，返回 task_id |
| GET | `/v1/video/status/{id}` | 查状态 |
| POST | `/v1/video/cancel/{id}` | 取消排队 |
| GET | `/v1/video/result/{id}` | 下载 mp4 |
| GET | `/health` | 服务状态 |

状态：`queued` → `processing` → `done` / `failed` / `cancelled`

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `diffusers/LTX-2.3-Diffusers` | base pipeline |
| `--gguf` | 无 | GGUF 文件路径 |
| `--host` | `0.0.0.0` | |
| `--port` | `8080` | |
| `--concurrency` | `1` | 单 GPU 别改 |
| `--queue-size` | `8` | 排满即拒 |

## 常见错误码

| HTTP | 含义 |
|------|------|
| 400 | 参数不对 |
| 404 | task_id 不存在 |
| 409 | 已开始，不能取消 |
| 410 | 已取消 |
| 425 | 还没生成完 |
| 500 | 生成出错 |
| 503 | 队列满 / 模型没加载完 |

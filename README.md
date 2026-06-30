# Sulphur-2 Video Generation Server

类似 llama-server，启动时加载 Sulphur-2 / LTX-2.3 GGUF 到显存，常驻等待请求。
支持任务队列、状态查询、取消排队。

## 关键概念

`Abiray/Sulphur-2-base-GGUF` 只提供 GGUF 权重文件，例如：

```text
sulphur_dev-Q3_K_M.gguf
sulphur_dev-Q4_K_M.gguf
sulphur_dev-Q8_0.gguf
```

它不是完整 diffusers pipeline，缺少 VAE、text encoder、vocoder、scheduler、config 等组件。

因此运行时需要两部分：

| 部分 | 来源 | 作用 |
|------|------|------|
| GGUF 文件 | `Abiray/Sulphur-2-base-GGUF` | 量化 transformer 主体 |
| base pipeline | `diffusers/LTX-2.3-Diffusers` | VAE / text encoder / vocoder / scheduler / config |

联网环境可以自动下载 base pipeline；离线环境必须提前下载并复制到本机。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 联网部署

只需要手动下载 GGUF 文件，base pipeline 会由程序自动从 HuggingFace 下载缓存。

```bash
python server.py \
  --model diffusers/LTX-2.3-Diffusers \
  --gguf ./sulphur_dev-Q3_K_M.gguf \
  --concurrency 1 \
  --queue-size 8
```

## 离线部署

### 1. 在有网机器下载 base pipeline

```bash
huggingface-cli download diffusers/LTX-2.3-Diffusers \
  --local-dir ./LTX-2.3-Diffusers \
  --local-dir-use-symlinks False
```

如果没有 `huggingface-cli`：

```bash
pip install -U huggingface_hub
```

### 2. 下载 GGUF 文件

从这里下载一个 GGUF 文件：

```text
https://huggingface.co/Abiray/Sulphur-2-base-GGUF/tree/main
```

例如：

```text
sulphur_dev-Q3_K_M.gguf
```

### 3. 把两个东西复制到离线机器

目录示例：

```text
sulphur-server/
├── server.py
├── requirements.txt
├── LTX-2.3-Diffusers/
│   ├── model_index.json
│   ├── transformer/
│   ├── vae/
│   ├── text_encoder/
│   ├── vocoder/
│   └── scheduler/
└── sulphur_dev-Q3_K_M.gguf
```

### 4. 离线启动

```bash
python server.py \
  --model ./LTX-2.3-Diffusers \
  --gguf ./sulphur_dev-Q3_K_M.gguf \
  --concurrency 1 \
  --queue-size 8
```

这时不会访问 HuggingFace。

## API

### 提交生成任务

```bash
curl -X POST http://localhost:8080/v1/video/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat walking on a sunny European street", "seed": 42}'
```

返回：

```json
{"task_id": "a1b2c3d4e5f6", "status": "queued", "queue_position": 0}
```

### 查询状态

```bash
curl http://localhost:8080/v1/video/status/a1b2c3d4e5f6
```

状态：`queued` → `processing` → `done` / `failed` / `cancelled`

### 取消排队

```bash
curl -X POST http://localhost:8080/v1/video/cancel/a1b2c3d4e5f6
```

只能取消 `queued` 状态的任务，已开始的无法取消。

### 下载结果

```bash
curl http://localhost:8080/v1/video/result/a1b2c3d4e5f6 -o output.mp4
```

### 健康检查

```bash
curl http://localhost:8080/health
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `diffusers/LTX-2.3-Diffusers` | base pipeline。联网可用 HF ID，离线必须用本地目录 |
| `--gguf` | 无 | GGUF 文件路径，如 `sulphur_dev-Q3_K_M.gguf` |
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8080` | 监听端口 |
| `--concurrency` | `1` | 并发数，单 GPU 保持 1 |
| `--queue-size` | `8` | 最大排队数，超出拒绝（503） |

## 生成参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | string | 必填 | 英文提示词 |
| `negative_prompt` | string | 内置 | 负面提示词 |
| `width` | int | 1216 | 宽度，需被 32 整除 |
| `height` | int | 704 | 高度，需被 32 整除 |
| `num_frames` | int | 121 | 帧数，8k+1（121, 161, 257...） |
| `num_inference_steps` | int | 30 | 推理步数，1-100 |
| `guidance_scale` | float | 3.0 | 引导强度，1-20 |
| `seed` | int | -1 | 随机种子，-1 为随机 |
| `fps` | int | 24 | 输出视频帧率，1-60 |

## GGUF 推荐选择

| 文件 | 大小 | 需要显存 | 推荐场景 |
|------|------|----------|----------|
| `sulphur_dev-Q3_K_S.gguf` | 10.3 GB | ~12 GB | 最省显存 |
| `sulphur_dev-Q3_K_M.gguf` | 11.1 GB | ~13 GB | 省显存 |
| `sulphur_dev-Q4_K_M.gguf` | 14.3 GB | ~16 GB | 推荐甜点 |
| `sulphur_dev-Q5_K_M.gguf` | 16.1 GB | ~18 GB | 质量优先 |
| `sulphur_dev-Q8_0.gguf` | 22.8 GB | ~24 GB | 近无损 |

## 错误码

| HTTP 状态 | 含义 |
|-----------|------|
| 400 | 参数不合法 |
| 404 | 任务不存在 |
| 409 | 取消已开始的任务 |
| 410 | 任务已取消 |
| 425 | 任务未完成就下载 |
| 500 | 生成/编码失败 |
| 503 | 队列满 / 模型未就绪 |

# Sulphur-2 Video Generation Server

类似 llama-server，加载 Sulphur-2 GGUF 到显存，常驻生成视频。
支持任务队列、状态查询、取消排队。

## 磁盘 vs 显存

| | 需要多少 | 说明 |
|---|---|---|
| **硬盘** | ~11 GB GGUF + ~50 GB base pipeline = **~61 GB** | 一次性下载 |
| **显存** | ~11 GB（Q3_K_M GGUF 常驻 GPU） | text_encoder/VAE 放 CPU，不占显存 |

base pipeline 中 ~48.7 GB 是 text_encoder，只做 prompt 编码，放 CPU 运行不进显存。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载 GGUF 文件

从以下地址选一个下载：

```
https://huggingface.co/Abiray/Sulphur-2-base-GGUF/tree/main
```

推荐 `sulphur_dev-Q3_K_M.gguf`（11.1 GB），12 GB 显存可跑。

### 3. 下载 base pipeline（联网）

```bash
python prepare_base.py
```

这会下载 `diffusers/LTX-2.3-Diffusers` 除 transformer 之外的所有组件到 `./LTX-2.3-Diffusers/`（约 50 GB）。

### 4. 启动服务

```bash
python server.py \
  --model ./LTX-2.3-Diffusers \
  --gguf ./sulphur_dev-Q3_K_M.gguf \
  --concurrency 1 \
  --queue-size 8
```

## 离线部署

在有网机器上：

```bash
# 下载 base pipeline（跳过 transformer）
python prepare_base.py --output ./LTX-2.3-Diffusers

# 下载 GGUF 文件
# sulphur_dev-Q3_K_M.gguf
```

复制到离线机器，启动：

```bash
python server.py \
  --model ./LTX-2.3-Diffusers \
  --gguf ./sulphur_dev-Q3_K_M.gguf
```

## API

### 提交生成任务

```bash
curl -X POST http://localhost:8080/v1/video/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat walking on a sunny European street", "seed": 42}'
```

返回 `task_id`：

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

只能取消 `queued` 状态，已开始的无法取消。

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
| `--model` | `diffusers/LTX-2.3-Diffusers` | base pipeline 路径（HF ID 或本地目录） |
| `--gguf` | 无 | GGUF 文件路径 |
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8080` | 监听端口 |
| `--concurrency` | `1` | 并发数，单 GPU 保持 1 |
| `--queue-size` | `8` | 最大排队数 |

## 生成参数（POST body）

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

## GGUF 推荐

| 文件 | 大小 | ~显存 | 场景 |
|------|------|------|------|
| `Q3_K_S` | 10.3 GB | 12 GB | 最省 |
| `Q3_K_M` | 11.1 GB | 12 GB | 平衡 |
| `Q4_K_M` | 14.3 GB | 16 GB | 推荐 |
| `Q5_K_M` | 16.1 GB | 18 GB | 质量 |
| `Q8_0` | 22.8 GB | 24 GB | 近无损 |

## 错误码

| HTTP | 含义 |
|------|------|
| 400 | 参数不合法 |
| 404 | 任务不存在 |
| 409 | 取消已开始的任务 |
| 410 | 任务已取消 |
| 425 | 任务未完成就下载 |
| 500 | 生成/编码失败 |
| 503 | 队列满 / 模型未就绪 |

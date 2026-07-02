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

12 GB 显存优先下载 `sulphur_dev-Q3_K_S.gguf`（10.3 GB），放到项目目录下。

### 4. 预下载 Diffusers 组件（只需一次）

```bash
python prepare_base.py
```

下载到 `./LTX-2.3-Diffusers/`，约 57 GB。这里不会下载 `transformer/`，因为 transformer 由 GGUF 文件替代；只下载 VAE、text encoder、scheduler、tokenizer、processor、connectors、audio_vae、vocoder 等 pipeline 组件。

### 5. 启动

```bash
python server.py --model ./LTX-2.3-Diffusers --gguf ./sulphur_dev-Q3_K_S.gguf
```

看到 `Server ready` 就成功了。

### 6. 打开 Web UI

浏览器访问 `http://localhost:8080/`，可以用表单发请求、看队列、取消任务、下载视频。

> 不要直接双击 `static/index.html`，那样走 `file://` 协议发不了请求。

## 为什么需要两个模型来源

`Abiray/Sulphur-2-base-GGUF` 只提供量化后的 transformer 单文件；Diffusers 目前也不能直接把整个 pipeline 从 GGUF 加载出来。因此这里是混合加载：

- `sulphur_dev-Q3_K_S.gguf`：替代 `transformer/`。
- `diffusers/LTX-2.3-Diffusers` 的非 transformer 组件：提供 VAE、text encoder、scheduler、tokenizer、processor、connectors、audio_vae、vocoder。

`SulphurAI/Sulphur-2-base` 是 Sulphur 的原始权重页面；如果跑完整/ComfyUI 权重，要从那里下载。但本服务走 Diffusers + GGUF：GGUF 已经替代 Sulphur 的 transformer，剩下需要的是 LTX-2.3 pipeline 组件目录，所以预下载脚本使用 `diffusers/LTX-2.3-Diffusers`。

## 资源

| | 需要 | 说明 |
|---|---|---|
| 硬盘 | ~68 GB | Q3_K_S GGUF 10.3 GB + 非 transformer Diffusers 组件约 57 GB |
| 显存 | 12 GB 边缘可试 | Q3_K_S 常驻 GPU，其余组件 CPU offload；分辨率/帧数过高仍可能 OOM |

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
| `--model` | `diffusers/LTX-2.3-Diffusers` | Diffusers pipeline 组件来源；离线时指向 `./LTX-2.3-Diffusers` |
| `--gguf` | 无 | GGUF transformer 文件路径 |
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

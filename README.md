# Sulphur-2 Video Generation Server

类似 llama-server，加载 GGUF 到显存，常驻等待请求生成视频。

## 从头开始

**前提**：Windows，NVIDIA 显卡 ≥12 GB 显存，Python 3.10+。

打开终端（PowerShell 或 bash），一步一步来：

### 1. 装 PyTorch（如果没装）

不需要全局安装 CUDA toolkit，只需 NVIDIA 驱动 + 下面这条命令：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

> **选对 CUDA 版本很关键**：上面是 CUDA 13 版（`cu130`），适配较新的 GPU（含 Blackwell / DGX Spark GB10，架构 sm_121）。老卡（Ampere/Ada，如 A6000）用 `cu126` 也行。装错版本会在加载时报 `CUDA error: no kernel image is available for execution on the device`——表示这份 torch 没有你显卡架构的计算内核，需换匹配的版本。

验证 GPU 可用（同时检查架构是否被支持）：

```bash
python -c "import torch; print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0)); print('cc', torch.cuda.get_device_capability(0)); print('arch_list', torch.cuda.get_arch_list())"
# cuda 必须是 True；cc（计算能力）应能在 arch_list 里找到匹配项
# 例：GB10 的 cc 是 (12, 1)=sm_121，arch_list 里必须有 sm_120/sm_121
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

如果是手动离线下载，最终目录应类似：

```text
sulphur-server/
├─ server.py
├─ prepare_base.py
├─ requirements.txt
├─ sulphur_dev-Q3_K_S.gguf
└─ LTX-2.3-Diffusers/
   ├─ model_index.json
   ├─ vae/
   ├─ text_encoder/
   ├─ audio_vae/
   ├─ vocoder/
   ├─ scheduler/
   ├─ tokenizer/
   ├─ processor/
   └─ connectors/
```

不要放入 `LTX-2.3-Diffusers/transformer/`；它会被 `sulphur_dev-Q3_K_S.gguf` 替代。

### 5. 启动

```bash
python server.py --model ./LTX-2.3-Diffusers --gguf ./sulphur_dev-Q3_K_S.gguf --port 4323
```

看到 `Server ready` 就成功了。服务只使用本地 `./LTX-2.3-Diffusers`，如果目录不存在会直接报错，不会在启动时下载大文件。

### 6. 打开 Web UI

浏览器访问 `http://localhost:8080/`，可以用表单发请求、看队列、取消任务、下载视频。

> 不要直接双击 `static/index.html`，那样走 `file://` 协议发不了请求。

## 为什么需要两个模型来源

`Abiray/Sulphur-2-base-GGUF` 只提供量化后的 transformer 单文件；Diffusers 目前也不能直接把整个 pipeline 从 GGUF 加载出来。因此这里是混合加载：

- `sulphur_dev-Q3_K_S.gguf`：替代 `transformer/`。
- `diffusers/LTX-2.3-Diffusers` 的非 transformer 组件：提供 VAE、text encoder、scheduler、tokenizer、processor、connectors、audio_vae、vocoder。

`SulphurAI/Sulphur-2-base` 是 Sulphur 的原始权重页面；如果跑完整/ComfyUI 权重，要从那里下载。但本服务走 Diffusers + GGUF：GGUF 已经替代 Sulphur 的 transformer，剩下需要的是 LTX-2.3 pipeline 组件目录，所以预下载脚本使用 `diffusers/LTX-2.3-Diffusers`。

## Web UI 参数（档位）

UI 不再让你手填裸参数，而是给出符合 LTX-2.3 约束的档位，避免填出非法值被服务端 400 拒绝：

- **Resolution（分辨率）**：横向 16:9 / 竖向 9:16 / 方形 1:1 三组档位，从最低到 4K。所有档位的宽高都已对齐到 32 像素（LTX-2 硬性要求）。注意官网标称的 1080p=1920×1080、4K=3840×2160 里 1080 和 2160 不能被 32 整除，故这里用等效的 `1920×1088`、`3840×2176`。默认 `1024×576`。
- **Duration + FPS（时长 + 帧率）**：选时长（2/3/5/8/10 秒）和帧率（24/25/48/50），前端自动换算成合法的 `8k+1` 帧数并夹到 9~1281，下方实时显示实际帧数与时长。
- **Quality（质量档）**：草稿/标准/精细/最高，对应 15/20/30/40 推理步。

## 按显卡选启动方式

服务默认按 **12 GB 显卡**配置：transformer（GGUF）常驻 GPU，其余组件 CPU offload，VAE 分块解码。大显存显卡可用 `--no-offload` / `--no-vae-tiling` 提速。

关键账：text_encoder 是 gemma3 12B，bf16 加载约 **24 GB**，且只在开头编码 prompt 时跑一次；VAE 只在结尾 decode 一次。所以 offload 这两者几乎不影响每步速度——每步跑的是常驻的 transformer。是否 `--no-offload`（全常驻）取决于 `transformer + 24GB text_encoder + ~7GB VAE/其他 + 激活` 是否塞得进显存。

| 显卡 | GGUF | 建议启动 | 说明 |
|------|------|----------|------|
| 12 GB | `Q3_K_S`（10.3 GB） | `python server.py --gguf ./sulphur_dev-Q3_K_S.gguf` | 默认全套省显存策略 |
| A6000 48 GB | `Q8_0`（22.8 GB） | `python server.py --gguf ./sulphur_dev-Q8_0.gguf --no-vae-tiling` | 全常驻会 OOM（22.8+24+7+激活 > 48），保持 offload；峰值 ~30 GB |
| A6000 48 GB | `Q4_K_M`（14.3 GB） | `python server.py --gguf ./sulphur_dev-Q4_K_M.gguf --no-offload --no-vae-tiling` | 14.3+24+7≈45 GB，可全常驻、零搬运最快 |
| DGX Spark 128 GB (GB10) | `bf16`（42 GB） | `python server.py --gguf ./sulphur_dev_bf16.gguf --no-offload --no-vae-tiling` | 统一内存架构，最高画质全常驻。需 CUDA 13 版 torch（见步骤 1） |

> **DGX Spark（GB10）注意**：① 是 CPU/GPU 共享的统一内存（128 GB LPDDR5x），offload 到"CPU"没有跨设备搬运意义，直接 `--no-offload` 全常驻最省事。② 架构是 Blackwell sm_121，**必须装 CUDA 13 版 torch**（`cu130`），否则报 `no kernel image`。
>
> A6000 上跑 bf16（42 GB）时 `42+24>48`，text_encoder 塞不下，**不要**加 `--no-offload`。

各档位在 24 fps、5 秒（121 帧）下的默认起点建议：`num_inference_steps=20`，显存够再上 30。如果 OOM，优先降分辨率档位或缩短时长。

参数上限（防止一个请求打爆显存）：`width`/`height` ≤ 3840，`num_frames` ≤ 1281，`prompt` ≤ 2000 字符。超出返回 400。

## 输出文件

生成完成后，服务会把 mp4 写入项目下的 `outputs/` 目录，例如：

```text
outputs/<task_id>.mp4
```

任务状态记录会在内存里保留约 24 小时后自动清理，但 `outputs/` 里的 mp4 文件会保留，方便之后手动整理或删除。

## 资源

| | 需要 | 说明 |
|---|---|---|
| 硬盘 | ~68 GB 起 | GGUF（Q3_K_S 10.3 GB ~ bf16 42 GB）+ 非 transformer Diffusers 组件约 57 GB |
| 显存 | 12 GB 边缘可试 | 默认 transformer 常驻 GPU、其余 CPU offload；大显存可 `--no-offload` 全常驻。分辨率/帧数过高仍可能 OOM |

常见 GGUF 档位大小：Q3_K_S 10.3 GB · Q4_K_M 14.3 GB · Q6_K 17.8 GB · Q8_0 22.8 GB · bf16 42 GB。

## 离线部署

联网机器上做完步骤 3 和 4，把整个目录拷到离线机器，然后直接步骤 2 + 5。

## API 速查

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/video/generate` | 提交任务，返回 task_id |
| GET | `/v1/video/tasks` | 列出当前仍在内存中的任务 |
| GET | `/v1/video/status/{id}` | 查状态 |
| POST | `/v1/video/cancel/{id}` | 取消排队 |
| GET | `/v1/video/result/{id}` | 下载 mp4 |
| GET | `/health` | 服务状态 |

状态：`queued` → `waiting` → `processing` → `done` / `failed` / `cancelled`

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `./LTX-2.3-Diffusers` | 本地 Diffusers 组件目录；必须先用 `prepare_base.py` 预下载好，服务启动时不会隐式下载大文件 |
| `--gguf` | 无 | GGUF transformer 文件路径 |
| `--host` | `0.0.0.0` | |
| `--port` | `8080` | |
| `--queue-size` | `8` | 排满即拒 |
| `--no-offload` | 关（默认 offload） | 关闭 CPU offload，把所有组件常驻 GPU。大显存（A6000/DGX）且用中等量化时更快；显存不足会 OOM |
| `--no-vae-tiling` | 关（默认分块） | 关闭 VAE 分块解码。大显存下让最后一次 decode 走整块，更快 |
| `--compile` | 关 | 对 transformer 做 `torch.compile`（实验性，GGUF 可能失败并自动回退） |

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

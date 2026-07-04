"""
Sulphur-2 Video Generation Server
类似 llama-server，启动时加载模型到显存，常驻等待请求。
支持 text-to-video、任务队列、取消、状态查询。
"""

import gc
import time
import uuid
import asyncio
import logging
import threading
from enum import Enum
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sulphur-server")

# ---- 全局状态 ----
pipe = None
task_queue: "asyncio.Queue[Task]" = None
tasks: dict[str, "Task"] = {}          # task_id -> Task
tasks_lock = threading.RLock()
generation_lock = threading.Lock()      # 单个 pipeline 不并发执行，避免线程安全和显存问题
RESULTS_DIR = Path(__file__).parent / "outputs"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    WAITING = "waiting"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task:
    def __init__(self, task_id: str, params: dict):
        self.id = task_id
        self.params = params
        self.status = TaskStatus.QUEUED
        self.error: str | None = None
        self.result_path: Path | None = None
        self.frame_count: int = 0
        self.current_step: int = 0
        self.total_steps: int = 0
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None


# ---- 请求/响应模型 ----

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000, description="文本提示词，推荐英文")
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted, low resolution",
        max_length=2000,
        description="负面提示词"
    )
    width: int = Field(default=1024, ge=64, le=3840, description="视频宽度，需被32整除")
    height: int = Field(default=576, ge=64, le=3840, description="视频高度，需被32整除")
    num_frames: int = Field(default=121, ge=9, le=1281, description="帧数，需满足 8k+1，如 121, 161, 257")
    num_inference_steps: int = Field(default=20, ge=1, le=100, description="推理步数")
    guidance_scale: float = Field(default=3.0, ge=1.0, le=20.0, description="引导强度")
    seed: int = Field(default=-1, ge=-1, le=2**63 - 1, description="随机种子，-1 为随机")
    fps: int = Field(default=24, ge=1, le=60, description="输出帧率")


class GenerateResponse(BaseModel):
    task_id: str
    status: str
    queue_position: int | None = None


class TaskStatusResponse(BaseModel):
    id: str
    status: str
    queue_position: int | None = None
    current_step: int = 0
    total_steps: int = 0
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    params: dict | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    queue_size: int
    queue_max: int
    waiting: int
    processing: int
    uptime_seconds: float


# ---- 模型加载 ----

def load_pipeline(model_id: str, gguf_path: str | None = None,
                  offload: bool = True, vae_tiling: bool = True, compile: bool = False):
    """
    加载模型到显存，常驻不释放。

    两种模式:
    1. gguf_path 为空 → 标准 diffusers 格式加载 (from_pretrained)
    2. gguf_path 指定 → 从 GGUF 文件加载量化 transformer，其他组件从 model_id 加载

    三个显存/速度开关（默认值面向 12GB 显卡）:
    - offload: 非 transformer 组件是否 CPU offload。大显存 + 中等量化可关闭以提速。
    - vae_tiling: VAE 是否分块解码。大显存可关闭以提速。
    - compile: 是否对 transformer 做 torch.compile（实验性，GGUF 可能失败并回退）。
    """
    if gguf_path:
        return _load_pipeline_gguf(model_id, gguf_path, offload=offload,
                                   vae_tiling=vae_tiling, compile=compile)
    else:
        return _load_pipeline_full(model_id, vae_tiling=vae_tiling, compile=compile)


def _load_pipeline_full(model_id: str, vae_tiling: bool = True, compile: bool = False):
    """标准 diffusers 模型加载"""
    from diffusers import DiffusionPipeline

    logger.info("Loading full model: %s", model_id)
    t0 = time.time()

    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
    except Exception as e:
        logger.error("Failed to load model from '%s': %s", model_id, e)
        raise RuntimeError(f"Model load failed: {e}") from e

    _move_to_cuda(pipe)
    _optimize_pipeline(pipe, vae_tiling=vae_tiling, skip_compile=not compile)

    elapsed = time.time() - t0
    logger.info("Model loaded in %.1fs on %s", elapsed, pipe.device)
    return pipe


def _patch_ltx2_gguf_keymap():
    """
    补齐 diffusers `from_single_file` 对 LTX-2 transformer 缺失的两条 key 重命名。

    Sulphur-2 / LTX-2 官方权重里的调制层叫 `prompt_adaln_single` /
    `audio_prompt_adaln_single`，而 diffusers 模型内部是 `prompt_adaln` /
    `audio_prompt_adaln`。diffusers 的转换脚本和 LoRA 加载都有这个映射，但 GGUF
    走的 single_file 路径（convert_ltx2_transformer_to_diffusers）漏了它们，导致这
    12 个权重成为 unexpected keys、模型对应层留在 meta device，随后 .to(device) 报
    "Cannot copy out of meta tensor"。

    这里包装原转换函数，转换后补做重命名再塞回加载表。若 diffusers 日后修复，
    转换结果里不再有 `*_single` 前缀，本包装自动变为 no-op。
    """
    from diffusers.loaders import single_file_model as sfm

    entry = sfm.SINGLE_FILE_LOADABLE_CLASSES.get("LTX2VideoTransformer3DModel")
    if entry is None:
        logger.warning("diffusers lacks LTX2 single-file support; skipping adaln key patch")
        return

    original_fn = entry["checkpoint_mapping_fn"]
    if getattr(original_fn, "_sulphur_adaln_patched", False):
        return

    def patched_fn(checkpoint, **kwargs):
        converted = original_fn(checkpoint, **kwargs)
        renamed = 0
        for key in list(converted.keys()):
            if key.startswith("audio_prompt_adaln_single."):
                new_key = "audio_prompt_adaln." + key[len("audio_prompt_adaln_single."):]
            elif key.startswith("prompt_adaln_single."):
                new_key = "prompt_adaln." + key[len("prompt_adaln_single."):]
            else:
                continue
            converted[new_key] = converted.pop(key)
            renamed += 1
        if renamed:
            logger.info("Remapped %d LTX-2 adaln_single key(s) for GGUF load", renamed)
        return converted

    patched_fn._sulphur_adaln_patched = True
    entry["checkpoint_mapping_fn"] = patched_fn
    logger.info("Patched diffusers LTX-2 single_file key map (prompt_adaln_single -> prompt_adaln)")


def _load_pipeline_gguf(base_model: str, gguf_path: str, offload: bool = True,
                        vae_tiling: bool = True, compile: bool = False):
    """从 GGUF 文件加载量化 transformer + 基础模型的其他组件"""
    from diffusers import LTX2Pipeline, LTX2VideoTransformer3DModel, GGUFQuantizationConfig

    _patch_ltx2_gguf_keymap()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading GGUF: %s", gguf_path)
    logger.info("Base model: %s", base_model)
    t0 = time.time()

    # 1) 加载 GGUF 量化的 transformer（常驻指定 GPU，不挂 offload hook）
    try:
        transformer = LTX2VideoTransformer3DModel.from_single_file(
            gguf_path,
            config=base_model,
            subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        transformer.to(device)
        logger.info("GGUF transformer loaded on %s", device)
    except Exception as e:
        logger.error("Failed to load GGUF transformer: %s", e)
        raise RuntimeError(f"GGUF load failed: {e}") from e

    # 2) 加载 pipeline。transformer 常驻 GPU。
    #    offload=True（小显存）：其余组件（尤其 ~24GB 的 text_encoder）单独做
    #      sequential CPU offload，留在 CPU、前向时按子模块换入 GPU。不能用整 pipe 的
    #      enable_sequential_cpu_offload —— 那会把常驻的 transformer 也挂上逐子模块
    #      hook，导致 GGUF 量化权重每步反复搬运，又慢又不稳。
    #    offload=False（大显存，如 A6000/DGX）：其余组件也搬上 GPU 常驻，零搬运最快。
    try:
        pipe = LTX2Pipeline.from_pretrained(
            base_model,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        if offload:
            _offload_non_transformer(pipe, device)
        else:
            _move_non_transformer_to_gpu(pipe, device)
    except Exception as e:
        logger.error("Failed to load pipeline with GGUF transformer: %s", e)
        raise RuntimeError(f"Pipeline assembly failed: {e}") from e

    _optimize_pipeline(pipe, vae_tiling=vae_tiling, skip_compile=not compile)

    elapsed = time.time() - t0
    strategy = "others CPU-offloaded" if offload else "all resident on GPU"
    logger.info("GGUF model loaded in %.1fs (transformer on %s, %s)", elapsed, device, strategy)
    return pipe


def _offload_non_transformer(pipe, device):
    """
    对 pipeline 中除 transformer 外的组件做 sequential CPU offload。

    transformer（GGUF 量化）常驻 GPU，不挂 hook；其余 nn.Module 组件
    （text_encoder / vae / audio_vae / vocoder 等）留在 CPU，前向时由
    accelerate 按子模块换入 GPU、用完换出，从而在 12GB 显存上容纳 ~48GB
    的 text_encoder。
    """
    from accelerate import cpu_offload

    offloaded = []
    for name in pipe.components:
        if name == "transformer":
            continue
        component = getattr(pipe, name, None)
        if isinstance(component, torch.nn.Module):
            cpu_offload(component, execution_device=device)
            offloaded.append(name)
    logger.info("Sequential CPU offload on: %s (transformer stays on %s)",
                ", ".join(offloaded), device)


def _move_non_transformer_to_gpu(pipe, device):
    """
    把除 transformer 外的组件也搬上 GPU 常驻（关闭 offload 时用）。

    适用于大显存（A6000 48GB / DGX 128GB）+ 中等量化：transformer 与
    text_encoder 等全部常驻 GPU，前向零搬运，最快。若显存不足会在这里 OOM，
    此时应改用 offload 模式或更高压缩的 GGUF。
    """
    resident = []
    for name in pipe.components:
        if name == "transformer":
            continue
        component = getattr(pipe, name, None)
        if isinstance(component, torch.nn.Module):
            component.to(device)
            resident.append(name)
    logger.info("Moved to %s (resident): %s", device, ", ".join(resident))


def _move_to_cuda(pipe):
    """移动模型到 GPU，处理 OOM"""
    try:
        pipe.to("cuda")
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA OOM — try a smaller quant or --gguf with higher compression")
        raise
    except Exception as e:
        logger.error("Failed to move model to CUDA: %s", e)
        raise


def _optimize_pipeline(pipe, vae_tiling=True, skip_compile=False):
    """开启 VAE tiling 省显存、可选 torch.compile 加速"""
    if vae_tiling and hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
        logger.info("VAE tiling enabled")
    elif not vae_tiling:
        logger.info("VAE tiling disabled (large VRAM mode)")

    if skip_compile:
        logger.info("Skipping torch.compile (GGUF quantized model)")
        return

    try:
        if hasattr(pipe, "transformer"):
            pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
            logger.info("Transformer compiled with torch.compile")
        elif hasattr(pipe, "unet"):
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
            logger.info("UNet compiled with torch.compile")
    except Exception as e:
        logger.warning("torch.compile failed (will run without it): %s", e)


# ---- 队列工作线程 ----

def run_generation(task: Task):
    """在独立线程中执行生成（同步 GPU 操作）"""
    logger.info("[%s] Started: '%s...'", task.id, task.params["prompt"][:60])
    output_path = RESULTS_DIR / f"{task.id}.mp4"
    tmp_path = RESULTS_DIR / f"{task.id}.tmp.mp4"
    generated_frame_count = 0

    try:
        p = task.params
        generator = None
        if p["seed"] >= 0:
            generator = torch.Generator().manual_seed(p["seed"])

        def on_step_end(pipeline, step, timestep, callback_kwargs):
            # 扩散每步结束回调；step 从 0 开始，故 +1 表示已完成步数
            with tasks_lock:
                task.current_step = step + 1
            return callback_kwargs

        call_kwargs = dict(
            prompt=p["prompt"],
            negative_prompt=p["negative_prompt"],
            width=p["width"],
            height=p["height"],
            num_frames=p["num_frames"],
            num_inference_steps=p["num_inference_steps"],
            guidance_scale=p["guidance_scale"],
            generator=generator,
            callback_on_step_end=on_step_end,
        )

        tmp_path.unlink(missing_ok=True)

        with generation_lock:
            with tasks_lock:
                if task.status == TaskStatus.CANCELLED:
                    return
                task.status = TaskStatus.PROCESSING
                task.started_at = time.time()
                task.total_steps = p["num_inference_steps"]

            # LTX2Pipeline 官方接口：返回 (video, audio)
            if pipe.__class__.__name__ == "LTX2Pipeline":
                video, audio = pipe(
                    **call_kwargs,
                    frame_rate=float(p["fps"]),
                    output_type="np",
                    return_dict=False,
                )
                generated_frame_count = frame_count(video)
                write_video_file(tmp_path, video, p["fps"], audio)
                del video, audio
            else:
                result = pipe(**call_kwargs, output_type="pil")
                frames = result.frames[0]
                generated_frame_count = frame_count(frames)
                write_video_file(tmp_path, frames, p["fps"])
                del result, frames

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tmp_path.replace(output_path)
        with tasks_lock:
            task.frame_count = generated_frame_count
            task.result_path = output_path
            task.status = TaskStatus.DONE

    except torch.cuda.OutOfMemoryError:
        logger.error("[%s] CUDA OOM", task.id)
        tmp_path.unlink(missing_ok=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        with tasks_lock:
            task.error = "GPU out of memory. Try reducing width/height/num_frames."
            task.status = TaskStatus.FAILED

    except Exception as e:
        logger.error("[%s] Generation failed: %s", task.id, e)
        tmp_path.unlink(missing_ok=True)
        with tasks_lock:
            task.error = str(e)
            task.status = TaskStatus.FAILED

    finally:
        with tasks_lock:
            task.finished_at = time.time()
            elapsed = task.finished_at - task.started_at if task.started_at else 0
            status = task.status.value.upper()
            frames = task.frame_count
        logger.info(
            "[%s] %s in %.1fs (%d frames)",
            task.id, status, elapsed, frames
        )


async def queue_worker(worker_id: int):
    """后台循环：从队列取任务，在线程池执行"""
    loop = asyncio.get_running_loop()
    while True:
        task = await task_queue.get()
        with tasks_lock:
            if task.status == TaskStatus.CANCELLED:
                task_queue.task_done()
                continue
            task.status = TaskStatus.WAITING

        try:
            await loop.run_in_executor(None, run_generation, task)
        finally:
            task_queue.task_done()


async def cleanup_expired_tasks():
    """定期清理已完成/失败/取消的过期任务；输出文件保留在 outputs/"""
    TTL = 24 * 60 * 60  # 24 小时后从内存中清除
    while True:
        await asyncio.sleep(60)
        now = time.time()
        removed = 0
        with tasks_lock:
            expired = [
                tid for tid, t in tasks.items()
                if t.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED)
                and t.finished_at
                and (now - t.finished_at > TTL)
            ]
            for tid in expired:
                del tasks[tid]
                removed += 1
        if removed:
            logger.info("Cleaned up %d expired task(s)", removed)


# ---- FastAPI 生命周期 ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe, task_queue
    model_id = app.state.model_id
    gguf_path = app.state.gguf_path
    RESULTS_DIR.mkdir(exist_ok=True)

    try:
        pipe = load_pipeline(
            model_id, gguf_path,
            offload=app.state.offload,
            vae_tiling=app.state.vae_tiling,
            compile=app.state.compile,
        )
    except Exception:
        logger.critical("Failed to load model — server cannot start")
        raise

    task_queue = asyncio.Queue(maxsize=app.state.queue_max_size)
    worker = asyncio.create_task(queue_worker(0))
    cleanup_task = asyncio.create_task(cleanup_expired_tasks())

    app.state.start_time = time.time()
    logger.info("Server ready — queue max=%d", app.state.queue_max_size)
    yield

    # shutdown
    logger.info("Shutting down...")
    cleanup_task.cancel()
    worker.cancel()
    # 等正在执行的生成结束（run_generation 在 executor 线程里，cancel 停不了它）
    # 再删 pipe，避免生成中途模型被销毁。
    with generation_lock:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logger.info("Shutdown complete")


app = FastAPI(title="Sulphur-2 Video Server", version="1.0", lifespan=lifespan)

# 允许任意来源跨域调用 API（本服务无凭证鉴权，故 allow_credentials=False）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 静态文件 / UI ----

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# ---- 端点 ----

@app.get("/health", response_model=HealthResponse)
async def health():
    with tasks_lock:
        waiting = sum(1 for t in tasks.values() if t.status == TaskStatus.WAITING)
        processing = sum(1 for t in tasks.values() if t.status == TaskStatus.PROCESSING)
    return HealthResponse(
        status="ok",
        model_loaded=pipe is not None,
        device=str(pipe.device) if pipe else "N/A",
        queue_size=task_queue.qsize() if task_queue else 0,
        queue_max=app.state.queue_max_size,
        waiting=waiting,
        processing=processing,
        uptime_seconds=round(time.time() - app.state.start_time, 1),
    )


@app.post("/v1/video/generate", response_model=GenerateResponse)
async def submit_generation(req: GenerateRequest):
    """提交生成任务，立即返回 task_id"""
    if pipe is None:
        raise HTTPException(503, "Model not loaded yet")

    # 参数校验
    errors = []
    if req.width % 32 != 0 or req.height % 32 != 0:
        errors.append("width 和 height 必须能被 32 整除")
    if (req.num_frames - 1) % 8 != 0:
        errors.append("num_frames 必须满足 8k+1，如 121, 161, 257")
    if errors:
        raise HTTPException(400, "; ".join(errors))

    task_id = uuid.uuid4().hex[:12]
    params = req.model_dump()
    task = Task(task_id, params)

    with tasks_lock:
        tasks[task_id] = task

    try:
        task_queue.put_nowait(task)
    except asyncio.QueueFull:
        with tasks_lock:
            tasks.pop(task_id, None)
        raise HTTPException(
            503,
            f"Queue full ({app.state.queue_max_size} max). Try again later or increase --queue-size."
        )

    return GenerateResponse(
        task_id=task_id,
        status="queued",
        queue_position=_queue_position(task_id),
    )


@app.get("/v1/video/tasks", response_model=list[TaskStatusResponse])
async def list_tasks():
    """列出当前仍保留在内存中的任务"""
    with tasks_lock:
        snapshot = list(tasks.values())
    snapshot.sort(key=lambda t: t.created_at, reverse=True)
    return [task_response(t) for t in snapshot]


@app.get("/v1/video/status/{task_id}", response_model=TaskStatusResponse)
async def task_status(task_id: str):
    """查询任务状态"""
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "Task not found")
    return task_response(task)


@app.post("/v1/video/cancel/{task_id}")
async def cancel_task(task_id: str):
    """取消排队中的任务（已开始的无法取消）"""
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "Task not found")
        if task.status == TaskStatus.CANCELLED:
            return {"task_id": task_id, "status": "cancelled", "message": "Already cancelled"}
        if task.status not in (TaskStatus.QUEUED, TaskStatus.WAITING):
            raise HTTPException(409, f"Cannot cancel task in '{task.status.value}' status")

        task.status = TaskStatus.CANCELLED
        task.finished_at = time.time()
    logger.info("[%s] Cancelled while queued", task_id)
    return {"task_id": task_id, "status": "cancelled"}


@app.get("/v1/video/result/{task_id}")
async def get_result(task_id: str):
    """下载生成的视频"""
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "Task not found")
        status = task.status
        error = task.error
        result_path = task.result_path

    if status in (TaskStatus.QUEUED, TaskStatus.WAITING, TaskStatus.PROCESSING):
        raise HTTPException(425, f"Task not ready yet (status: {status.value})")
    if status == TaskStatus.FAILED:
        raise HTTPException(500, error or "Generation failed")
    if status == TaskStatus.CANCELLED:
        raise HTTPException(410, "Task was cancelled")

    if result_path is None or not result_path.exists():
        raise HTTPException(404, "Result file not found")

    return FileResponse(
        result_path,
        media_type="video/mp4",
        filename=f"{task_id}.mp4",
    )


# ---- 辅助函数 ----

def task_response(task: Task) -> TaskStatusResponse:
    with tasks_lock:
        return TaskStatusResponse(
            id=task.id,
            status=task.status.value,
            queue_position=_queue_position(task.id),
            current_step=task.current_step,
            total_steps=task.total_steps,
            created_at=task.created_at,
            started_at=task.started_at,
            finished_at=task.finished_at,
            params=task.params,
            error=task.error,
        )


def _queue_position(task_id: str) -> int | None:
    """计算排队位置（粗糙：按创建时间排序 queued 任务）"""
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None or task.status != TaskStatus.QUEUED:
            return None
        queued = [t for t in tasks.values() if t.status == TaskStatus.QUEUED]
        queued.sort(key=lambda t: t.created_at)
        for i, t in enumerate(queued):
            if t.id == task_id:
                return i + 1
    return None


def frame_count(frames) -> int:
    if isinstance(frames, np.ndarray):
        return int(frames.shape[1] if frames.ndim == 5 else frames.shape[0])
    return len(frames or [])


def write_video_file(output_path: Path, frames, fps: int, audio=None):
    """把生成结果写入 outputs/*.mp4，避免完成后的任务长期持有全部帧。"""
    import imageio

    # LTX2Pipeline: frames 通常是 numpy，形状 [B, F, H, W, C]，值域 0-1
    if isinstance(frames, np.ndarray):
        video = (frames * 255).round().clip(0, 255).astype("uint8")
        video = video[0] if video.ndim == 5 else video

        # 有音频时优先用 LTX2 官方 encode_video
        if audio is not None and hasattr(pipe, "vocoder"):
            from diffusers.pipelines.ltx2.export_utils import encode_video

            audio_tensor = audio[0].float().cpu() if hasattr(audio[0], "float") else torch.from_numpy(audio[0]).float()
            encode_video(
                torch.from_numpy(video),
                fps=float(fps),
                audio=audio_tensor,
                audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
                output_path=str(output_path),
            )
            return

        writer = imageio.get_writer(str(output_path), fps=fps)
        try:
            for frame in video:
                writer.append_data(frame)
        finally:
            writer.close()
        return

    # 旧 pipeline: PIL frames
    writer = imageio.get_writer(str(output_path), fps=fps)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


# ---- 入口 ----

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Sulphur-2 Video Generation Server")
    parser.add_argument("--model", default="./LTX-2.3-Diffusers",
                        help="本地 Diffusers 组件目录（由 prepare_base.py 预下载），默认 ./LTX-2.3-Diffusers")
    parser.add_argument("--gguf", default=None,
                        help="GGUF 量化模型文件路径，如 sulphur_dev-Q3_K_S.gguf")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    parser.add_argument("--queue-size", type=int, default=8,
                        help="最大排队数，超出拒绝新请求（默认 8）")
    parser.add_argument("--no-offload", action="store_true",
                        help="关闭 CPU offload，把所有组件常驻 GPU（大显存如 A6000/DGX 用，更快；显存不足会 OOM）")
    parser.add_argument("--no-vae-tiling", action="store_true",
                        help="关闭 VAE 分块解码（大显存用，更快）")
    parser.add_argument("--compile", action="store_true",
                        help="对 transformer 做 torch.compile（实验性，GGUF 可能失败并自动回退）")
    args = parser.parse_args()

    if args.queue_size < 1:
        parser.error("--queue-size must be >= 1")

    model_path = Path(args.model)
    if not model_path.exists():
        parser.error(f"Model directory not found: {args.model}. Run prepare_base.py first.")
    if not (model_path / "model_index.json").exists():
        parser.error(f"model_index.json not found under: {args.model}")

    if args.gguf and not Path(args.gguf).exists():
        parser.error(f"GGUF file not found: {args.gguf}")

    app.state.model_id = args.model
    app.state.gguf_path = args.gguf
    app.state.queue_max_size = args.queue_size
    app.state.offload = not args.no_offload
    app.state.vae_tiling = not args.no_vae_tiling
    app.state.compile = args.compile

    mode = f"GGUF: {args.gguf}" if args.gguf else f"full: {args.model}"
    mem = "resident" if args.no_offload else "offload"
    logger.info("Starting server on %s:%d | %s | queue=%d | mem=%s | vae_tiling=%s | compile=%s",
                args.host, args.port, mode, args.queue_size, mem,
                not args.no_vae_tiling, args.compile)
    uvicorn.run(app, host=args.host, port=args.port)

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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sulphur-server")

# ---- 全局状态 ----
pipe = None
task_queue: "asyncio.Queue[Task]" = None
tasks: dict[str, "Task"] = {}          # task_id -> Task
tasks_lock = threading.Lock()
RESULTS_DIR = Path(__file__).parent / "outputs"


class TaskStatus(str, Enum):
    QUEUED = "queued"
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
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None


# ---- 请求/响应模型 ----

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="文本提示词，推荐英文")
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted, low resolution",
        description="负面提示词"
    )
    width: int = Field(default=1024, ge=64, description="视频宽度，需被32整除")
    height: int = Field(default=576, ge=64, description="视频高度，需被32整除")
    num_frames: int = Field(default=121, ge=9, description="帧数，需满足 8k+1，如 121, 161, 257")
    num_inference_steps: int = Field(default=20, ge=1, le=100, description="推理步数")
    guidance_scale: float = Field(default=3.0, ge=1.0, le=20.0, description="引导强度")
    seed: int = Field(default=-1, description="随机种子，-1 为随机")
    fps: int = Field(default=24, ge=1, le=60, description="输出帧率")


class GenerateResponse(BaseModel):
    task_id: str
    status: str
    queue_position: int | None = None


class TaskStatusResponse(BaseModel):
    id: str
    status: str
    queue_position: int | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    params: dict | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    concurrency: int
    queue_size: int
    queue_max: int
    processing: int
    uptime_seconds: float


# ---- 模型加载 ----

def load_pipeline(model_id: str, gguf_path: str | None = None):
    """
    加载模型到显存，常驻不释放。

    两种模式:
    1. gguf_path 为空 → 标准 diffusers 格式加载 (from_pretrained)
    2. gguf_path 指定 → 从 GGUF 文件加载量化 transformer，其他组件从 model_id 加载
    """
    if gguf_path:
        return _load_pipeline_gguf(model_id, gguf_path)
    else:
        return _load_pipeline_full(model_id)


def _load_pipeline_full(model_id: str):
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
    _optimize_pipeline(pipe)

    elapsed = time.time() - t0
    logger.info("Model loaded in %.1fs on %s", elapsed, pipe.device)
    return pipe


def _load_pipeline_gguf(base_model: str, gguf_path: str):
    """从 GGUF 文件加载量化 transformer + 基础模型的其他组件"""
    from diffusers import LTX2Pipeline, LTX2VideoTransformer3DModel, GGUFQuantizationConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading GGUF: %s", gguf_path)
    logger.info("Base model: %s", base_model)
    t0 = time.time()

    # 1) 加载 GGUF 量化的 transformer（放到指定 GPU）
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

    # 2) 加载 pipeline，其他组件 CPU offload 省显存
    try:
        pipe = LTX2Pipeline.from_pretrained(
            base_model,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_sequential_cpu_offload(device=device)
    except Exception as e:
        logger.error("Failed to load pipeline with GGUF transformer: %s", e)
        raise RuntimeError(f"Pipeline assembly failed: {e}") from e

    _optimize_pipeline(pipe, skip_compile=True)  # GGUF 量化模型不做 torch.compile

    elapsed = time.time() - t0
    logger.info("GGUF model loaded in %.1fs (transformer on %s, others CPU-offloaded)", elapsed, device)
    return pipe


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


def _optimize_pipeline(pipe, skip_compile=False):
    """开启 VAE tiling 省显存、可选 torch.compile 加速"""
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
        logger.info("VAE tiling enabled")

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

        call_kwargs = dict(
            prompt=p["prompt"],
            negative_prompt=p["negative_prompt"],
            width=p["width"],
            height=p["height"],
            num_frames=p["num_frames"],
            num_inference_steps=p["num_inference_steps"],
            guidance_scale=p["guidance_scale"],
            generator=generator,
        )

        tmp_path.unlink(missing_ok=True)

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
            task.status = TaskStatus.PROCESSING
            task.started_at = time.time()

        await loop.run_in_executor(None, run_generation, task)
        task_queue.task_done()


async def cleanup_expired_tasks():
    """定期清理已完成/失败/取消的过期任务；输出文件保留在 outputs/"""
    TTL = 600  # 10 分钟后从内存中清除
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
        pipe = load_pipeline(model_id, gguf_path)
    except Exception:
        logger.critical("Failed to load model — server cannot start")
        raise

    task_queue = asyncio.Queue(maxsize=app.state.queue_max_size)
    workers = [
        asyncio.create_task(queue_worker(i))
        for i in range(app.state.concurrency)
    ]
    cleanup_task = asyncio.create_task(cleanup_expired_tasks())
    app.state.workers = workers

    app.state.start_time = time.time()
    logger.info(
        "Server ready — concurrency=%d, queue max=%d",
        app.state.concurrency, app.state.queue_max_size
    )
    yield

    # shutdown
    logger.info("Shutting down...")
    cleanup_task.cancel()
    for w in workers:
        w.cancel()
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Shutdown complete")


app = FastAPI(title="Sulphur-2 Video Server", version="1.0", lifespan=lifespan)

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
        processing = sum(1 for t in tasks.values() if t.status == TaskStatus.PROCESSING)
    return HealthResponse(
        status="ok",
        model_loaded=pipe is not None,
        device=str(pipe.device) if pipe else "N/A",
        concurrency=app.state.concurrency,
        queue_size=task_queue.qsize() if task_queue else 0,
        queue_max=app.state.queue_max_size,
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


@app.get("/v1/video/status/{task_id}", response_model=TaskStatusResponse)
async def task_status(task_id: str):
    """查询任务状态"""
    queue_position = _queue_position(task_id)
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "Task not found")
        return TaskStatusResponse(
            id=task.id,
            status=task.status.value,
            queue_position=queue_position,
            created_at=task.created_at,
            started_at=task.started_at,
            finished_at=task.finished_at,
            params=task.params,
            error=task.error,
        )


@app.post("/v1/video/cancel/{task_id}")
async def cancel_task(task_id: str):
    """取消排队中的任务（已开始的无法取消）"""
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "Task not found")
        if task.status == TaskStatus.CANCELLED:
            return {"task_id": task_id, "status": "cancelled", "message": "Already cancelled"}
        if task.status != TaskStatus.QUEUED:
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
    if task.status == TaskStatus.QUEUED or task.status == TaskStatus.PROCESSING:
        raise HTTPException(425, f"Task not ready yet (status: {task.status.value})")
    if task.status == TaskStatus.FAILED:
        raise HTTPException(500, task.error or "Generation failed")
    if task.status == TaskStatus.CANCELLED:
        raise HTTPException(410, "Task was cancelled")

    if task.result_path is None or not task.result_path.exists():
        raise HTTPException(404, "Result file not found")

    return FileResponse(
        task.result_path,
        media_type="video/mp4",
        filename=f"{task_id}.mp4",
    )


# ---- 辅助函数 ----

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
                return i
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

        writer = imageio.get_writer(str(output_path), format="mp4", fps=fps)
        for frame in video:
            writer.append_data(frame)
        writer.close()
        return

    # 旧 pipeline: PIL frames
    writer = imageio.get_writer(str(output_path), format="mp4", fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


# ---- 入口 ----

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Sulphur-2 Video Generation Server")
    parser.add_argument("--model", default="diffusers/LTX-2.3-Diffusers",
                        help="基础模型（提供 VAE / text encoder / scheduler），默认 LTX-2.3 diffusers 格式")
    parser.add_argument("--gguf", default=None,
                        help="GGUF 量化模型文件路径，如 sulphur_dev-Q3_K_M.gguf")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="并发 worker 数，单 GPU 保持 1，多 GPU 可调大（默认 1）")
    parser.add_argument("--queue-size", type=int, default=8,
                        help="最大排队数，超出拒绝新请求（默认 8）")
    args = parser.parse_args()

    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.queue_size < 1:
        parser.error("--queue-size must be >= 1")

    if args.gguf and not Path(args.gguf).exists():
        parser.error(f"GGUF file not found: {args.gguf}")

    app.state.model_id = args.model
    app.state.gguf_path = args.gguf
    app.state.concurrency = args.concurrency
    app.state.queue_max_size = args.queue_size

    mode = f"GGUF: {args.gguf}" if args.gguf else f"full: {args.model}"
    logger.info("Starting server on %s:%d | %s | concurrency=%d, queue=%d",
                args.host, args.port, mode, args.concurrency, args.queue_size)
    uvicorn.run(app, host=args.host, port=args.port)

"""
准备离线 Diffusers 组件：仅下载 diffusers/LTX-2.3-Diffusers 中非 transformer 的组件。

Sulphur-2-base-GGUF 只提供量化后的 transformer；VAE、text encoder、scheduler、
tokenizer、connectors、audio_vae、vocoder 等仍需来自 LTX-2.3 Diffusers pipeline。
transformer（~38 GB）由 GGUF 文件替代，不需要下载。

用法:
    # 联网下载到当前目录
    python prepare_base.py

    # 指定输出目录
    python prepare_base.py --output ./LTX-2.3-Diffusers

下载量约 57 GB（text_encoder ~48.7 GB，connectors ~6.3 GB），只需执行一次。
"""

import argparse
import sys
from pathlib import Path

COMPONENTS = [
    "model_index.json",
    "vae/*",
    "text_encoder/*",
    "audio_vae/*",
    "vocoder/*",
    "scheduler/*",
    "tokenizer/*",
    "processor/*",
    "connectors/*",
]

REPO_ID = "diffusers/LTX-2.3-Diffusers"


def download_base(output_dir: str):
    from huggingface_hub import snapshot_download

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading base pipeline from {REPO_ID}")
    print(f"Output directory: {out.resolve()}")
    print(f"Components: {', '.join(c.replace('/*', '/') for c in COMPONENTS if c != 'model_index.json')}")
    print()
    print("Download size: ~57 GB (text_encoder ~48.7 GB, connectors ~6.3 GB)")
    print("Skipping transformer/: replaced by your GGUF file")
    print()

    snapshot_download(
        REPO_ID,
        local_dir=str(out),
        allow_patterns=COMPONENTS,
        local_dir_use_symlinks=False,
    )

    print()
    print("Done! Use --model to point to this directory:")
    print(f"  python server.py --model {out.resolve()} --gguf <your.gguf>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LTX-2.3 Diffusers components for Sulphur GGUF (no transformer)")
    parser.add_argument("--output", default="./LTX-2.3-Diffusers",
                        help="输出目录，默认 ./LTX-2.3-Diffusers")
    args = parser.parse_args()

    try:
        import huggingface_hub
    except ImportError:
        print("请先安装 huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    download_base(args.output)

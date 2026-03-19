# Minimal Colab Repro

## Files

- `vlm.py`
- `methods/vlm_planner.py`
- `demo_vlm_runner.py`
- `low_level_controller/low_level_controller.py`
- `dataset/safe_detailed_1009.jsonl`
- `requirements.txt`

## Install

```bash
pip install -r requirements.txt pillow scipy ai2thor_colab
```

## Colab Setup

```python
import os
import ai2thor_colab

os.environ["VLM_API_SECRET_KEY"] = "your-key"
os.environ["VLM_BASE_URL"] = "https://api.360.cn/v1/chat/completions"
os.environ["VLM_MODEL"] = "alibaba/qwen3-vl-plus"

ai2thor_colab.start_xserver()
```

## Run Minimal Closed Loop

```bash
python demo_vlm_runner.py --scene FloorPlan407 --task "Open the Cabinet." --max-steps 2
```

## Run One Real Dataset Sample

The default sample below is `dataset/safe_detailed_1009.jsonl` line 1.

```bash
python demo_vlm_runner.py --dataset dataset/safe_detailed_1009.jsonl --sample-index 1 --max-steps 2
```

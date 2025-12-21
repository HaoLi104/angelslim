# SpecExit 推理评测指南

## 📋 准备工作

### 1. 确认训练输出

训练完成后，checkpoint 保存在：
```
/data/ocean/specexit_workspace/models/drafter_qwen3-8b-specexit/
```

需要确认最终的 checkpoint 目录。通常有两种情况：

**情况 A：使用最终保存的模型**
- 如果训练脚本在最后保存了最终模型，可能在 `output_dir` 下有 `pytorch_model.bin` 和 `config.json`
- 路径：`/data/ocean/specexit_workspace/models/drafter_qwen3-8b-specexit/`

**情况 B：使用某个 checkpoint**
- checkpoint 目录：`/data/ocean/specexit_workspace/models/drafter_qwen3-8b-specexit/checkpoint-XXXX/`
- 通常使用最后一个 checkpoint（数字最大的）

检查命令：
```bash
# 列出所有 checkpoint
ls -lh /data/ocean/specexit_workspace/models/drafter_qwen3-8b-specexit/

# 检查某个 checkpoint 是否包含必要文件
ls -lh /data/ocean/specexit_workspace/models/drafter_qwen3-8b-specexit/checkpoint-XXXX/
# 应该看到：config.json, pytorch_model.bin 或 model.safetensors
```

### 2. 确认配置文件

确保 checkpoint 目录中有 `config.json`，并且包含 `early_stop_method: "confidence_progress_remain"`。

如果 checkpoint 中没有 config.json，需要从训练配置目录复制：
```bash
cp angelslim/compressor/speculative/train/configs/qwen3-8b-eagle3.json \
   /data/ocean/specexit_workspace/models/drafter_qwen3-8b-specexit/config.json
```

##  运行评测

### 方法 1：使用 run_specexit.py（推荐）

在远端执行：

```bash
cd /data/ocean/specexit_workspace/angelslim

# 设置环境变量（根据实际情况调整）
export BASE_MODEL_PATH=/data/ocean/specexit_workspace/models/base_qwen3-8b
export EAGLE_MODEL_PATH=/data/ocean/specexit_workspace/models/drafter_qwen3-8b-specexit  # 或 checkpoint-XXXX

# 运行评测（使用 gsm8k 基准测试）
python3 tools/run_specexit.py \
    --base-model-path "$BASE_MODEL_PATH" \
    --eagle-model-path "$EAGLE_MODEL_PATH" \
    --model-id qwen3-8b-specexit \
    --bench-name gsm8k \
    --output-dir ./specexit_outputs \
    --experiments baseline,speculative,specexit \
    --specexit-stop-method confidence_progress_remain \
    --temperature 1.0 \
    --total-token 60 \
    --depth 5 \
    --top-k 10 \
    --max-new-token 1024 \
    --num-gpus-per-model 1 \
    --num-gpus-total 2
```

### 方法 2：使用脚本（简化版）

修改 `scripts/speculative/run_with_specexit.sh` 中的路径，然后运行：

```bash
cd /data/ocean/specexit_workspace/angelslim

# 设置环境变量
export BASE_MODEL_PATH=/data/ocean/specexit_workspace/models/base_qwen3-8b
export EAGLE_MODEL_PATH=/data/ocean/specexit_workspace/models/drafter_qwen3-8b-specexit
export BENCH_NAME=gsm8k  # 可选：gsm8k, mt_bench, alpaca, humaneval
export OUTPUT_DIR=./specexit_outputs
export EARLY_STOP_METHOD=confidence_progress_remain

# 运行脚本
bash scripts/speculative/run_with_specexit.sh
```

### 方法 3：在 tmux 中后台运行

```bash
# 创建 tmux session
tmux new -s specexit_eval

# 运行评测命令
cd /data/ocean/specexit_workspace/angelslim
export BASE_MODEL_PATH=/data/ocean/specexit_workspace/models/base_qwen3-8b
export EAGLE_MODEL_PATH=/data/ocean/specexit_workspace/models/drafter_qwen3-8b-specexit

python3 tools/run_specexit.py \
    --base-model-path "$BASE_MODEL_PATH" \
    --eagle-model-path "$EAGLE_MODEL_PATH" \
    --model-id qwen3-8b-specexit \
    --bench-name gsm8k \
    --output-dir ./specexit_outputs \
    --experiments baseline,speculative,specexit \
    --specexit-stop-method confidence_progress_remain \
    --temperature 1.0 \
    --total-token 60 \
    --depth 5 \
    --top-k 10 \
    --max-new-token 1024 \
    --num-gpus-per-model 1 \
    --num-gpus-total 2

# Detach: Ctrl+B, 然后按 D
# 重新连接: tmux attach -t specexit_eval
```

## 📊 参数说明

### 核心参数

- `--base-model-path`: 目标模型路径（Qwen3-8B）
- `--eagle-model-path`: Drafter 模型路径（新训练的）
- `--model-id`: 模型标识符（用于结果文件命名）
- `--bench-name`: 基准测试名称
  - `gsm8k`: 数学问题（推荐，测试集较大）
  - `mt_bench`: 多轮对话
  - `alpaca`: 指令跟随
  - `humaneval`: 代码生成

### SpecExit 相关参数

- `--specexit-stop-method`: 早退方法
  - `confidence_progress_remain`: 使用3个side head信号（推荐）
  - `confidence`: 仅使用置信度
  - `progress`: 仅使用进度
  - `remain`: 仅使用剩余步长

- `--experiments`: 要运行的实验
  - `baseline`: 基线（纯目标模型）
  - `speculative`: 投机解码（无早退）
  - `specexit`: SpecExit（有早退信号）
  - 建议：`baseline,speculative,specexit` 以便对比

### 推理参数

- `--temperature`: 采样温度（默认 1.0）
- `--total-token`: 最大 draft token 数（默认 60）
- `--depth`: 树深度（默认 5）
- `--top-k`: 候选分支数（默认 10）
- `--max-new-token`: 最大生成 token 数（默认 1024）

### GPU 配置

- `--num-gpus-per-model`: 每个模型使用的 GPU 数（通常 1）
- `--num-gpus-total`: 总共使用的 GPU 数（根据可用 GPU 调整）

## 📈 结果解读

评测完成后，会在 `--output-dir` 目录下生成：

```
specexit_outputs/
├── baseline/
│   └── baseline.jsonl          # 基线结果
├── speculative/
│   └── eagle.jsonl              # 投机解码结果
├── specexit/
│   └── eagle.jsonl              # SpecExit 结果
└── specexit_summary.json        # 汇总结果
```

### 关键指标

- **平均新Token**: 每个样本平均生成的 token 数
- **平均耗时**: 每个样本的平均推理时间（秒）
- **吞吐(token/s)**: 每秒生成的 token 数（**越高越好**）
- **接受长度**: 平均接受的 draft token 数（投机解码相关）
- **相对Baseline加速**: 相比基线的加速比（**越高越好**）

### 预期结果

根据训练质量，预期：
- SpecExit 相比 baseline 应该达到 **2-3x 加速**
- 相比纯投机解码，SpecExit 应该通过早退获得额外收益
- 接受长度应该合理（通常在 2-4 之间）

## ⚠️ 常见问题

### 1. 找不到 config.json

如果 checkpoint 中没有 config.json，从训练配置复制：
```bash
cp angelslim/compressor/speculative/train/configs/qwen3-8b-eagle3.json \
   $EAGLE_MODEL_PATH/config.json
```

### 2. 权重形状不匹配

如果出现 "shape mismatch" 错误，检查：
- config.json 中 `early_stop_method` 是否为 `confidence_progress_remain`
- `fc.weight` 的形状应该为 `(hidden_size+3, hidden_size*3)`

### 3. 显存不足

- 减小 `--num-gpus-per-model` 或 `--num-gpus-total`
- 减小 `--total-token` 或 `--depth`
- 使用更少的 GPU

### 4. 早退未生效

如果日志中看到 "drafter 权重缺少 SpecExit side head"，说明：
- checkpoint 的 fc 层输出维度不对
- 需要检查训练是否正确保存了 side head 权重

##  快速开始命令（复制即用）

```bash
cd /data/ocean/specexit_workspace/angelslim

# 确认 checkpoint 路径（替换为实际路径）
EAGLE_MODEL_PATH=/data/ocean/specexit_workspace/models/drafter_qwen3-8b-specexit  # 或 checkpoint-XXXX

# 如果 checkpoint 中没有 config.json，先复制
if [ ! -f "$EAGLE_MODEL_PATH/config.json" ]; then
    cp angelslim/compressor/speculative/train/configs/qwen3-8b-eagle3.json \
       "$EAGLE_MODEL_PATH/config.json"
fi

# 运行评测
python3 tools/run_specexit.py \
    --base-model-path /data/ocean/specexit_workspace/models/base_qwen3-8b \
    --eagle-model-path "$EAGLE_MODEL_PATH" \
    --model-id qwen3-8b-specexit \
    --bench-name gsm8k \
    --output-dir ./specexit_outputs \
    --experiments baseline,speculative,specexit \
    --specexit-stop-method confidence_progress_remain \
    --temperature 1.0 \
    --total-token 60 \
    --depth 5 \
    --top-k 10 \
    --max-new-token 1024 \
    --num-gpus-per-model 1 \
    --num-gpus-total 2
```


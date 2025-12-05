import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from angelslim.engine import SpecEngine


@dataclass
class ExperimentPlan:
	"""用于描述单个实验配置的简单数据结构"""

	name: str
	display_name: str
	mode: str  # baseline 或 eagle
	description: str
	early_stop_method: Optional[str]
	needs_baseline: bool
	config_overrides: Optional[Dict[str, object]] = None

	@property
	def is_baseline(self) -> bool:
		return self.mode == "baseline"


def setup_seed(seed: int) -> None:
	"""设置随机种子，确保多次运行可复现"""

	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
	"""解析命令行参数"""

	parser = argparse.ArgumentParser(
		description="融合投机解码、DeepConf 与 SpecExit 的快速对比脚本",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)

	parser.add_argument("--base-model-path", type=str, required=True)
	parser.add_argument("--eagle-model-path", type=str, required=True)
	parser.add_argument("--model-id", type=str, required=True)
	parser.add_argument("--bench-name", type=str, default="gsm8k")
	parser.add_argument("--output-dir", type=str, help="结果输出目录，未提供则自动创建")

	parser.add_argument(
		"--deploy-backend",
		type=str,
		choices=["pytorch"],
		default="pytorch",
		help="当前实现仅支持 PyTorch 推理后端",
	)

	parser.add_argument(
		"--experiments",
		type=str,
		default="baseline,speculative,specexit",
		help="逗号分隔的实验列表，可选：baseline/speculative/specexit/specexit_ppl",
	)
	parser.add_argument(
		"--specexit-stop-method",
		type=str,
		default="confidence_progress_remain",
		help="SpecExit 早退开关，可按 need 调整例如 confidence 或 confidence_progress",
	)
	parser.add_argument(
		"--use-confusion-prune",
		action="store_true",
		help="开启基于区间困惑度的 DeepConf 剪枝（与 side head 信号互补）",
	)
	parser.add_argument("--confusion-window", type=int, default=16)
	parser.add_argument("--confusion-threshold", type=float, default=25.0)
	parser.add_argument(
		"--baseline-file",
		type=str,
		help="复用已有 baseline 结果，jsonl 文件路径，省去再次运行",
	)
	parser.add_argument(
		"--summary-file",
		type=str,
		help="额外写出融合实验汇总的 json 文件路径",
	)

	parser.add_argument("--num-choices", type=int, default=1)
	parser.add_argument("--temperature", type=float, default=1.0)
	parser.add_argument("--max-new-token", type=int, default=1024)
	parser.add_argument("--total-token", type=int, default=60)
	parser.add_argument("--depth", type=int, default=5)
	parser.add_argument("--top-k", type=int, default=10)
	parser.add_argument("--top-p", type=float, default=1.0)

	parser.add_argument("--num-gpus-per-model", type=int, default=1)
	parser.add_argument("--num-gpus-total", type=int, default=1)
	parser.add_argument("--max-gpu-memory", type=str)

	parser.add_argument("--question-begin", type=int)
	parser.add_argument("--question-end", type=int)
	parser.add_argument("--batch-size", type=int, default=200, help="预留参数，兼容后续 vLLM")
	parser.add_argument("--seed", type=int, default=42)

	return parser.parse_args()


def parse_experiment_list(experiment_str: str) -> List[str]:
	"""将用户输入的逗号分隔实验名整理为列表"""

	experiments = [exp.strip().lower() for exp in experiment_str.split(",") if exp.strip()]
	if not experiments:
		raise ValueError("未指定任何实验，至少需要一个实验模式")
	return experiments


def build_plan(name: str, stop_method: str) -> ExperimentPlan:
	"""依据预设名称构建对应的实验计划"""

	if name in {"baseline"}:
		return ExperimentPlan(
			name="baseline",
			display_name="Baseline",
			mode="baseline",
			description="仅运行目标模型，作为速度与质量基准",
			early_stop_method=None,
			needs_baseline=False,
		)

	if name in {"spec", "speculative", "eagle"}:
		return ExperimentPlan(
			name="speculative",
			display_name="投机解码",
			mode="eagle",
			description="经典 Eagle 投机解码，关闭早退信号",
			early_stop_method=None,
			needs_baseline=True,
		)

	if name in {"specexit", "fusion", "deepconf"}:
		return ExperimentPlan(
			name="specexit",
			display_name="投机+DeepConf+SpecExit",
			mode="eagle",
			description="在投机解码上叠加 DeepConf 决策与 SpecExit 早退信号",
			early_stop_method=stop_method,
			needs_baseline=True,
		)

	if name in {"specexit_ppl", "deepconf_ppl", "specexit_confusion"}:
		return ExperimentPlan(
			name="specexit_ppl",
			display_name="投机+DeepConf(困惑度)",
			mode="eagle",
			description="以区间困惑度为依据的 DeepConf 剪枝版本",
			early_stop_method=stop_method,
			needs_baseline=True,
			config_overrides={"use_confusion_prune": True},
		)

	raise ValueError(f"未知实验类型: {name}")


def ensure_baseline_plan(plans: List[ExperimentPlan], has_baseline_file: bool) -> List[ExperimentPlan]:
	"""若需要 baseline 但未提供，自动插入 baseline 计划"""

	needs_baseline = any(plan.needs_baseline for plan in plans)
	already_has_baseline = any(plan.is_baseline for plan in plans)
	if needs_baseline and not (already_has_baseline or has_baseline_file):
		print("[提示] 未提供 baseline 结果，自动追加 baseline 运行以便对比。")
		return [build_plan("baseline", stop_method="")] + plans
	return plans


def build_base_config(args: argparse.Namespace) -> Dict[str, Optional[object]]:
	"""整理公用的 Benchmark 配置项"""

	config = {
		"base_model_path": args.base_model_path,
		"eagle_model_path": args.eagle_model_path,
		"model_id": args.model_id,
		"bench_name": args.bench_name,
		"num_choices": args.num_choices,
		"temperature": args.temperature,
		"max_new_token": args.max_new_token,
		"num_gpus_per_model": args.num_gpus_per_model,
		"num_gpus_total": args.num_gpus_total,
		"max_gpu_memory": args.max_gpu_memory,
		"question_begin": args.question_begin,
		"question_end": args.question_end,
		"calculate_metrics": False,
		"top_p": args.top_p,
		"top_k": args.top_k,
		"depth": args.depth,
	}

	# PyTorch 版本需要 total_token
	config["total_token"] = args.total_token
	config["use_confusion_prune"] = args.use_confusion_prune
	config["confusion_window"] = args.confusion_window
	config["confusion_threshold"] = args.confusion_threshold
	return config


def summarize_eagle_file(answer_file: str) -> Dict[str, Optional[float]]:
	"""读取 Eagle/SpecExit 结果文件，统计核心指标"""

	stats = {
		"samples": 0,
		"total_tokens": 0.0,
		"total_time": 0.0,
		"total_accept": 0.0,
		"accept_steps": 0,
	}

	with open(answer_file, "r", encoding="utf-8") as fin:
		for line in fin:
			data = json.loads(line)
			choice = data["choices"][0]
			stats["samples"] += 1
			stats["total_tokens"] += sum(choice.get("new_tokens", []))
			stats["total_time"] += sum(choice.get("wall_time", []))
			accept_list = choice.get("accept_length", [])
			stats["total_accept"] += sum(accept_list)
			stats["accept_steps"] += len(accept_list)

	metrics = {
		"samples": stats["samples"],
		"avg_tokens_per_sample": (
			stats["total_tokens"] / stats["samples"] if stats["samples"] else 0.0
		),
		"avg_latency": (
			stats["total_time"] / stats["samples"] if stats["samples"] else 0.0
		),
		"tokens_per_second": (
			stats["total_tokens"] / stats["total_time"] if stats["total_time"] else 0.0
		),
		"avg_accept_length": (
			stats["total_accept"] / stats["accept_steps"] if stats["accept_steps"] else None
		),
	}
	return metrics


def summarize_baseline_file(answer_file: str) -> Dict[str, Optional[float]]:
	"""读取 baseline jsonl，统计平均 token / 耗时"""

	stats = {
		"samples": 0,
		"total_tokens": 0.0,
		"total_time": 0.0,
	}

	with open(answer_file, "r", encoding="utf-8") as fin:
		for line in fin:
			data = json.loads(line)
			choice = data["choices"][0]
			stats["samples"] += 1
			stats["total_tokens"] += sum(choice.get("new_tokens", []))
			stats["total_time"] += sum(choice.get("wall_time", []))

	metrics = {
		"samples": stats["samples"],
		"avg_tokens_per_sample": (
			stats["total_tokens"] / stats["samples"] if stats["samples"] else 0.0
		),
		"avg_latency": (
			stats["total_time"] / stats["samples"] if stats["samples"] else 0.0
		),
		"tokens_per_second": (
			stats["total_tokens"] / stats["total_time"] if stats["total_time"] else 0.0
		),
		"avg_accept_length": None,
	}
	return metrics


def format_metric(value: Optional[float], digits: int = 2) -> str:
	"""将浮点指标转为字符串，便于打印"""

	if value is None:
		return "-"
	return f"{value:.{digits}f}"


def print_summary_table(records: List[Dict[str, object]]) -> None:
	"""以 ASCII 表格形式展示结果"""

	if not records:
		print("暂无可展示的结果")
		return

	headers = [
		"实验",
		"平均新Token",
		"平均耗时(秒)",
		"吞吐(token/s)",
		"接受长度",
		"相对Baseline加速",
	]

	rows = []
	for record in records:
		metrics = record["metrics"]
		rows.append(
			[
				record["display_name"],
				format_metric(metrics.get("avg_tokens_per_sample")),
				format_metric(metrics.get("avg_latency")),
				format_metric(metrics.get("tokens_per_second")),
				format_metric(metrics.get("avg_accept_length"), digits=2),
				format_metric(metrics.get("speedup_vs_baseline")),
			]
		)

	col_widths = [len(h) for h in headers]
	for row in rows:
		for idx, cell in enumerate(row):
			col_widths[idx] = max(col_widths[idx], len(cell))

	def _fmt_line(cells: List[str]) -> str:
		return " | ".join(cell.ljust(col_widths[idx]) for idx, cell in enumerate(cells))

	print("\n==== 实验结果速览 ====")
	print(_fmt_line(headers))
	print("-" * (sum(col_widths) + 3 * (len(headers) - 1)))
	for row in rows:
		print(_fmt_line(row))
	print()


def run_single_experiment(
	plan: ExperimentPlan,
	args: argparse.Namespace,
	base_config: Dict[str, Optional[object]],
	base_output_dir: str,
	baseline_file: Optional[str],
) -> Dict[str, object]:
	"""执行单个实验并输出指标"""

	engine = SpecEngine(deploy_backend=args.deploy_backend)
	config = dict(base_config)
	if plan.config_overrides:
		config.update(plan.config_overrides)
	exp_output_dir = os.path.join(base_output_dir, plan.name)
	os.makedirs(exp_output_dir, exist_ok=True)
	config["output_dir"] = exp_output_dir
	config["early_stop_method"] = plan.early_stop_method

	print(f"\n[{plan.display_name}] 启动，输出目录: {exp_output_dir}")
	engine.setup_benchmark(**config)

	if plan.is_baseline:
		engine.run_baseline_benchmark()
		result_file = engine.benchmark_engine.baseline_file
		metrics = summarize_baseline_file(result_file)
	else:
		engine.run_eagle_benchmark()
		result_file = engine.benchmark_engine.eagle_file
		metrics = summarize_eagle_file(result_file)

	speedup = None
	if not plan.is_baseline and baseline_file:
		try:
			speedup = engine.calculate_speedup_ratio(
				baseline_file=baseline_file,
				eagle_file=result_file,
				model_path=args.base_model_path,
			)
		except Exception as exc:  # noqa: BLE001
			print(f"[警告] 计算加速比失败: {exc}")
	metrics["speedup_vs_baseline"] = speedup

	print(f"[{plan.display_name}] 完成，结果文件: {result_file}")
	return {
		"name": plan.name,
		"display_name": plan.display_name,
		"description": plan.description,
		"result_file": result_file,
		"output_dir": exp_output_dir,
		"metrics": metrics,
	}


def main() -> None:
	args = parse_args()
	setup_seed(args.seed)

	experiment_names = parse_experiment_list(args.experiments)
	plans = [build_plan(name, args.specexit_stop_method) for name in experiment_names]

	baseline_file = None
	if args.baseline_file:
		baseline_path = os.path.abspath(args.baseline_file)
		if not os.path.exists(baseline_path):
			raise FileNotFoundError(f"找不到 baseline 文件: {baseline_path}")
		baseline_file = baseline_path

	plans = ensure_baseline_plan(plans, has_baseline_file=baseline_file is not None)

	if args.output_dir:
		base_output_dir = os.path.abspath(args.output_dir)
	else:
		base_output_dir = os.path.join(os.getcwd(), "specexit_runs", args.model_id)
	os.makedirs(base_output_dir, exist_ok=True)

	base_config = build_base_config(args)

	all_records: List[Dict[str, object]] = []
	for plan in plans:
		record = run_single_experiment(
			plan=plan,
			args=args,
			base_config=base_config,
			base_output_dir=base_output_dir,
			baseline_file=baseline_file,
		)
		if plan.is_baseline:
			baseline_file = record["result_file"]
		all_records.append(record)

	print_summary_table(all_records)

	summary_path = args.summary_file or os.path.join(base_output_dir, "specexit_summary.json")
	with open(summary_path, "w", encoding="utf-8") as fout:
		json.dump(all_records, fout, ensure_ascii=False, indent=2)
	print(f"汇总已写入: {summary_path}")


if __name__ == "__main__":
	main()

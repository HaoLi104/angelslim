#!/usr/bin/env python3
"""
保存训练完成的 drafter 模型为推理格式
从训练时的 trainer 状态中提取模型并保存
"""

import argparse
import os
from pathlib import Path

import torch

from angelslim.compressor.speculative import DraftModelConfig, create_draft_model
from angelslim.utils import rank0_print


def parse_args():
    parser = argparse.ArgumentParser(description="Save final drafter model from training")
    
    parser.add_argument(
        "--draft_model_config_path",
        type=str,
        required=True,
        help="Path to draft model config JSON file",
    )
    parser.add_argument(
        "--target_model_name_or_path",
        type=str,
        required=True,
        help="Path to target model (for loading embeddings)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to training checkpoint directory or output_dir",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path to save the final model",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=None,
        help="Specific checkpoint step to load (if None, try to find latest)",
    )
    
    return parser.parse_args()


def find_latest_checkpoint(checkpoint_dir: str):
    """Find the latest checkpoint directory"""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Look for checkpoint-* directories
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
    
    if checkpoints:
        return str(checkpoints[-1])
    
    # If no checkpoint-* found, check if the directory itself contains model files
    if (checkpoint_dir / "pytorch_model.bin").exists() or (checkpoint_dir / "model.safetensors").exists():
        return str(checkpoint_dir)
    
    return None


def load_model_from_checkpoint(checkpoint_path: str, config_path: str, target_model_path: str):
    """Load drafter model from checkpoint"""
    rank0_print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load config and create model
    draft_model_config = DraftModelConfig.from_file(config_path)
    draft_model = create_draft_model(draft_model_config)
    
    # Load embeddings from target model
    rank0_print("Loading embedding weights from target model...")
    draft_model.load_embed_weights(target_model_path)
    draft_model.freeze_embed_weights()
    
    # Load checkpoint weights
    checkpoint_path = Path(checkpoint_path)
    
    # Try pytorch_model.bin first
    model_file = checkpoint_path / "pytorch_model.bin"
    if not model_file.exists():
        # Try trainer_state.bin (transformers format)
        trainer_state_file = checkpoint_path / "trainer_state.bin"
        if trainer_state_file.exists():
            rank0_print("Found trainer_state.bin, extracting model weights...")
            trainer_state = torch.load(trainer_state_file, map_location="cpu")
            if "model_state_dict" in trainer_state:
                state_dict = trainer_state["model_state_dict"]
            else:
                raise ValueError("trainer_state.bin doesn't contain model_state_dict")
        else:
            raise FileNotFoundError(f"No model file found in {checkpoint_path}")
    else:
        rank0_print(f"Loading weights from {model_file}...")
        state_dict = torch.load(model_file, map_location="cpu")
    
    # Filter out embedding weights (they're loaded from target model)
    filtered_state_dict = {k: v for k, v in state_dict.items() if "embed_tokens" not in k}
    
    # Load state dict
    missing_keys, unexpected_keys = draft_model.load_state_dict(filtered_state_dict, strict=False)
    
    if missing_keys:
        rank0_print(f"Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        rank0_print(f"Warning: Unexpected keys: {unexpected_keys}")
    
    rank0_print("Model loaded successfully")
    return draft_model, draft_model_config


def save_model(model, config, output_path: str):
    """Save model and config to output path"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    rank0_print(f"Saving model to {output_path}...")
    
    # Save config
    config_path = output_path / "config.json"
    rank0_print(f"Saving config to {config_path}...")
    config.save_pretrained(str(output_path))
    
    # Save model weights
    model_path = output_path / "pytorch_model.bin"
    rank0_print(f"Saving model weights to {model_path}...")
    
    # Get state dict (excluding embeddings which are loaded from target)
    state_dict = {k: v.cpu() for k, v in model.state_dict().items() if "embed_tokens" not in k}
    torch.save(state_dict, model_path)
    
    rank0_print(f"Model saved successfully to {output_path}")


def main():
    args = parse_args()
    
    # Find checkpoint
    if args.checkpoint_step:
        checkpoint_path = Path(args.checkpoint_dir) / f"checkpoint-{args.checkpoint_step}"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint_path = str(checkpoint_path)
    else:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        if checkpoint_path is None:
            # If no checkpoint found, try to load from output_dir directly
            # This assumes the model is in memory from a recent training run
            rank0_print("No checkpoint found. Please provide --checkpoint_step or ensure checkpoint exists.")
            rank0_print("Alternatively, you can manually save the model after training completes.")
            return
        rank0_print(f"Using latest checkpoint: {checkpoint_path}")
    
    # Load model
    model, config = load_model_from_checkpoint(
        checkpoint_path,
        args.draft_model_config_path,
        args.target_model_name_or_path,
    )
    
    # Save model
    save_model(model, config, args.output_path)
    
    rank0_print("Done!")


if __name__ == "__main__":
    main()


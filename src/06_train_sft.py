import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.env import get_dirs, get_env, resolve_path
from src.utils.logging_utils import setup_logger, banner, log_config_safely, log_env_safely, timer


def load_cfg(path: str) -> dict:
    import yaml
    return yaml.safe_load(open(path, "r", encoding="utf-8"))


def _count_trainable_params(model) -> tuple[int, int]:
    """Return (trainable_params, total_params) for the current model."""
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def _pick_device():
    """Choose a single device for stable local fine-tuning."""
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def _validate_paths(train_path: Path, eval_path: Optional[Path]) -> None:
    """Fail fast if required training/eval files are missing."""
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")
    if train_path.is_dir():
        raise ValueError(f"Training dataset path must be a file, got directory: {train_path}")
    if eval_path is not None:
        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation dataset path not found: {eval_path}")
        if eval_path.is_dir():
            raise ValueError(f"Evaluation dataset path must be a file, got directory: {eval_path}")


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return str(value)


def _save_metadata(path: Path, meta: dict) -> None:
    """Persist metadata in a stable, human-readable JSON format."""
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logger = setup_logger("06_train_sft")
    # This stage fine-tunes next-utterance generation behavior using chat-format SFT data.
    banner(logger, "Step 06: Train SFT (LoRA)")
    log_env_safely(logger, ["DATA_DIR", "OUTPUTS_DIR", "BASE_MODEL"])
    try:
        dirs = get_dirs()
        cfg = load_cfg(args.config)
        log_config_safely(logger, cfg, "config")
        # Prefer the config's model for reproducibility; fall back to env only if config omits it.
        base = cfg.get("base_model") or get_env("BASE_MODEL")
        if not base:
            raise ValueError("Missing base model. Set `base_model` in config or BASE_MODEL env var.")
        data_cfg = cfg["data"]
        if "train_path" not in data_cfg:
            raise ValueError("Missing required config field: data.train_path")
        train_path = resolve_path(data_cfg["train_path"], dirs["data"])
        eval_path = resolve_path(data_cfg.get("eval_path", ""), dirs["data"]) if data_cfg.get("eval_path") else None
        _validate_paths(train_path, eval_path)
        logger.info("train_path=%s eval_path=%s model=%s", train_path, eval_path, base)

        with timer(logger, "load_dataset"):
            try:
                ds_train = load_dataset("json", data_files=str(train_path), split="train")
            except Exception as e:
                raise ValueError(f"Failed to load training dataset from {train_path}: {e}") from e
            ds_eval = None
            if eval_path:
                try:
                    ds_eval = load_dataset("json", data_files=str(eval_path), split="train")
                except Exception as e:
                    raise ValueError(f"Failed to load eval dataset from {eval_path}: {e}") from e
        logger.info("train_samples=%s eval_samples=%s", len(ds_train), len(ds_eval) if ds_eval else 0)

        with timer(logger, "load_model_and_tokenizer"):
            tok = AutoTokenizer.from_pretrained(base, use_fast=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            device, device_name = _pick_device()
            logger.info("Selected device=%s", device_name)
            if device_name == "cpu":
                logger.warning("Training on CPU can be very slow; use CUDA or MPS when available.")

            model = AutoModelForCausalLM.from_pretrained(base)
            model.to(device)
            logger.info("model_device=%s", device)

        if cfg["lora"]["enabled"]:
            target_modules = cfg["lora"].get("target_modules")
            lora = LoraConfig(
                r=int(cfg["lora"]["r"]),
                lora_alpha=int(cfg["lora"]["alpha"]),
                lora_dropout=float(cfg["lora"]["dropout"]),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            model = get_peft_model(model, lora)
            logger.info(
                "LoRA enabled r=%s alpha=%s dropout=%s target_modules=%s",
                cfg["lora"]["r"],
                cfg["lora"]["alpha"],
                cfg["lora"]["dropout"],
                target_modules,
            )
        else:
            target_modules = None
            logger.info("LoRA disabled; full model fine-tuning mode.")

        trainable_params, total_params = _count_trainable_params(model)
        trainable_pct = (100.0 * trainable_params / total_params) if total_params else 0.0

        prec = cfg.get("precision", {})
        bf16 = (prec.get("bf16") is True) or (prec.get("bf16") == "auto" and torch.cuda.is_available())
        fp16 = not bf16 and ((prec.get("fp16") is True) or (prec.get("fp16") == "auto" and torch.cuda.is_available()))

        tr_cfg = cfg["training"]
        output_dir = resolve_path(tr_cfg["output_dir"], dirs["outputs"])
        output_dir.mkdir(parents=True, exist_ok=True)
        max_steps = int(tr_cfg["max_steps"]) if tr_cfg.get("max_steps") else -1
        num_epochs = float(tr_cfg.get("num_train_epochs", 1))
        per_device_bs = int(tr_cfg["per_device_train_batch_size"])
        per_device_eval_bs = int(tr_cfg.get("per_device_eval_batch_size", per_device_bs))
        grad_accum = int(tr_cfg.get("gradient_accumulation_steps", 1))
        effective_batch_size = per_device_bs * grad_accum
        max_length = int(cfg["training"].get("max_seq_len", cfg["training"].get("max_seq_length", 2048)))
        logging_steps = int(tr_cfg.get("logging_steps", 10))
        save_steps = int(tr_cfg.get("save_steps", max(50, logging_steps)))
        seed = int(tr_cfg.get("seed", 42))
        eval_strategy = "steps" if ds_eval else "no"
        eval_steps = save_steps if ds_eval else None
        save_strategy = str(tr_cfg.get("save_strategy", "steps"))
        learning_rate = float(tr_cfg["learning_rate"])
        torch_dtype_used = str(next(model.parameters()).dtype) if total_params > 0 else "unknown"
        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info("Run summary:")
        logger.info(
            "  model=%s output_dir=%s device=%s dtype=%s",
            base,
            output_dir,
            device_name,
            torch_dtype_used,
        )
        logger.info("  train_path=%s eval_path=%s", train_path, eval_path)
        logger.info(
            "  train_samples=%s eval_samples=%s train_bs=%s grad_accum=%s effective_batch_size=%s max_seq_length=%s",
            len(ds_train),
            len(ds_eval) if ds_eval else 0,
            per_device_bs,
            grad_accum,
            effective_batch_size,
            max_length,
        )
        logger.info(
            "  num_train_epochs=%s max_steps=%s learning_rate=%s",
            num_epochs,
            max_steps,
            learning_rate,
        )
        logger.info(
            "  lora_enabled=%s r=%s alpha=%s dropout=%s target_modules=%s",
            cfg["lora"]["enabled"],
            cfg["lora"].get("r"),
            cfg["lora"].get("alpha"),
            cfg["lora"].get("dropout"),
            target_modules,
        )
        logger.info(
            "output_dir=%s max_steps=%s num_train_epochs=%s effective_batch_size=%s trainable_params=%s/%s (%.2f%%)",
            output_dir,
            max_steps,
            num_epochs,
            effective_batch_size,
            trainable_params,
            total_params,
            trainable_pct,
        )
        if max_steps > 0 and max_steps <= 50:
            logger.warning("max_steps=%s suggests a demo/smoke-test run; results may be limited.", max_steps)
        if len(ds_train) < 100:
            logger.warning("train_sample_count=%s is small and likely for demo/smoke-test validation.", len(ds_train))
        args_tr = SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=per_device_bs,
            per_device_eval_batch_size=per_device_eval_bs,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            learning_rate=learning_rate,
            warmup_ratio=float(tr_cfg.get("warmup_ratio", 0.03)),
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_strategy=save_strategy,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            report_to=[],
            bf16=bf16,
            fp16=fp16,
            seed=seed,
            max_length=max_length,
            packing=False,
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tok,
            train_dataset=ds_train,
            eval_dataset=ds_eval,
            args=args_tr,
        )
        meta_path = output_dir / "train_run_metadata.json"
        run_meta = {
            "project_task": "multilingual next-utterance generation",
            "script": "06_train_sft.py",
            "config_path": str(Path(args.config).resolve()),
            "timestamp": timestamp,
            "base_model_name": base,
            "output_dir": str(output_dir),
            "train_dataset_path": str(train_path),
            "eval_dataset_path": str(eval_path) if eval_path else None,
            "train_sample_count": len(ds_train),
            "eval_sample_count": len(ds_eval) if ds_eval else 0,
            "seed": seed,
            "learning_rate": learning_rate,
            "num_train_epochs": num_epochs,
            "max_steps": max_steps,
            "per_device_train_batch_size": per_device_bs,
            "per_device_eval_batch_size": per_device_eval_bs,
            "gradient_accumulation_steps": grad_accum,
            "effective_train_batch_size": effective_batch_size,
            "max_seq_length": max_length,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "eval_steps": eval_steps,
            "save_strategy": _to_jsonable(args_tr.save_strategy),
            "eval_strategy": _to_jsonable(args_tr.eval_strategy),
            "bf16": bool(bf16),
            "fp16": bool(fp16),
            "lora_enabled": bool(cfg["lora"]["enabled"]),
            "lora_r": int(cfg["lora"]["r"]) if cfg["lora"]["enabled"] else None,
            "lora_alpha": int(cfg["lora"]["alpha"]) if cfg["lora"]["enabled"] else None,
            "lora_dropout": float(cfg["lora"]["dropout"]) if cfg["lora"]["enabled"] else None,
            "target_modules": target_modules,
            "device_used": device_name,
            "torch_dtype_used": torch_dtype_used,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": round(trainable_pct, 6),
            "final_adapter_path": None,
            "tokenizer_saved_path": None,
            "training_completed": False,
        }
        _save_metadata(meta_path, run_meta)
        logger.info("train_metadata_path(initial)=%s", meta_path)
        with timer(logger, "train"):
            trainer.train()
        out_dir = output_dir / "lora_adapter"
        out_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
        logger.info("adapter_saved_to=%s tokenizer_saved_to=%s", out_dir, out_dir)
        run_meta["final_adapter_path"] = str(out_dir)
        run_meta["tokenizer_saved_path"] = str(out_dir)
        run_meta["training_completed"] = True
        _save_metadata(meta_path, run_meta)
        logger.info("train_metadata_path(final)=%s", meta_path)
        banner(logger, "Step 06: Done", char="-")
    except Exception:
        logger.exception("Step 06 failed")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

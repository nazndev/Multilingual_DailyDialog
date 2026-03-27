import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

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
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


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
        logger.info("train_path=%s eval_path=%s model=%s", train_path, eval_path, base)

        with timer(logger, "load_dataset"):
            ds_train = load_dataset("json", data_files=str(train_path), split="train")
            ds_eval = None
            if eval_path and Path(eval_path).exists():
                ds_eval = load_dataset("json", data_files=str(eval_path), split="train")
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
            lora = LoraConfig(
                r=int(cfg["lora"]["r"]),
                lora_alpha=int(cfg["lora"]["alpha"]),
                lora_dropout=float(cfg["lora"]["dropout"]),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora)
            logger.info("LoRA enabled r=%s alpha=%s", cfg["lora"]["r"], cfg["lora"]["alpha"])
        else:
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
        grad_accum = int(tr_cfg.get("gradient_accumulation_steps", 1))
        effective_batch_size = per_device_bs * grad_accum
        max_length = int(cfg["training"].get("max_seq_len", cfg["training"].get("max_seq_length", 2048)))
        logging_steps = int(tr_cfg.get("logging_steps", 10))
        save_steps = int(tr_cfg.get("save_steps", max(50, logging_steps)))
        seed = int(tr_cfg.get("seed", 42))
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
        if max_steps > 0 and max_steps < 20:
            logger.warning("Very small max_steps=%s may be insufficient for meaningful learning.", max_steps)
        if len(ds_train) < 50:
            logger.warning("Very small dataset (%s rows): treat this run as a smoke test.", len(ds_train))
        args_tr = SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=per_device_bs,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            learning_rate=float(tr_cfg["learning_rate"]),
            warmup_ratio=float(tr_cfg.get("warmup_ratio", 0.03)),
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_strategy="steps" if ds_eval else "no",
            eval_steps=save_steps if ds_eval else None,
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
        with timer(logger, "train"):
            trainer.train()
        out_dir = output_dir / "lora_adapter"
        out_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
        logger.info("adapter_saved_to=%s", out_dir)
        run_meta = {
            "script": "06_train_sft.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": base,
            "device": device_name,
            "train_path": str(train_path),
            "eval_path": str(eval_path) if eval_path else None,
            "train_samples": len(ds_train),
            "eval_samples": len(ds_eval) if ds_eval else 0,
            "output_dir": str(output_dir),
            "adapter_dir": str(out_dir),
            "max_seq_length": max_length,
            "max_steps": max_steps,
            "num_train_epochs": num_epochs,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "per_device_train_batch_size": per_device_bs,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": effective_batch_size,
            "seed": seed,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_percent": round(trainable_pct, 4),
        }
        meta_path = output_dir / "train_run_metadata.json"
        meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        logger.info("train_metadata_path=%s", meta_path)
        banner(logger, "Step 06: Done", char="-")
    except Exception:
        logger.exception("Step 06 failed")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

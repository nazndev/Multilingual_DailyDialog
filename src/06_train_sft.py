import argparse
import inspect
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.env import get_dirs, get_env, resolve_path
from src.utils.logging_utils import setup_logger, banner, log_config_safely, log_env_safely, timer
from src.utils.prompting import normalize_messages_for_model


def load_cfg(path: str) -> dict:
    import yaml
    return yaml.safe_load(open(path, "r", encoding="utf-8"))


def count_parameters(model) -> tuple[int, int, float]:
    """Return (total_parameters, trainable_parameters, trainable_percentage)."""
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    trainable_percentage = (100.0 * trainable / total) if total else 0.0
    return total, trainable, trainable_percentage


def _pick_device():
    """Choose a single device for stable local fine-tuning when not using device_map."""
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def _parse_torch_dtype(name: Optional[str]):
    if not name or str(name).lower() in ("auto", "none", "null"):
        return None
    n = str(name).lower()
    if n in ("bfloat16", "bf16"):
        return torch.bfloat16
    if n in ("float16", "fp16"):
        return torch.float16
    if n in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported model.torch_dtype={name!r}")


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


def save_json(path: Path, data: dict) -> None:
    """Persist JSON in a stable, human-readable format."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _detect_model_family(model_id: str) -> str:
    lowered = (model_id or "").lower()
    if "qwen" in lowered:
        return "qwen"
    if "gemma" in lowered:
        return "gemma"
    raise ValueError(f"Unsupported model family for base_model={model_id!r}; expected qwen or gemma.")


def _pad_batch(batch: list[list[int]], pad_value: int) -> list[list[int]]:
    max_len = max(len(x) for x in batch) if batch else 0
    return [x + [pad_value] * (max_len - len(x)) for x in batch]


class ResponseOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = _pad_batch([list(f["input_ids"]) for f in features], self.pad_token_id)
        attention_mask = _pad_batch([list(f["attention_mask"]) for f in features], 0)
        labels = _pad_batch([list(f["labels"]) for f in features], -100)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _to_token_ids(tok, text: str) -> list[int]:
    enc = tok(text, add_special_tokens=False, return_attention_mask=False)
    return list(enc["input_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logger = setup_logger("06_train_sft")
    banner(logger, "Step 06: Train SFT (LoRA)")
    log_env_safely(logger, ["DATA_DIR", "OUTPUTS_DIR", "BASE_MODEL"])
    try:
        dirs = get_dirs()
        cfg = load_cfg(args.config)
        log_config_safely(logger, cfg, "config")
        base = cfg.get("base_model") or get_env("BASE_MODEL")
        if not base:
            raise ValueError("Missing base model. Set `base_model` in config or BASE_MODEL env var.")
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        load_in_4bit = bool(model_cfg.get("load_in_4bit", False))
        load_in_8bit = bool(model_cfg.get("load_in_8bit", False))
        if load_in_4bit and load_in_8bit:
            raise ValueError("model.load_in_4bit and model.load_in_8bit cannot both be true.")
        device_map = model_cfg.get("device_map")
        dtype_str = model_cfg.get("torch_dtype")
        torch_dtype = _parse_torch_dtype(dtype_str)

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
        train_sample_count = len(ds_train)
        eval_sample_count = len(ds_eval) if ds_eval else 0
        logger.info("train_samples=%s eval_samples=%s", train_sample_count, eval_sample_count)

        tr_cfg = cfg["training"]
        gradient_checkpointing = bool(tr_cfg.get("gradient_checkpointing", False))
        resume_from_checkpoint = tr_cfg.get("resume_from_checkpoint")
        if resume_from_checkpoint:
            resume_from_checkpoint = str(resolve_path(resume_from_checkpoint, dirs["outputs"]))

        with timer(logger, "load_tokenizer"):
            tok = AutoTokenizer.from_pretrained(base, use_fast=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

        model_family = _detect_model_family(base)
        logger.info("model_family=%s", model_family)

        max_length = int(cfg["training"].get("max_seq_len", cfg["training"].get("max_seq_length", 2048)))
        debug_state = {"printed": False}
        with timer(logger, "format_sft_datasets"):
            def _fmt(ex):
                if not isinstance(ex, Mapping):
                    return {"ok": False}
                row = dict(ex)
                raw_messages = row.get("messages")
                if not isinstance(raw_messages, list) or not raw_messages:
                    return {"ok": False}
                try:
                    messages = [dict(m) for m in raw_messages if isinstance(m, Mapping)]
                    if model_family == "gemma":
                        messages = normalize_messages_for_model(messages, "gemma")
                        allowed_roles = {"user", "model"}
                        target_role = "model"
                        if any((m.get("role") == "system") for m in messages):
                            raise ValueError("Gemma sample contains system role after normalization.")
                    else:
                        allowed_roles = {"system", "user", "assistant"}
                        target_role = "assistant"

                    if len(messages) < 2:
                        raise ValueError("Need at least one prompt message and one target message.")
                    for i, m in enumerate(messages):
                        role = m.get("role")
                        content = m.get("content")
                        if role not in allowed_roles:
                            raise ValueError(f"Invalid role={role!r} at messages[{i}] for family={model_family}.")
                        if not isinstance(content, str):
                            raise ValueError(f"messages[{i}].content must be string.")
                    final_msg = messages[-1]
                    if final_msg.get("role") != target_role:
                        raise ValueError(
                            f"Final message role must be {target_role!r}, got {final_msg.get('role')!r}."
                        )
                    target_text = (final_msg.get("content") or "").strip()
                    if not target_text:
                        raise ValueError("Final target span is empty.")

                    prompt_messages = messages[:-1]
                    prefix_text = tok.apply_chat_template(
                        prompt_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    full_text = tok.apply_chat_template(messages, tokenize=False)
                    prefix_ids = _to_token_ids(tok, prefix_text)
                    full_ids = _to_token_ids(tok, full_text)
                    target_start = len(prefix_ids)
                    if target_start >= len(full_ids):
                        raise ValueError("Target start index is outside formatted sequence.")
                    labels = [-100] * target_start + full_ids[target_start:]
                    attention_mask = [1] * len(full_ids)

                    if len(full_ids) > max_length:
                        cut = len(full_ids) - max_length
                        full_ids = full_ids[cut:]
                        attention_mask = attention_mask[cut:]
                        labels = labels[cut:]
                        target_start = max(0, target_start - cut)

                    supervised_tokens = sum(1 for t in labels if t != -100)
                    masked_tokens = len(labels) - supervised_tokens
                    if supervised_tokens <= 0:
                        raise ValueError("Label mask has no supervised target tokens.")

                    if not debug_state["printed"]:
                        preview = full_text if len(full_text) <= 300 else (full_text[:300] + "...")
                        logger.info("debug_preview_text=%s", preview.replace("\n", "\\n"))
                        logger.info(
                            "debug_lengths total_tokens=%s target_start=%s supervised_tokens=%s masked_tokens=%s",
                            len(full_ids),
                            target_start,
                            supervised_tokens,
                            masked_tokens,
                        )
                        debug_state["printed"] = True

                    return {
                        "ok": True,
                        "text": full_text,
                        "input_ids": full_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                        "target_start": target_start,
                        "supervised_tokens": supervised_tokens,
                        "masked_tokens": masked_tokens,
                    }
                except Exception:
                    return {"ok": False}

            train_cols = list(ds_train.column_names)
            ds_train = ds_train.map(_fmt, remove_columns=train_cols, batched=False)
            if ds_eval is not None:
                eval_cols = list(ds_eval.column_names)
                ds_eval = ds_eval.map(_fmt, remove_columns=eval_cols, batched=False)

            ds_train = ds_train.filter(lambda x: bool(x.get("ok")), batched=False)
            if ds_eval is not None:
                ds_eval = ds_eval.filter(lambda x: bool(x.get("ok")), batched=False)
            ds_train = ds_train.remove_columns(["ok"])
            if ds_eval is not None:
                ds_eval = ds_eval.remove_columns(["ok"])

            train_sample_count = len(ds_train)
            eval_sample_count = len(ds_eval) if ds_eval else 0
            logger.info(
                "sft_dataset_after_filter train_samples=%s eval_samples=%s",
                train_sample_count,
                eval_sample_count,
            )
            if train_sample_count == 0:
                raise ValueError("No valid training samples after role and target-span validation.")
            train_supervised = sum(int(x.get("supervised_tokens", 0)) for x in ds_train)
            train_masked = sum(int(x.get("masked_tokens", 0)) for x in ds_train)
            logger.info(
                "label_mask_validation train_supervised_tokens=%s train_masked_tokens=%s",
                train_supervised,
                train_masked,
            )
            if train_supervised <= 0:
                raise ValueError("Label mask validation failed: supervised token count is zero.")

            logger.info("formatted_train_dataset_fields=%s", ds_train.column_names)
            logger.info("formatted_eval_dataset_fields=%s", ds_eval.column_names if ds_eval is not None else "none")

        with timer(logger, "load_model_and_tokenizer"):
            device, device_name = _pick_device()
            logger.info("Selected device=%s", device_name)
            if device_name == "cpu":
                logger.warning("Training on CPU can be very slow; use CUDA or MPS when available.")

            quantization_mode = "none"
            quantization_config = None
            if load_in_4bit:
                quantization_mode = "4bit"
                compute_dtype = torch_dtype or torch.bfloat16
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif load_in_8bit:
                quantization_mode = "8bit"
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            use_quant = load_in_4bit or load_in_8bit
            if use_quant and device_map is None:
                device_map = "auto"

            model_kwargs: dict[str, Any] = {}
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = device_map or "auto"
            else:
                if torch_dtype is not None:
                    model_kwargs["torch_dtype"] = torch_dtype
                if device_map is not None:
                    model_kwargs["device_map"] = device_map
            model = AutoModelForCausalLM.from_pretrained(base, **model_kwargs)
            if use_quant:
                model = prepare_model_for_kbit_training(model)
            elif device_map is None:
                model.to(device)
            logger.info("model_device_map=%s quantization=%s", device_map, quantization_mode)

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

        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        total_params, trainable_params, trainable_pct = count_parameters(model)

        prec = cfg.get("precision", {})
        bf16 = (prec.get("bf16") is True) or (prec.get("bf16") == "auto" and torch.cuda.is_available())
        fp16 = not bf16 and ((prec.get("fp16") is True) or (prec.get("fp16") == "auto" and torch.cuda.is_available()))

        output_dir = resolve_path(tr_cfg["output_dir"], dirs["outputs"])
        output_dir.mkdir(parents=True, exist_ok=True)
        max_steps = int(tr_cfg["max_steps"]) if tr_cfg.get("max_steps") else -1
        num_epochs = float(tr_cfg.get("num_train_epochs", 1))
        per_device_bs = int(tr_cfg["per_device_train_batch_size"])
        per_device_eval_bs = int(tr_cfg.get("per_device_eval_batch_size", per_device_bs))
        grad_accum = int(tr_cfg.get("gradient_accumulation_steps", 1))
        effective_train_batch_size = per_device_bs * grad_accum
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
            train_sample_count,
            eval_sample_count,
            per_device_bs,
            grad_accum,
            effective_train_batch_size,
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
            effective_train_batch_size,
            trainable_params,
            total_params,
            trainable_pct,
        )
        if max_steps != -1 and max_steps <= 50:
            logger.warning("max_steps=%s suggests a demo/smoke-test run; results may be limited.", max_steps)
        if train_sample_count < 100:
            logger.warning(
                "train_sample_count=%s is small and likely for demo/smoke-test validation.",
                train_sample_count,
            )
        ta_params = inspect.signature(TrainingArguments.__init__).parameters
        train_args_kwargs: dict[str, Any] = {
            "output_dir": str(output_dir),
            "per_device_train_batch_size": per_device_bs,
            "per_device_eval_batch_size": per_device_eval_bs,
            "gradient_accumulation_steps": grad_accum,
            "num_train_epochs": num_epochs,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "warmup_ratio": float(tr_cfg.get("warmup_ratio", 0.03)),
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "save_strategy": save_strategy,
            "report_to": [],
            "bf16": bf16,
            "fp16": fp16,
            "seed": seed,
            "gradient_checkpointing": gradient_checkpointing,
            "remove_unused_columns": False,
        }
        if eval_strategy == "steps":
            if "evaluation_strategy" in ta_params:
                train_args_kwargs["evaluation_strategy"] = eval_strategy
            elif "eval_strategy" in ta_params:
                train_args_kwargs["eval_strategy"] = eval_strategy
            train_args_kwargs["eval_steps"] = eval_steps
        else:
            if "evaluation_strategy" in ta_params:
                train_args_kwargs["evaluation_strategy"] = eval_strategy
            elif "eval_strategy" in ta_params:
                train_args_kwargs["eval_strategy"] = eval_strategy
        train_args = TrainingArguments(**train_args_kwargs)
        collator = ResponseOnlyCollator(pad_token_id=tok.pad_token_id)
        trainer_params = inspect.signature(Trainer.__init__).parameters
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": train_args,
            "train_dataset": ds_train,
            "eval_dataset": ds_eval,
            "data_collator": collator,
        }
        if "tokenizer" in trainer_params:
            trainer_kwargs["tokenizer"] = tok
        trainer = Trainer(**trainer_kwargs)
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
            "train_sample_count": train_sample_count,
            "eval_sample_count": eval_sample_count,
            "seed": seed,
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "num_train_epochs": num_epochs,
            "max_steps": max_steps,
            "per_device_train_batch_size": per_device_bs,
            "per_device_eval_batch_size": per_device_eval_bs,
            "gradient_accumulation_steps": grad_accum,
            "effective_train_batch_size": effective_train_batch_size,
            "effective_batch_size": effective_train_batch_size,
            "max_seq_length": max_length,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "eval_steps": eval_steps,
            "save_strategy": _to_jsonable(train_args.save_strategy),
            "eval_strategy": _to_jsonable(
                getattr(
                    train_args,
                    "evaluation_strategy",
                    getattr(train_args, "eval_strategy", None),
                )
            ),
            "bf16": bool(bf16),
            "fp16": bool(fp16),
            "dtype": torch_dtype_used,
            "torch_dtype_used": torch_dtype_used,
            "torch_dtype_config": dtype_str,
            "quantization_mode": quantization_mode,
            "load_in_4bit": load_in_4bit,
            "load_in_8bit": load_in_8bit,
            "device_map": device_map,
            "gradient_checkpointing": gradient_checkpointing,
            "lora_enabled": bool(cfg["lora"]["enabled"]),
            "lora_r": int(cfg["lora"]["r"]) if cfg["lora"]["enabled"] else None,
            "lora_alpha": int(cfg["lora"]["alpha"]) if cfg["lora"]["enabled"] else None,
            "lora_dropout": float(cfg["lora"]["dropout"]) if cfg["lora"]["enabled"] else None,
            "lora_settings": {
                "r": cfg["lora"].get("r"),
                "alpha": cfg["lora"].get("alpha"),
                "dropout": cfg["lora"].get("dropout"),
            },
            "target_modules": target_modules,
            "device_used": device_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": round(trainable_pct, 6),
            "final_adapter_path": None,
            "tokenizer_saved_path": None,
            "training_completed": False,
            "resume_from_checkpoint": resume_from_checkpoint,
            "model_family": model_family,
            "response_only_loss": True,
        }
        save_json(meta_path, run_meta)
        logger.info("train_metadata_path(initial)=%s", meta_path)
        with timer(logger, "train"):
            trainer.train(resume_from_checkpoint=resume_from_checkpoint if resume_from_checkpoint else None)
        out_dir = output_dir / "lora_adapter"
        out_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
        logger.info("adapter_saved_to=%s tokenizer_saved_to=%s", out_dir, out_dir)
        run_meta["final_adapter_path"] = str(out_dir)
        run_meta["tokenizer_saved_path"] = str(out_dir)
        run_meta["training_completed"] = True
        save_json(meta_path, run_meta)
        logger.info("train_metadata_path(final)=%s", meta_path)
        banner(logger, "Step 06: Done", char="-")
    except Exception:
        logger.exception("Step 06 failed")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

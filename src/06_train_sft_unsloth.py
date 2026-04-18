import argparse
import inspect
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from unsloth import FastLanguageModel  # type: ignore[import-not-found]
from unsloth.chat_templates import get_chat_template  # type: ignore[import-not-found]

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.env import get_dirs, get_env, resolve_path
from src.utils.logging_utils import banner, log_config_safely, log_env_safely, setup_logger, timer
from src.utils.prompting import select_unsloth_chat_template


def load_cfg(path: str) -> dict:
    import yaml

    return yaml.safe_load(open(path, "r", encoding="utf-8"))


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


def _detect_model_family(model_id: str) -> str:
    lowered = (model_id or "").lower()
    if "qwen" in lowered:
        return "qwen"
    if "gemma" in lowered:
        return "gemma"
    raise ValueError(f"Unsupported model family for base_model={model_id!r}; expected qwen or gemma.")


def _validate_paths(train_path: Path, eval_path: Optional[Path]) -> None:
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")
    if train_path.is_dir():
        raise ValueError(f"Training dataset path must be a file, got directory: {train_path}")
    if eval_path is not None:
        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation dataset path not found: {eval_path}")
        if eval_path.is_dir():
            raise ValueError(f"Evaluation dataset path must be a file, got directory: {eval_path}")


def _validate_dataset_family(model_family: str, train_path: Path, eval_path: Optional[Path]) -> None:
    train_l = str(train_path).lower()
    eval_l = str(eval_path).lower() if eval_path else ""
    if model_family == "qwen":
        if "qwen_bn" not in train_l or (eval_path and "qwen_bn" not in eval_l):
            raise ValueError(
                f"Dataset mismatch for qwen model: expected qwen_bn paths, got train={train_path}, eval={eval_path}."
            )
    if model_family == "gemma":
        if "gemma_bn" not in train_l or (eval_path and "gemma_bn" not in eval_l):
            raise ValueError(
                f"Dataset mismatch for gemma model: expected gemma_bn paths, got train={train_path}, eval={eval_path}."
            )


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return str(value)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _count_parameters(model) -> tuple[int, int, float]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = (100.0 * trainable / total) if total else 0.0
    return total, trainable, pct


def _tokenize_response_only_example(example: dict, tokenizer, max_length: int) -> dict:
    messages = example.get("messages") or []
    if not isinstance(messages, list) or len(messages) < 2:
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

    last_msg = messages[-1]
    last_role = (last_msg.get("role") or "").strip()
    if last_role not in ("assistant", "model"):
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

    target_text = (last_msg.get("content") or "").strip()
    if not target_text:
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

    prompt_messages = messages[:-1]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    prompt_ids = list(prompt_ids)

    target_ids = tokenizer(
        target_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    target_ids = list(target_ids)

    eos_id = tokenizer.eos_token_id
    if eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
        target_ids = target_ids + [eos_id]

    max_target_len = max_length - 1
    target_ids = target_ids[:max_target_len]

    available_prompt_len = max_length - len(target_ids)
    if available_prompt_len < 0:
        available_prompt_len = 0
    if len(prompt_ids) > available_prompt_len:
        prompt_ids = prompt_ids[-available_prompt_len:]

    input_ids = prompt_ids + target_ids
    labels = ([-100] * len(prompt_ids)) + target_ids
    attention_mask = [1] * len(input_ids)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must have pad_token_id or eos_token_id")

    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids += [pad_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _keep_valid_tokenized_example(example: dict, max_length: int) -> bool:
    input_ids = example.get("input_ids") or []
    labels = example.get("labels") or []
    return len(input_ids) == max_length and any(x != -100 for x in labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    logger = setup_logger("06_train_sft_unsloth")
    banner(logger, "Step 06: Train SFT (Unsloth)")
    log_env_safely(logger, ["DATA_DIR", "OUTPUTS_DIR", "BASE_MODEL"])

    try:
        dirs = get_dirs()
        cfg = load_cfg(args.config)
        log_config_safely(logger, cfg, "config")

        base = cfg.get("base_model") or get_env("BASE_MODEL")
        if not base:
            raise ValueError("Missing base model. Set `base_model` in config or BASE_MODEL env var.")
        model_family = _detect_model_family(base)
        chat_template = select_unsloth_chat_template(base)
        logger.info("model_family=%s", model_family)
        logger.info("unsloth_chat_template=%s", chat_template)

        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        load_in_4bit = bool(model_cfg.get("load_in_4bit", False))
        load_in_8bit = bool(model_cfg.get("load_in_8bit", False))
        if load_in_4bit and load_in_8bit:
            raise ValueError("model.load_in_4bit and model.load_in_8bit cannot both be true.")
        if load_in_8bit:
            raise ValueError("Unsloth training path currently supports 4-bit or full precision in this script.")
        dtype = _parse_torch_dtype(model_cfg.get("torch_dtype"))

        data_cfg = cfg["data"]
        if "train_path" not in data_cfg:
            raise ValueError("Missing required config field: data.train_path")
        train_path = resolve_path(data_cfg["train_path"], dirs["data"])
        eval_path = resolve_path(data_cfg.get("eval_path", ""), dirs["data"]) if data_cfg.get("eval_path") else None
        _validate_paths(train_path, eval_path)
        _validate_dataset_family(model_family, train_path, eval_path)
        logger.info("train_path=%s eval_path=%s", train_path, eval_path)

        with timer(logger, "load_dataset"):
            ds_train = load_dataset("json", data_files=str(train_path), split="train")
            ds_eval = load_dataset("json", data_files=str(eval_path), split="train") if eval_path else None

        tr_cfg = cfg["training"]
        max_seq_length = int(tr_cfg.get("max_seq_len", tr_cfg.get("max_seq_length", 2048)))

        with timer(logger, "load_unsloth_model_tokenizer"):
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )
            tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

        with timer(logger, "tokenize_dataset"):
            train_cols = list(ds_train.column_names)
            ds_train = ds_train.map(
                _tokenize_response_only_example,
                batched=False,
                remove_columns=train_cols,
                num_proc=1,
                fn_kwargs={"tokenizer": tokenizer, "max_length": max_seq_length},
            )
            ds_train = ds_train.filter(
                _keep_valid_tokenized_example,
                batched=False,
                num_proc=1,
                fn_kwargs={"max_length": max_seq_length},
            )
            if ds_eval is not None:
                eval_cols = list(ds_eval.column_names)
                ds_eval = ds_eval.map(
                    _tokenize_response_only_example,
                    batched=False,
                    remove_columns=eval_cols,
                    num_proc=1,
                    fn_kwargs={"tokenizer": tokenizer, "max_length": max_seq_length},
                )
                ds_eval = ds_eval.filter(
                    _keep_valid_tokenized_example,
                    batched=False,
                    num_proc=1,
                    fn_kwargs={"max_length": max_seq_length},
                )

        train_rows = len(ds_train)
        eval_rows = len(ds_eval) if ds_eval is not None else 0
        if train_rows == 0:
            raise ValueError("No valid training samples after response-only tokenization.")
        sample_labels = ds_train[0]["labels"]
        sample_input_ids = ds_train[0]["input_ids"]
        supervised_tokens = sum(1 for x in sample_labels if x != -100)
        masked_tokens = sum(1 for x in sample_labels if x == -100)
        print("DEBUG SAMPLE LENGTHS:")
        print(len(sample_input_ids))
        print(len(sample_labels))
        print("DEBUG SUPERVISED TOKENS:", supervised_tokens)
        print("DEBUG MASKED TOKENS:", masked_tokens)
        if supervised_tokens == 0:
            raise ValueError("Response-only tokenization produced zero supervised label tokens on the first sample.")
        if len(sample_input_ids) != max_seq_length or len(sample_labels) != max_seq_length:
            raise ValueError(
                f"Tokenized length mismatch: expected max_seq_length={max_seq_length}, "
                f"got input_ids={len(sample_input_ids)}, labels={len(sample_labels)}."
            )
        logger.info("tokenized_train_rows=%s tokenized_eval_rows=%s", train_rows, eval_rows)
        logger.info(
            "debug_sample_input_ids_len=%s debug_sample_labels_len=%s supervised_tokens=%s masked_tokens=%s",
            len(sample_input_ids),
            len(sample_labels),
            supervised_tokens,
            masked_tokens,
        )

        lora_cfg = cfg["lora"]
        if not bool(lora_cfg.get("enabled", True)):
            raise ValueError("This Unsloth script expects LoRA enabled.")
        peft_sig = inspect.signature(FastLanguageModel.get_peft_model).parameters
        peft_kwargs: dict[str, Any] = {
            "r": int(lora_cfg["r"]),
            "lora_alpha": int(lora_cfg["alpha"]),
            "lora_dropout": float(lora_cfg["dropout"]),
            "target_modules": lora_cfg.get("target_modules"),
        }
        if "bias" in peft_sig:
            peft_kwargs["bias"] = "none"
        if "use_gradient_checkpointing" in peft_sig:
            peft_kwargs["use_gradient_checkpointing"] = "unsloth"
        model = FastLanguageModel.get_peft_model(model, **peft_kwargs)

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
        logging_steps = int(tr_cfg.get("logging_steps", 10))
        save_steps = int(tr_cfg.get("save_steps", max(50, logging_steps)))
        seed = int(tr_cfg.get("seed", 42))
        eval_strategy = "steps" if ds_eval is not None else "no"
        eval_steps = save_steps if ds_eval is not None else None
        save_strategy = str(tr_cfg.get("save_strategy", "steps"))
        learning_rate = float(tr_cfg["learning_rate"])
        timestamp = datetime.now(timezone.utc).isoformat()

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
            "remove_unused_columns": False,
        }
        if "evaluation_strategy" in ta_params:
            train_args_kwargs["evaluation_strategy"] = eval_strategy
        elif "eval_strategy" in ta_params:
            train_args_kwargs["eval_strategy"] = eval_strategy
        if eval_steps is not None:
            train_args_kwargs["eval_steps"] = eval_steps
        train_args = TrainingArguments(**train_args_kwargs)

        sft_sig = inspect.signature(SFTTrainer.__init__).parameters
        sft_kwargs: dict[str, Any] = {
            "model": model,
            "args": train_args,
            "train_dataset": ds_train,
            "eval_dataset": ds_eval,
        }
        if "tokenizer" in sft_sig:
            sft_kwargs["tokenizer"] = tokenizer
        if "processing_class" in sft_sig:
            sft_kwargs["processing_class"] = tokenizer
        if "dataset_text_field" in sft_sig:
            sft_kwargs["dataset_text_field"] = None
        if "max_seq_length" in sft_sig:
            sft_kwargs["max_seq_length"] = max_seq_length
        if "packing" in sft_sig:
            sft_kwargs["packing"] = False
        sft_kwargs["dataset_num_proc"] = 1
        trainer = SFTTrainer(**sft_kwargs)

        total_params, trainable_params, trainable_pct = _count_parameters(model)
        meta_path = output_dir / "train_run_metadata.json"
        run_meta = {
            "project_task": "multilingual next-utterance generation",
            "script": "06_train_sft_unsloth.py",
            "config_path": str(Path(args.config).resolve()),
            "timestamp": timestamp,
            "base_model_name": base,
            "model_family": model_family,
            "unsloth_chat_template": chat_template,
            "output_dir": str(output_dir),
            "train_dataset_path": str(train_path),
            "eval_dataset_path": str(eval_path) if eval_path else None,
            "train_sample_count": train_rows,
            "eval_sample_count": eval_rows,
            "seed": seed,
            "learning_rate": learning_rate,
            "num_train_epochs": num_epochs,
            "max_steps": max_steps,
            "per_device_train_batch_size": per_device_bs,
            "per_device_eval_batch_size": per_device_eval_bs,
            "gradient_accumulation_steps": grad_accum,
            "effective_train_batch_size": effective_train_batch_size,
            "max_seq_length": max_seq_length,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "eval_steps": eval_steps,
            "save_strategy": _to_jsonable(train_args.save_strategy),
            "eval_strategy": _to_jsonable(
                getattr(train_args, "evaluation_strategy", getattr(train_args, "eval_strategy", None))
            ),
            "bf16": bool(bf16),
            "fp16": bool(fp16),
            "load_in_4bit": load_in_4bit,
            "lora_enabled": True,
            "lora_r": int(lora_cfg["r"]),
            "lora_alpha": int(lora_cfg["alpha"]),
            "lora_dropout": float(lora_cfg["dropout"]),
            "target_modules": lora_cfg.get("target_modules"),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": round(trainable_pct, 6),
            "response_only_loss": True,
            "final_adapter_path": None,
            "tokenizer_saved_path": None,
            "training_completed": False,
        }
        save_json(meta_path, run_meta)
        logger.info("train_metadata_path(initial)=%s", meta_path)

        with timer(logger, "train"):
            trainer.train()

        out_dir = output_dir / "lora_adapter"
        out_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        logger.info("adapter_saved_to=%s tokenizer_saved_to=%s", out_dir, out_dir)

        run_meta["final_adapter_path"] = str(out_dir)
        run_meta["tokenizer_saved_path"] = str(out_dir)
        run_meta["training_completed"] = True
        save_json(meta_path, run_meta)
        logger.info("train_metadata_path(final)=%s", meta_path)

        banner(logger, "Step 06: Done", char="-")
    except Exception:
        logger.exception("Step 06 (unsloth) failed")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

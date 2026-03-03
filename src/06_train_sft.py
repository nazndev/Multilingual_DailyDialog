import argparse
import sys
import traceback
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
        base = get_env("BASE_MODEL") or cfg["base_model"]
        data_cfg = cfg["data"]
        train_path = resolve_path(data_cfg["train_path"], dirs["data"])
        eval_path = resolve_path(data_cfg.get("eval_path", ""), dirs["data"]) if data_cfg.get("eval_path") else None
        logger.info("train_path=%s eval_path=%s base_model=%s", train_path, eval_path, base)

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

            # NOTE: Avoid device_map="auto" during training on Apple Silicon.
            # It can offload parameters and leave some on the "meta" device, which
            # then crashes backward on MPS (device mismatch).
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

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

        prec = cfg.get("precision", {})
        bf16 = (prec.get("bf16") is True) or (prec.get("bf16") == "auto" and torch.cuda.is_available())
        fp16 = not bf16 and ((prec.get("fp16") is True) or (prec.get("fp16") == "auto" and torch.cuda.is_available()))

        tr_cfg = cfg["training"]
        output_dir = resolve_path(tr_cfg["output_dir"], dirs["outputs"])
        max_steps = int(tr_cfg["max_steps"]) if tr_cfg.get("max_steps") else -1
        num_epochs = float(tr_cfg.get("num_train_epochs", 1))
        logger.info("output_dir=%s max_steps=%s num_train_epochs=%s", output_dir, max_steps, num_epochs)

        max_length = int(cfg["training"].get("max_seq_len", 2048))
        args_tr = SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=int(tr_cfg["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(tr_cfg["gradient_accumulation_steps"]),
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            learning_rate=float(tr_cfg["learning_rate"]),
            warmup_ratio=float(tr_cfg["warmup_ratio"]),
            logging_steps=int(tr_cfg["logging_steps"]),
            save_steps=int(tr_cfg["save_steps"]),
            eval_strategy="steps" if ds_eval else "no",
            eval_steps=int(tr_cfg["save_steps"]) if ds_eval else None,
            report_to=[],
            bf16=bf16,
            fp16=fp16,
            seed=int(tr_cfg.get("seed", 42)),
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
        banner(logger, "Step 06: Done", char="-")
    except Exception:
        logger.exception("Step 06 failed")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Fase 5 — Fine-tuning de un transformer ligero como regresor.

- Modelo: distilbert-base-uncased (66 M parámetros) con una sola salida:
  AutoModelForSequenceClassification(num_labels=1, problem_type="regression").
- Target escalado: y / 20 durante el entrenamiento; las predicciones se multiplican por
  20 y se recortan a [0, 20].
- Pérdida: MSE (por defecto) o Huber (--loss huber, Pregunta 3).
- Longitud: los CVs limpios tienen a lo sumo ~160 tokens de DistilBERT, así que
  max_length=256 cubre el CV completo (se reporta cuántos se truncarían). No hace falta
  head+tail ni chunking: todas las fechas llegan al modelo.
- Learning rate elegido en validación entre 5e-5, 3e-5 y 2e-5 (corridas con --tag, 15
  épocas, paciencia 3): 5e-5 ganó (MAE 0.165) y seguía mejorando en la última época,
  porque el MAE de validación es ruidoso mientras el LR es alto y converge cuando el
  scheduler lineal lo baja a 0. Configuración final: 20 épocas, early stopping sobre el
  MAE de validación con paciencia 5.

Uso:
    python -m src.train_bert
    python -m src.train_bert --loss huber
    python -m src.train_bert --text-col text_raw
"""

import argparse
import math
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback, Trainer,
                          TrainingArguments)

from src.evaluation import model_name, run_and_save
from src.utils import (BERT_DIR, DEFAULT_TEXT_COL, EXP_CAP, METRICS_DIR, SEED, TEXT_COLS,
                       load_splits, save_json, set_seed)

MODEL = "distilbert-base-uncased"


class HuberTrainer(Trainer):
    """Trainer con pérdida Huber (delta en la escala del target, 0.05 = 1 año)."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        preds = outputs.logits.squeeze(-1)
        loss = torch.nn.functional.huber_loss(preds, labels.to(preds.dtype), delta=0.05)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.clip(np.asarray(preds).reshape(-1) * EXP_CAP, 0, EXP_CAP)
    return {"mae": float(np.mean(np.abs(preds - np.asarray(labels) * EXP_CAP)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--text-col", default=DEFAULT_TEXT_COL, choices=TEXT_COLS)
    parser.add_argument("--loss", default="mse", choices=["mse", "huber"])
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="", help="sufijo para corridas de búsqueda (p. ej. lr3e-5)")
    args = parser.parse_args()
    set_seed(args.seed)

    base = "bert" if args.loss == "mse" else "bert_huber"
    name = model_name(f"{base}_{args.tag}" if args.tag else base, args.text_col)
    out_dir = BERT_DIR / name
    frames = load_splits()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    lengths = [len(tokenizer(t)["input_ids"]) for t in frames["train"][args.text_col]]
    truncated = int(np.sum(np.array(lengths) > args.max_length))
    print(f"Tokens por CV (train): mediana {np.median(lengths):.0f}, máx {max(lengths)}; "
          f"truncados con max_length={args.max_length}: {truncated}")

    def to_dataset(split: str) -> Dataset:
        df = frames[split]
        enc = tokenizer(df[args.text_col].tolist(), truncation=True, max_length=args.max_length)
        enc["labels"] = (df["y"].to_numpy(dtype=np.float32) / EXP_CAP).tolist()
        return Dataset.from_dict(dict(enc))

    train_ds, val_ds = to_dataset("train"), to_dataset("val")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, problem_type="regression")

    # Warmup del 10 % de los pasos (warmup_ratio ya no existe en transformers recientes)
    warmup_steps = int(0.1 * math.ceil(len(train_ds) / args.batch_size) * args.epochs)
    # Checkpoints en el disco local (no en Drive: lo borrado en Drive va a la papelera y
    # sigue ocupando cuota) y sin estado del optimizador. Solo el mejor modelo va a out_dir.
    ckpt_dir = Path(tempfile.gettempdir()) / "bert_checkpoints" / name
    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        save_only_model=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=args.epochs,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        logging_strategy="epoch",
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        report_to="none",
    )
    trainer_cls = HuberTrainer if args.loss == "huber" else Trainer
    trainer = trainer_cls(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    save_json(trainer.state.log_history, METRICS_DIR / f"bert_log_{name}.json")
    best_checkpoint = trainer.state.best_model_checkpoint
    shutil.rmtree(ckpt_dir, ignore_errors=True)  # el mejor ya está en out_dir

    device = model.device
    model.eval()

    @torch.no_grad()
    def predict(texts: list[str]) -> np.ndarray:
        preds = []
        for i in range(0, len(texts), 64):
            enc = tokenizer(texts[i:i + 64], truncation=True, max_length=args.max_length,
                            padding=True, return_tensors="pt").to(device)
            preds.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
        return np.concatenate(preds) * EXP_CAP

    hardware = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    epochs_run = max((e.get("epoch", 0) for e in trainer.state.log_history), default=0)
    run_and_save(name, predict, frames, args.text_col, train_seconds, extra={
        "base_model": args.model, "loss": args.loss, "learning_rate": args.lr,
        "batch_size": args.batch_size, "max_epochs": args.epochs, "patience": args.patience,
        "epochs_run": epochs_run, "best_checkpoint": best_checkpoint,
        "max_length": args.max_length, "tokens_median": float(np.median(lengths)),
        "tokens_max": int(max(lengths)), "n_truncated_train": truncated,
        "seed": args.seed, "hardware": hardware,
    })


if __name__ == "__main__":
    main()

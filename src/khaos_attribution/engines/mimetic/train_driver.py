"""mimetic-a0's trainer, run in the engine's own environment: stable-audio-tools' train.py, step for step, from its
library's factories (the script is not part of the installed package), without its preview renders or logger, with
the step limit kept and every step's loss written to khaos_losses.csv for the session to follow."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LOSSES_CSV = "khaos_losses.csv"
METADATA_MODULE = '''from pathlib import Path


def get_custom_metadata(info, audio):
    return {"prompt": Path(info["path"]).with_suffix(".txt").read_text(encoding="utf-8").strip()}
'''


def model_config(base_config: dict, args) -> dict:
    """Mick's config with the training section his release leaves out: the adapter, the rate, no previews."""
    cfg = json.loads(json.dumps(base_config))
    cfg["training"] = {
        "learning_rate": float(args.lr),
        "lora_config": {"rank": int(args.rank), "alpha": int(args.alpha), "adapter_type": args.adapter_type},
        "demo": {"demo_every": 10 ** 9, "demo_steps": 1, "num_demos": 1,
                 "demo_cond": [{"prompt": "", "seconds_start": 0, "seconds_total": 1}], "demo_cfg_scales": [1]},
    }
    return cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mimetic_train_driver")
    p.add_argument("--clips", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--checkpoint-every", type=int, required=True)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--adapter-type", default="lora")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    import pytorch_lightning as pl  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from stable_audio_tools.data.dataset import create_dataloader_from_config  # noqa: PLC0415
    from stable_audio_tools.models.factory import create_model_from_config  # noqa: PLC0415
    from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict  # noqa: PLC0415
    from stable_audio_tools.training import create_training_wrapper_from_config  # noqa: PLC0415

    from khaos_attribution.engines.mimetic.loading import model_files, pick_device  # noqa: PLC0415

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed, workers=True)
    config_path, weights_path = model_files()
    cfg = model_config(json.loads(Path(config_path).read_text(encoding="utf-8")), args)
    (save_dir / "model_config.json").write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    metadata = save_dir / "khaos_metadata.py"
    metadata.write_text(METADATA_MODULE, encoding="utf-8")
    # Every pair counts in every epoch, as Khaos's epochs-to-steps arithmetic assumes.
    dataset = {"dataset_type": "audio_dir", "random_crop": True, "drop_last": False,
               "datasets": [{"id": "khaos", "path": str(Path(args.clips).resolve()), "custom_metadata_module": str(metadata)}]}
    (save_dir / "dataset_config.json").write_text(json.dumps(dataset, indent=1) + "\n", encoding="utf-8")

    model = create_model_from_config(cfg)
    tokenizers = {key: (cond.tokenizer, cond.max_length) for key, cond in model.conditioner.conditioners.items()
                  if hasattr(cond, "tokenizer") and hasattr(cond, "max_length")}
    loader = create_dataloader_from_config(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                           sample_rate=cfg["sample_rate"], sample_size=cfg["sample_size"],
                                           audio_channels=cfg.get("audio_channels", 2), tokenizers=tokenizers or None)
    copy_state_dict(model, load_ckpt_state_dict(str(weights_path)))
    wrapper = create_training_wrapper_from_config(cfg, model)

    losses = (save_dir / LOSSES_CSV).open("a", newline="", encoding="utf-8")
    writer = csv.writer(losses)
    if losses.tell() == 0:
        writer.writerow(["step", "loss"])

    class LossRows(pl.Callback):
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            writer.writerow([trainer.global_step, f"{float(loss):.6f}"])
            losses.flush()

    device = pick_device(args.device)
    trainer = pl.Trainer(
        devices=1, accelerator="cpu" if device == "cpu" else "gpu", precision="32", max_steps=args.steps,
        max_epochs=10 ** 7, logger=False, log_every_n_steps=1, num_sanity_val_steps=0, enable_progress_bar=False,
        default_root_dir=str(save_dir), reload_dataloaders_every_n_epochs=0,
        callbacks=[pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=str(save_dir),
                                                save_top_k=-1, save_last=False),
                   LossRows()],
    )
    try:
        trainer.fit(wrapper, loader)
        # The last step is always kept, so a run whose length is not a multiple of the cadence still exports.
        if trainer.global_step % args.checkpoint_every:
            trainer.save_checkpoint(str(save_dir / f"epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt"))
    finally:
        losses.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Small, explicit SmolVLA fine-tuning with episode-disjoint validation."""

import json
from pathlib import Path

from common import digest, episode_split, output_directory, write_json
from config import PolicyConfig
from dataset_tools import open_dataset
from smolvla import SmolVLA


def optimizer_update(policy, preprocessor, batch: dict, optimizer, max_grad_norm: float) -> float:
    import torch

    policy.train()
    optimizer.zero_grad(set_to_none=True)
    loss, _ = policy(preprocessor(batch))
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("Training returned a non-finite scalar loss")
    loss.backward()
    parameters = [p for p in policy.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise ValueError("Training returned non-finite gradients")
    optimizer.step()
    return float(loss.detach())


def save_checkpoint(model, optimizer, folder: Path, step: int, dataset_fingerprint: str) -> None:
    import torch

    folder.mkdir(parents=True, exist_ok=False)
    # Config/tokenizer paths must remain portable after the runtime has been deleted.
    config = model.policy.config
    local_vlm = config.vlm_model_name
    try:
        config.vlm_model_name = model.metadata["vlm_repo"]
        model.policy.save_pretrained(folder)
    finally:
        config.vlm_model_name = local_vlm
    model.pre.save_pretrained(folder, config_filename="policy_preprocessor.json")
    model.post.save_pretrained(folder, config_filename="policy_postprocessor.json")
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "dataset_fingerprint": dataset_fingerprint,
        },
        folder / "training_state.pt",
    )


def train(config: dict, model=None) -> Path:
    import torch
    from lerobot.utils.collate import lerobot_collate_fn

    train_ids, validation_ids = episode_split(config["dataset"])
    for key in ["steps", "batch_size", "validation_every"]:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["learning_rate"] <= 0 or config["max_grad_norm"] <= 0:
        raise ValueError("Learning rate and gradient norm must be positive")
    torch.manual_seed(config["seed"])
    model = model if model is not None else SmolVLA(PolicyConfig(**config["policy"]), 256)
    if not model.policy.config.train_expert_only or not model.policy.config.freeze_vision_encoder:
        raise ValueError("The T4 introductory recipe requires a frozen VLM and train_expert_only")
    # The saved checkpoint's normalization is deliberately kept fixed in this curriculum.
    # Both public and collected data must obey the same LIBERO units/camera convention.
    preview = open_dataset(config["dataset"], train_ids)
    times = [index / preview.fps for index in model.policy.config.action_delta_indices]
    training = open_dataset(config["dataset"], train_ids, times)
    validation = open_dataset(config["dataset"], validation_ids, times)
    parameters = [p for p in model.policy.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("The model has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"])
    fingerprint = digest(config["dataset"])
    first_step = 0
    if config.get("resume_from"):
        resume = Path(config["resume_from"])
        if Path(config["policy"]["path"]).resolve() != resume.resolve():
            raise ValueError("Resume must load the model and optimizer from the same checkpoint")
        state = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=True)
        if state["dataset_fingerprint"] != fingerprint:
            raise ValueError("Cannot resume with a different dataset or episode split")
        optimizer.load_state_dict(state["optimizer"])
        first_step = state["step"]
    if first_step >= config["steps"]:
        raise ValueError("steps must exceed the saved training step")
    folder = output_directory("finetuning")
    report = {
        "status": "running",
        "config": config,
        "dataset_fingerprint": fingerprint,
        "normalization": "fixed pretrained checkpoint processors",
        "trainable_parameters": sum(p.numel() for p in parameters),
        "dataset_fps": training.fps,
        "action_chunk_frames": len(times),
        "validation_scope": "Fixed held-out minibatch, not full-dataset loss",
        "starting_step": first_step,
        "model": model.metadata,
    }
    write_json(folder / "training.json", report)
    try:
        with (folder / "losses.jsonl").open("w") as log:
            for step in range(first_step, config["steps"]):
                # Stateless sampling/noise seeds make a resumed step use the same batch.
                generator = torch.Generator().manual_seed(config["seed"] + step)
                indices = torch.randint(
                    len(training), (config["batch_size"],), generator=generator
                ).tolist()
                batch = lerobot_collate_fn([training[index] for index in indices])
                torch.manual_seed(config["seed"] + step)
                loss = optimizer_update(
                    model.policy, model.pre, batch, optimizer, config["max_grad_norm"]
                )
                row = {"step": step + 1, "train_loss": loss}
                if (step + 1) % config["validation_every"] == 0 or step + 1 == config["steps"]:
                    model.policy.eval()
                    # Reuse the same held-out samples and noise for comparable validation losses.
                    torch.manual_seed(config["seed"] + 1000000)
                    held_out = lerobot_collate_fn(
                        [validation[i % len(validation)] for i in range(config["batch_size"])]
                    )
                    with torch.inference_mode():
                        value, _ = model.policy(model.pre(held_out))
                    if not torch.isfinite(value):
                        raise ValueError("Validation loss is not finite")
                    row["validation_loss"] = float(value)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(row, flush=True)
        checkpoint = folder / "pretrained_model"
        save_checkpoint(model, optimizer, checkpoint, config["steps"], fingerprint)
        report.update(
            status="completed", checkpoint=str(checkpoint), completed_steps=config["steps"]
        )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(folder / "training.json", report)
    print(f"Results: {folder}", flush=True)
    return folder

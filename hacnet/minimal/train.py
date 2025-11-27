"""
Minimal training loop for HAC-Net.

Uses a simple filesystem dataset loader and trains the HACNetMinimal model with
waveform + spectral reconstruction losses plus boundary regularizers.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import soundfile as sf
from tqdm import tqdm

from hacnet.minimal.models.config import HACNetMinimalConfig
from hacnet.minimal.models.hacnet_minimal import HACNetMinimal
from hacnet.minimal.utils.losses import (
    waveform_loss,
    spectral_loss,
    SpectralConfig,
    refractory_loss,
)
from hacnet.minimal.utils.monitor import TrainingMonitor


class AudioDataset(Dataset):
    def __init__(self, manifest_path: Path, sample_rate: int, max_duration: float = 5.0, limit: int | None = None):
        with open(manifest_path) as f:
            items = json.load(f)
        if limit is not None:
            items = items[:limit]
        self.items = items
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        metadata = self.items[idx]
        audio_path = metadata["audio_path"]
        samples, sr = sf.read(audio_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        waveform = torch.from_numpy(samples).unsqueeze(0)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform[:, : self.max_samples]
        length = waveform.shape[1]
        return waveform.squeeze(0), length


def collate_fn(batch):
    waveforms, lengths = zip(*batch)
    max_len = max(lengths)
    padded = torch.zeros(len(batch), max_len)
    for i, waveform in enumerate(waveforms):
        padded[i, : waveform.shape[0]] = waveform
    lengths = torch.tensor(lengths, dtype=torch.long)
    return padded, lengths


def train(args):
    if args.device:
        device = torch.device(args.device)
    elif not args.cpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif not args.cpu and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")
    config = HACNetMinimalConfig(hidden_dim=args.hidden_dim)
    if args.router_lags:
        lags = [int(x) for x in args.router_lags.split(",") if x.strip()]
        if lags:
            config.router.lags = tuple(lags)
    config.router.threshold_mode = args.router_threshold
    config.router.fixed_threshold = args.router_fixed_threshold
    config.router.rolling_percentile = args.router_rolling_percentile
    config.router.rolling_window = args.router_rolling_window
    config.router.pre_smooth_window = args.router_pre_smooth_window
    config.router.smooth_window = args.router_smooth_window
    config.router.peak_height = args.router_peak_height
    config.router.peak_distance = args.router_peak_distance
    config.router.peak_prominence = args.router_peak_prominence
    config.router.peak_rel_height = args.router_peak_rel_height
    model = HACNetMinimal(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    dataset = AudioDataset(
        Path(args.manifest),
        sample_rate=config.encoder.sample_rate,
        limit=args.max_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    steps_per_epoch = max(1, len(dataloader))

    monitor_dir = Path(args.monitor_dir) if args.monitor_dir else Path(args.output_dir)
    spectral_cfg = SpectralConfig()
    monitor = TrainingMonitor(
        output_dir=monitor_dir,
        sample_rate=config.encoder.sample_rate,
        plot_interval=args.plot_interval,
    )
    monitor.save_audio = args.save_audio

    wandb_run = None
    wandb_module = None
    if args.wandb:
        try:
            import wandb as wandb_module
        except ImportError:
            print("[WARN] wandb requested but not installed. Run `pip install wandb` to enable logging.")
        else:
            wandb_config = {
                "hidden_dim": config.hidden_dim,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "router_lags": config.router.lags,
                "router_threshold": config.router.threshold_mode,
                "max_samples": args.max_samples,
            }
            if args.wandb_api_key:
                wandb_module.login(key=args.wandb_api_key)
            elif "WANDB_API_KEY" in os.environ:
                wandb_module.login(key=os.environ["WANDB_API_KEY"])
            wandb_run = wandb_module.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                config=wandb_config,
            )

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for step, (waveforms, lengths) in enumerate(pbar):
            waveforms = waveforms.to(device)
            lengths = lengths.to(device)

            output = model(waveforms, lengths=lengths)
            target = waveforms.unsqueeze(1)
            pred = output.reconstruction
            min_len = min(pred.shape[-1], target.shape[-1])
            pred = pred[..., :min_len]
            target = target[..., :min_len]

            loss_wave = waveform_loss(pred, target)
            loss_spec = spectral_loss(pred, target, spectral_cfg)
            loss_refr = refractory_loss(output.router.change_score, config.router.refractory, output.router.boundary_mask)

            loss = loss_wave + args.spectral_weight * loss_spec + args.refr_weight * loss_refr

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            true_ratio = output.router.boundary_mask.float().mean().item()
            mean_prob = output.router.boundary_prob[..., -1].float().mean().item()

            monitor.log_metrics(
                epoch,
                step,
                {
                    "loss": float(loss.item()),
                    "loss_wave": float(loss_wave.item()),
                    "loss_spec": float(loss_spec.item()),
                    "loss_refr": float(loss_refr.item()),
                    "boundary_fraction": true_ratio,
                    "mean_prob": mean_prob,
                },
            )
            if wandb_run is not None:
                global_step = epoch * steps_per_epoch + step
                wandb_module.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/loss_wave": float(loss_wave.item()),
                        "train/loss_spec": float(loss_spec.item()),
                        "train/loss_refr": float(loss_refr.item()),
                        "train/boundary_fraction": true_ratio,
                        "train/mean_prob": mean_prob,
                    },
                    step=global_step,
                )

            if step % args.log_interval == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    wave=f"{loss_wave.item():.3f}",
                    spec=f"{loss_spec.item():.3f}",
                    F=f"{true_ratio:.3f}",
                )

        torch.save(model.state_dict(), Path(args.output_dir) / f"hacnet_minimal_epoch{epoch}.pt")
        monitor.flush()
        monitor.plot_losses()
        if args.plot_interval > 0 and epoch % args.plot_interval == 0:
            idx = min(args.example_index, target.shape[0] - 1)
            monitor.plot_spectrogram(
                epoch,
                target[idx, 0].detach().cpu(),
                pred[idx, 0].detach().cpu(),
            )
            monitor.plot_boundaries(
                epoch,
                output.router.change_score.detach().cpu(),
                output.router.boundary_mask.detach().cpu(),
                output.hop_seconds,
                waveforms[idx].detach().cpu(),
                sample_index=idx,
                reconstructed=pred[idx, 0].detach().cpu(),
            )
            monitor.plot_embedding_drift(
                epoch,
                output.frame_embeddings.detach().cpu(),
                output.unit_embeddings.detach().cpu(),
                output.chunker.mask.detach().cpu(),
                sample_index=idx,
            )
            if wandb_run is not None:
                global_step = (epoch + 1) * steps_per_epoch
                specs_path = monitor.output_dir / f"spectrogram_epoch{epoch}.png"
                bounds_path = monitor.output_dir / f"boundaries_epoch{epoch}.png"
                log_payload = {}
                if specs_path.exists():
                    log_payload["plots/spectrogram"] = wandb_module.Image(str(specs_path))
                if bounds_path.exists():
                    log_payload["plots/boundaries"] = wandb_module.Image(str(bounds_path))
                example_audio = waveforms[idx].detach().cpu().numpy()
                recon_audio = pred[idx, 0].detach().cpu().numpy()
                log_payload["audio/original"] = wandb_module.Audio(
                    example_audio,
                    sample_rate=config.encoder.sample_rate,
                    caption=f"epoch_{epoch}_original",
                )
                log_payload["audio/reconstruction"] = wandb_module.Audio(
                    recon_audio,
                    sample_rate=config.encoder.sample_rate,
                    caption=f"epoch_{epoch}_reconstruction",
                )
                wandb_module.log(log_payload, step=global_step)

        if wandb_run is not None:
            global_step = (epoch + 1) * steps_per_epoch
            specs_path = monitor.output_dir / f"spectrogram_epoch{epoch}.png"
            bounds_path = monitor.output_dir / f"boundaries_epoch{epoch}.png"
            log_payload = {}
            if specs_path.exists():
                log_payload["plots/spectrogram"] = wandb_module.Image(str(specs_path))
            if bounds_path.exists():
                log_payload["plots/boundaries"] = wandb_module.Image(str(bounds_path))
            example_audio = waveforms[idx].detach().cpu().numpy()
            recon_audio = pred[idx, 0].detach().cpu().numpy()
            log_payload["audio/original"] = wandb_module.Audio(
                example_audio,
                sample_rate=config.encoder.sample_rate,
                caption=f"epoch_{epoch}_original",
            )
            log_payload["audio/reconstruction"] = wandb_module.Audio(
                recon_audio,
                sample_rate=config.encoder.sample_rate,
                caption=f"epoch_{epoch}_reconstruction",
            )
            wandb_module.log(log_payload, step=global_step)

    monitor.plot_embedding_drift(
        epoch,
        output.frame_embeddings.detach().cpu(),
        output.unit_embeddings.detach().cpu(),
        output.chunker.mask.detach().cpu(),
        sample_index=idx,
    )

    if wandb_run is not None:
        wandb_run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Train minimal HAC-Net model.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to dataset manifest JSON.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save checkpoints.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--spectral-weight", type=float, default=1.0)
    parser.add_argument("--refr-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training even if GPU is available.")
    parser.add_argument("--device", type=str, default=None, help="Explicit device to use (e.g., cuda, mps, cpu).")
    parser.add_argument("--num-workers", type=int, default=14, help="DataLoader worker processes.")
    parser.add_argument("--monitor-dir", type=str, default=None, help="Directory for CSV/plots (defaults to output-dir).")
    parser.add_argument("--plot-interval", type=int, default=1, help="Epoch interval for spectrogram/boundary plots.")
    parser.add_argument("--example-index", type=int, default=0, help="Batch index used for qualitative plots.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of training clips.")
    parser.add_argument("--router-lags", type=str, default=None,
                        help="Comma-separated lag list overriding router config (e.g., '1,2,4').")
    parser.add_argument("--router-threshold", type=str,
                        choices=["fixed", "rolling", "derivative", "smooth_derivative", "smooth_derivative_peaks"],
                        default="fixed",
                        help="Router thresholding strategy (fixed or rolling percentile).")
    parser.add_argument("--router-fixed-threshold", type=float, default=0.5,
                        help="Threshold value when using fixed mode.")
    parser.add_argument("--router-rolling-percentile", type=float, default=90.0,
                        help="Percentile (0-100) used for rolling threshold mode.")
    parser.add_argument("--router-rolling-window", type=int, default=200,
                        help="Rolling threshold window size in frames (≈window / 100 for seconds).")
    parser.add_argument("--router-pre-smooth-window", type=int, default=5,
                        help="Window used before derivative modes (smooth_derivative*).")
    parser.add_argument("--router-smooth-window", type=int, default=7,
                        help="Window used for smoothing derivative outputs.")
    parser.add_argument("--router-peak-height", type=float, default=1e-3,
                        help="Minimum peak height for smooth_derivative_peaks.")
    parser.add_argument("--router-peak-distance", type=int, default=5,
                        help="Minimum peak spacing (frames) for smooth_derivative_peaks.")
    parser.add_argument("--router-peak-prominence", type=float, default=5e-4,
                        help="Minimum peak prominence for smooth_derivative_peaks.")
    parser.add_argument("--router-peak-rel-height", type=float, default=0.5,
                        help="Relative height (fraction of max) for smooth_derivative_peaks.")
    parser.add_argument("--save-audio", action="store_true", dest="save_audio",
                        help="Enable saving original/reconstructed WAV files when plotting.")
    parser.add_argument("--no-save-audio", action="store_false", dest="save_audio",
                        help="Disable saving original/reconstructed WAV files.")
    parser.set_defaults(save_audio=True)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="hacnet-minimal",
                        help="W&B project name (default: hacnet-minimal).")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="Optional W&B run name.")
    parser.add_argument("--wandb-entity", type=str, default="qbacquele-phd",
                        help="W&B entity (default: qbacquele-phd).")
    parser.add_argument("--wandb-api-key", type=str, default=None,
                        help="W&B API key (optional; falls back to env).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)

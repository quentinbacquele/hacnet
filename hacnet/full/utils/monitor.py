"""
Training monitor utilities: CSV logging, loss plots, and qualitative figures.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import soundfile as sf
import torch


@dataclass
class TrainingMonitor:
    output_dir: Path
    sample_rate: int
    plot_interval: int = 1
    history: List[Dict[str, float]] = field(default_factory=list)
    _prev_frame_snapshot: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _prev_unit_snapshot: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    save_audio: bool = False

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "training_metrics.csv"

    def log_metrics(self, epoch: int, step: int, metrics: Dict[str, float]) -> None:
        entry = {"epoch": epoch, "step": step}
        entry.update(metrics)
        self.history.append(entry)

    def flush(self) -> None:
        if not self.history:
            return
        fieldnames = sorted(self.history[0].keys())
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)

    def plot_losses(self) -> None:
        if not self.history:
            return
        steps = [h["epoch"] + h["step"] * 1e-3 for h in self.history]
        plt.figure(figsize=(8, 4))
        plt.plot(steps, [h["loss"] for h in self.history], label="total")
        for key in ("loss_wave", "loss_spec", "loss_refr"):
            if key in self.history[0]:
                plt.plot(steps, [h[key] for h in self.history], label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "loss_curves.png")
        plt.close()

    def plot_spectrogram(
        self,
        epoch: int,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> None:
        if epoch % self.plot_interval != 0:
            return
        orig_spec = self._stft_mag(original)
        recon_spec = self._stft_mag(reconstructed)
        duration = original.shape[-1] / self.sample_rate
        freq_max = self.sample_rate / 2.0
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        extent = [0, duration, 0, freq_max]
        axes[0].imshow(orig_spec, aspect="auto", origin="lower", extent=extent)
        axes[0].set_title("Original")
        axes[1].imshow(recon_spec, aspect="auto", origin="lower", extent=extent)
        axes[1].set_title("Reconstructed")
        for ax in axes:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
        plt.tight_layout()
        plt.savefig(self.output_dir / f"spectrogram_epoch{epoch}.png")
        plt.close(fig)

    def plot_boundaries(
        self,
        epoch: int,
        change_scores: torch.Tensor,
        boundary_mask: torch.Tensor,
        hop_seconds: float,
        example_waveform: torch.Tensor,
        sample_index: int = 0,
        reconstructed: Optional[torch.Tensor] = None,
    ) -> None:
        if epoch % self.plot_interval != 0:
            return
        batch_size = change_scores.shape[0]
        if batch_size == 0:
            return
        sample_index = max(0, min(sample_index, batch_size - 1))
        change = change_scores[sample_index].detach().cpu()
        boundaries = torch.nonzero(boundary_mask[sample_index]).squeeze(-1).cpu()
        frame_times = (
            torch.arange(change.shape[0], dtype=torch.float32) * float(hop_seconds)
        )

        full_spec = self._stft_mag(example_waveform)
        duration = example_waveform.shape[-1] / self.sample_rate
        freq_max = self.sample_rate / 2.0
        extent = [0, duration, 0, freq_max]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.patch.set_alpha(0.0)
        bg_ax = ax.twinx()
        bg_ax.imshow(
            full_spec,
            aspect="auto",
            origin="lower",
            extent=extent,
            alpha=0.5,
            cmap="magma",
        )
        bg_ax.set_ylim(0, freq_max)
        bg_ax.set_ylabel("Frequency (Hz)")
        bg_ax.set_yticks([])
        bg_ax.patch.set_alpha(0.0)

        ax.plot(frame_times.numpy(), change.numpy(), label="change score", color="white")
        if boundaries.numel() > 0:
            ax.scatter(
                frame_times[boundaries].numpy(),
                change[boundaries].numpy(),
                color="red",
                s=12,
                label="boundaries",
                zorder=3,
            )
            for t_val in frame_times[boundaries].numpy():
                ax.axvline(t_val, color="red", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Change score")
        ax.legend(loc="upper right")
        ax.set_title("Change score with spectrogram background")
        plt.tight_layout()
        plt.savefig(self.output_dir / f"boundaries_epoch{epoch}.png")
        plt.close(fig)

        if self.save_audio and reconstructed is not None:
            orig = example_waveform.cpu().numpy()
            recon = reconstructed.cpu().numpy()
            sf.write(self.output_dir / f"original_epoch{epoch}.wav", orig, self.sample_rate)
            sf.write(self.output_dir / f"reconstruction_epoch{epoch}.wav", recon, self.sample_rate)

    def plot_embedding_drift(
        self,
        epoch: int,
        frame_embeddings: torch.Tensor,
        unit_embeddings: torch.Tensor,
        unit_mask: torch.Tensor,
        sample_index: int = 0,
    ) -> None:
        """
        Visualize embedding snapshots and their per-epoch delta for a single example.
        """
        if epoch % self.plot_interval != 0:
            return
        if frame_embeddings.shape[0] == 0:
            return
        sample_index = max(0, min(sample_index, frame_embeddings.shape[0] - 1))

        frame_snapshot = frame_embeddings[sample_index]
        valid_units = unit_mask[sample_index].bool()
        if valid_units.any():
            unit_snapshot = unit_embeddings[sample_index][valid_units]
        else:
            unit_snapshot = unit_embeddings[sample_index][:0]

        frame_prev = self._prev_frame_snapshot
        unit_prev = self._prev_unit_snapshot
        frame_diff = self._compute_overlap_diff(frame_snapshot, frame_prev)
        unit_diff = self._compute_overlap_diff(unit_snapshot, unit_prev)

        fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
        self._plot_embedding_panel(
            axes[0, 0],
            frame_prev,
            "Frames (prev)",
            cmap="viridis",
        )
        self._plot_embedding_panel(
            axes[0, 1],
            frame_snapshot,
            f"Frames (epoch {epoch})",
            cmap="viridis",
        )
        self._plot_embedding_panel(
            axes[0, 2],
            frame_diff,
            "Frames Δ",
            cmap="RdBu",
            center_zero=True,
        )
        self._plot_embedding_panel(
            axes[1, 0],
            unit_prev,
            "Units (prev)",
            cmap="viridis",
        )
        self._plot_embedding_panel(
            axes[1, 1],
            unit_snapshot,
            f"Units (epoch {epoch})",
            cmap="viridis",
        )
        self._plot_embedding_panel(
            axes[1, 2],
            unit_diff,
            "Units Δ",
            cmap="RdBu",
            center_zero=True,
        )
        plt.savefig(self.output_dir / f"embedding_drift_epoch{epoch}.png")
        plt.close(fig)

        self._prev_frame_snapshot = frame_snapshot.detach().cpu()
        self._prev_unit_snapshot = unit_snapshot.detach().cpu()

    def _compute_overlap_diff(
        self,
        current: Optional[torch.Tensor],
        previous: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if current is None or current.numel() == 0 or previous is None or previous.numel() == 0:
            return None
        dim = current.shape[-1]
        if previous.shape[-1] != dim:
            return None
        min_len = min(current.shape[0], previous.shape[0])
        if min_len == 0:
            return None
        return current[:min_len] - previous[:min_len]

    def _plot_embedding_panel(
        self,
        axis: plt.Axes,
        tensor: Optional[torch.Tensor],
        title: str,
        cmap: str,
        center_zero: bool = False,
    ) -> None:
        axis.set_title(title)
        if tensor is None or tensor.numel() == 0:
            axis.axis("off")
            return
        data = tensor.detach().cpu().T.numpy()
        if center_zero:
            vmax = max(float(abs(data).max()), 1e-6)
            im = axis.imshow(data, aspect="auto", origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = axis.imshow(data, aspect="auto", origin="lower", cmap=cmap)
        axis.set_xlabel("Time / Units")
        axis.set_ylabel("Channels")
        plt.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    def _stft_mag(self, waveform: torch.Tensor) -> torch.Tensor:
        n_fft = 512
        hop = 128
        win = torch.hann_window(n_fft)
        spec = torch.stft(
            waveform.unsqueeze(0),
            n_fft=n_fft,
            hop_length=hop,
            window=win.to(waveform.device),
            return_complex=True,
        )
        return spec.abs().log1p().squeeze(0).cpu().numpy()

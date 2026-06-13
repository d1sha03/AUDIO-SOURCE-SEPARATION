"""
audio_utils.py
Audio DSP utilities and matplotlib visualisations for the GSN app.
All plot functions return None gracefully when matplotlib is unavailable.
"""

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np

# ---------------------------------------------------------------------------
# Safe optional imports
# ---------------------------------------------------------------------------
MPL_OK     = False
LIBROSA_OK = False
SF_OK      = False
TA_OK      = False

try:
    import soundfile as sf
    SF_OK = True
except Exception:
    pass

try:
    import torch
    import torchaudio
    TA_OK = True
except Exception:
    pass

try:
    import librosa
    import librosa.display
    LIBROSA_OK = True
except Exception:
    pass

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as ticker
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    MPL_OK = True
except Exception:
    pass

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
BG_DARK    = "#0B0F19"
BG_CARD    = "#141926"
BG_SURFACE = "#1C2133"
C_TEXT     = "#D1D5DB"
C_MUTED    = "#6B7280"
C_BORDER   = "#2A3042"
C_ACCENT   = "#6366F1"     # indigo
C_TEAL     = "#14B8A6"
C_AMBER    = "#F59E0B"
C_ROSE     = "#F43F5E"
C_SKY      = "#38BDF8"
C_SLATE    = "#94A3B8"

SPEC_CMAP = None
if MPL_OK:
    SPEC_CMAP = LinearSegmentedColormap.from_list(
        "gsn",
        ["#0B0F19", "#1E3A5F", "#14B8A6", "#F59E0B", "#F43F5E", "#F1F5F9"],
        N=512,
    )

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _apply_theme(fig, axes):
    if not MPL_OK or fig is None:
        return
    fig.patch.set_facecolor(BG_DARK)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        if ax is None:
            continue
        ax.set_facecolor(BG_CARD)
        ax.tick_params(colors=C_MUTED, labelsize=8)
        ax.xaxis.label.set_color(C_MUTED)
        ax.yaxis.label.set_color(C_MUTED)
        ax.title.set_color(C_TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(C_BORDER)


def _add_cbar(fig, ax, img, label="dB"):
    if not MPL_OK:
        return
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="3%", pad=0.06)
    cb  = fig.colorbar(img, cax=cax)
    cb.set_label(label, fontsize=8, color=C_MUTED)
    cb.ax.tick_params(labelsize=7, colors=C_MUTED)


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_audio_numpy(path: str):
    """Returns (np.ndarray [C, T], sample_rate)."""
    if TA_OK:
        w, sr = torchaudio.load(path)
        return w.numpy(), int(sr)
    if SF_OK:
        d, sr = sf.read(path, always_2d=True)
        return d.T.astype(np.float32), int(sr)
    raise ImportError("Install torchaudio or soundfile.")


def to_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=0) if audio.ndim == 2 else audio.flatten()


def numpy_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    if not SF_OK:
        raise ImportError("soundfile required.")
    buf = io.BytesIO()
    out = audio.T if audio.ndim == 2 else audio
    sf.write(buf, np.clip(out, -1.0, 1.0), sr,
             format="WAV", subtype="PCM_24")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    r = ref.flatten()
    e = est.flatten()
    n = min(len(r), len(e))
    r, e = r[:n] - r[:n].mean(), e[:n] - e[:n].mean()
    a = np.dot(e, r) / (np.dot(r, r) + 1e-8)
    t = a * r
    noise = e - t
    return float(10 * np.log10(
        (np.sum(t**2) + 1e-8) / (np.sum(noise**2) + 1e-8)
    ))


def compute_rms_db(audio: np.ndarray) -> float:
    rms = np.sqrt(np.mean(audio.flatten()**2) + 1e-12)
    return float(20 * np.log10(rms + 1e-12))


def compute_spectral_centroid(audio: np.ndarray, sr: int) -> float:
    if not LIBROSA_OK:
        return 0.0
    sc = librosa.feature.spectral_centroid(y=to_mono(audio), sr=sr)[0]
    return float(sc.mean())


# ---------------------------------------------------------------------------
# Waveform envelope helper
# ---------------------------------------------------------------------------

def _envelope(sig, hop=None):
    if hop is None:
        hop = max(1, len(sig) // 600)
    env = np.array([
        np.max(np.abs(sig[i:i + hop]))
        for i in range(0, max(1, len(sig) - hop), hop)
    ])
    return env


# ---------------------------------------------------------------------------
# Downsample helper — fixes OverflowError
# ---------------------------------------------------------------------------

def _downsample(sig, sr, max_points=5000):
    """Reduce signal to max_points for safe matplotlib rendering."""
    if len(sig) <= max_points:
        return sig
    factor = len(sig) // max_points
    return sig[::factor]


def _time_axis(sig, sr):
    return np.linspace(0, len(sig) / sr, len(sig))


# ---------------------------------------------------------------------------
# Plot: Waveform Comparison
# ---------------------------------------------------------------------------

def plot_waveform_comparison(mixture, separated, sr,
                              source_name="vocals"):
    if not MPL_OK:
        return None

    mix_m = to_mono(mixture)
    sep_m = to_mono(separated)

    # Downsample before plotting
    mix_d = _downsample(mix_m, sr)
    sep_d = _downsample(sep_m, sr)
    t_mix = np.linspace(0, len(mix_m) / sr, len(mix_d))
    t_sep = np.linspace(0, len(sep_m) / sr, len(sep_d))

    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=False)
    fig.suptitle("Waveform Comparison", fontsize=13,
                 color=C_TEXT, fontweight="600")

    panels = [
        (t_mix, mix_d, mix_m, C_SKY,  "Mixture"),
        (t_sep, sep_d, sep_m, C_ROSE, f"Separated — {source_name.title()}"),
    ]

    for ax, (t, sig_d, sig_orig, col, title) in zip(axes, panels):
        ax.plot(t, sig_d, color=col, lw=0.6, alpha=0.85)
        ax.fill_between(t, sig_d, alpha=0.08, color=col)

        # Envelope on downsampled
        env = _envelope(sig_d)
        te  = np.linspace(0, t[-1], len(env))
        ax.plot(te,  env, color=col, lw=1.0, alpha=0.35, ls="--")
        ax.plot(te, -env, color=col, lw=1.0, alpha=0.35, ls="--")

        rms = compute_rms_db(sig_orig)
        ax.text(0.008, 0.92, title, transform=ax.transAxes,
                fontsize=9.5, fontweight="600", color=col, va="top")
        ax.text(0.992, 0.92, f"RMS {rms:.1f} dBFS",
                transform=ax.transAxes, fontsize=8,
                color=C_MUTED, ha="right", va="top")
        ax.set_ylim(-1.08, 1.08)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _apply_theme(fig, axes)
    return fig


# ---------------------------------------------------------------------------
# Plot: Spectrogram Comparison
# ---------------------------------------------------------------------------

def plot_spectrogram_comparison(mixture, separated, sr,
                                 n_fft=2048, hop_length=512,
                                 source_name="vocals"):
    if not MPL_OK or not LIBROSA_OK:
        return None

    mix_m = to_mono(mixture)
    sep_m = to_mono(separated)
    dur   = len(mix_m) / sr

    mix_s = librosa.stft(mix_m, n_fft=n_fft, hop_length=hop_length)
    sep_s = librosa.stft(sep_m, n_fft=n_fft, hop_length=hop_length)
    ref   = np.abs(mix_s).max()
    mix_db = librosa.amplitude_to_db(np.abs(mix_s), ref=ref)
    sep_db = librosa.amplitude_to_db(np.abs(sep_s), ref=ref)
    dif_db = np.abs(sep_db - mix_db)
    score  = compute_si_sdr(mix_m, sep_m)

    panels = [
        (mix_db, "Mixture",                        SPEC_CMAP,  -80,  0),
        (sep_db, f"{source_name.title()} (GSN)",   SPEC_CMAP,  -80,  0),
        (dif_db, "Absolute Difference",            "RdYlGn_r",   0, 40),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Log-Magnitude Spectrogram", fontsize=13,
                 color=C_TEXT, fontweight="600")

    for i, (ax, (data, title, cmap, vmin, vmax)) in enumerate(
        zip(axes, panels)
    ):
        img = ax.imshow(
            data, origin="lower", aspect="auto",
            cmap=cmap, vmin=vmin, vmax=vmax,
            extent=[0, dur, 0, sr / 2 / 1000],
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=10, fontweight="600")
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Frequency (kHz)", fontsize=9)
        _add_cbar(fig, ax, img)

        if i == 1:
            for fhz in [220, 440, 880]:
                ax.axhline(y=fhz / 1000, color="white",
                           lw=0.6, ls=":", alpha=0.3)
            ax.text(
                0.97, 0.97, f"SI-SDR {score:.2f} dB",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, fontweight="700", color=C_ACCENT,
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor=BG_CARD, edgecolor=C_ACCENT,
                          alpha=0.92),
            )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _apply_theme(fig, axes)
    return fig


# ---------------------------------------------------------------------------
# Plot: Full GSN comparison (6 panels)
# ---------------------------------------------------------------------------

def plot_gsn_comparison(mixture, separated, sr,
                         n_fft=2048, hop_length=512,
                         source_name="vocals"):
    if not MPL_OK or not LIBROSA_OK:
        return None

    mix_m = to_mono(mixture)
    sep_m = to_mono(separated)
    n     = min(len(mix_m), len(sep_m))
    mix_m = mix_m[:n]
    sep_m = sep_m[:n]
    dif_m = sep_m - mix_m
    dur   = n / sr

    # STFTs
    mix_s  = librosa.stft(mix_m,  n_fft=n_fft, hop_length=hop_length)
    sep_s  = librosa.stft(sep_m,  n_fft=n_fft, hop_length=hop_length)
    dif_s  = librosa.stft(dif_m,  n_fft=n_fft, hop_length=hop_length)
    ref    = np.abs(mix_s).max()
    mix_db = librosa.amplitude_to_db(np.abs(mix_s), ref=ref)
    sep_db = librosa.amplitude_to_db(np.abs(sep_s), ref=ref)
    dif_db = librosa.amplitude_to_db(np.abs(dif_s), ref=ref)
    si_val = compute_si_sdr(mix_m, sep_m)

    # Downsample waveforms for plotting
    mix_d = _downsample(mix_m, sr)
    sep_d = _downsample(sep_m, sr)
    dif_d = _downsample(dif_m, sr)

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(
        f"GSN Hybrid Separation — {source_name.title()}  "
        f"[SI-SDR: {si_val:+.2f} dB]",
        fontsize=13, color=C_TEXT, fontweight="700", y=1.01,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.36)

    waves = [
        (mix_d, mix_m, C_SKY,   f"Mixture   [RMS {compute_rms_db(mix_m):.1f} dB]"),
        (sep_d, sep_m, C_ROSE,  f"{source_name.title()} GSN   "
                                 f"[RMS {compute_rms_db(sep_m):.1f} dB]"),
        (dif_d, dif_m, C_AMBER, "Residual  (Sep - Mix)"),
    ]
    specs = [
        (mix_db, SPEC_CMAP,  "Mixture STFT",                -80, 0),
        (sep_db, SPEC_CMAP,  f"{source_name.title()} STFT", -80, 0),
        (dif_db, "RdYlGn_r", "Residual STFT",               -80, 0),
    ]

    all_ax = []

    # Row 1 — waveforms (downsampled)
    for col, (sig_d, sig_orig, clr, title) in enumerate(waves):
        ax = fig.add_subplot(gs[0, col])
        all_ax.append(ax)

        t = np.linspace(0, dur, len(sig_d))
        ax.plot(t, sig_d, color=clr, lw=0.5, alpha=0.88)
        ax.fill_between(t, sig_d, alpha=0.08, color=clr)

        env = _envelope(sig_d)
        te  = np.linspace(0, dur, len(env))
        ax.plot(te,  env, color=clr, lw=0.9, alpha=0.35, ls="--")
        ax.plot(te, -env, color=clr, lw=0.9, alpha=0.35, ls="--")

        ax.set_title(title, fontsize=9, fontweight="600", color=clr)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylim(-1.08, 1.08)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

        if col == 1:
            ax.text(
                0.98, 0.96, f"SI-SDR: {si_val:+.2f} dB",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, fontweight="700", color=C_ACCENT,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=BG_CARD, edgecolor=C_ACCENT,
                          alpha=0.92),
            )

    # Row 2 — spectrograms
    for col, (data, cmap, title, vmin, vmax) in enumerate(specs):
        ax = fig.add_subplot(gs[1, col])
        all_ax.append(ax)

        img = ax.imshow(
            data, origin="lower", aspect="auto",
            cmap=cmap, vmin=vmin, vmax=vmax,
            extent=[0, dur, 0, sr / 2 / 1000],
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=9, fontweight="600")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Freq (kHz)", fontsize=8)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        _add_cbar(fig, ax, img)

        if col == 1:
            for fhz in [220, 440, 880]:
                ax.axhline(y=fhz / 1000, color="white",
                           lw=0.6, ls=":", alpha=0.3)

    _apply_theme(fig, all_ax)
    fig.patch.set_facecolor(BG_DARK)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ---------------------------------------------------------------------------
# Plot: Metrics radar
# ---------------------------------------------------------------------------

def plot_metrics_radar(scores: dict):
    if not MPL_OK:
        return None

    labels = list(scores.keys())
    values = list(scores.values())
    n      = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5),
                            subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_CARD)

    # Use plot not scatter — avoids marker overflow
    ax.plot(angles, values, "-", lw=2.2, color=C_ACCENT)
    ax.fill(angles, values, alpha=0.18, color=C_ACCENT)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, color=C_TEXT)
    ax.set_ylim(0, 8)
    ax.set_yticks([2, 4, 6, 8])
    ax.set_yticklabels(["2", "4", "6", "8"], fontsize=7, color=C_MUTED)
    ax.tick_params(colors=C_MUTED)
    ax.grid(color=C_BORDER, alpha=0.6)
    ax.spines["polar"].set_color(C_BORDER)
    ax.set_title("SI-SDR Across Phases (dB)", fontsize=11,
                 color=C_TEXT, pad=18, fontweight="600")
    return fig


# ---------------------------------------------------------------------------
# Plot: Stage timing
# ---------------------------------------------------------------------------

def plot_stage_timing(stage_times: dict):
    if not MPL_OK:
        return None

    stages = {
        "Load Audio":        stage_times.get("load",   0.0),
        "CLAP Routing":      stage_times.get("clap",   0.0),
        "Demucs Separation": stage_times.get("demucs", 0.0),
        "GSN Refinement":    stage_times.get("gsn",    0.0),
    }
    labels = list(stages.keys())
    times  = list(stages.values())
    colors = [C_SKY, C_TEAL, C_AMBER, C_ROSE]

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(labels, times, color=colors,
                   edgecolor=BG_DARK, linewidth=0.8,
                   height=0.5, alpha=0.88)

    for bar, t in zip(bars, times):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{t:.2f}s", va="center", fontsize=9,
            fontweight="600", color=C_TEXT,
        )

    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_title("Pipeline Stage Timing", fontsize=11,
                 fontweight="600", color=C_TEXT)
    ax.set_xlim(0, max(times + [0.1]) * 1.3)
    ax.invert_yaxis()
    _apply_theme(fig, [ax])
    fig.patch.set_facecolor(BG_DARK)
    plt.tight_layout()
    return fig
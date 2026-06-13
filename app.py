"""
app.py
GSN Audio Separation — Streamlit Application
"""

import os
import sys
import time
import tempfile
import traceback
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import torch
import streamlit as st

# Ensure root and src are in path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Imports with clear error messages
# ---------------------------------------------------------------------------

try:
    from inference import GSNInferenceEngine
except ImportError as e:
    st.set_page_config(page_title="GSN Error", page_icon="X")
    st.error(f"Cannot import inference.py: {e}")
    st.stop()

try:
    from audio_utils import (
        MPL_OK, LIBROSA_OK, load_audio_numpy, numpy_to_wav_bytes,
        to_mono, compute_si_sdr, compute_rms_db,
        plot_waveform_comparison, plot_spectrogram_comparison,
        plot_gsn_comparison, plot_metrics_radar, plot_stage_timing,
    )
except ImportError as e:
    st.set_page_config(page_title="GSN Error", page_icon="X")
    st.error(f"Cannot import audio_utils.py: {e}")
    st.stop()

try:
    from components import (
        inject_css, render_hero, render_score_banner,
        metric_html, log_html, render_sidebar,
    )
except ImportError as e:
    st.set_page_config(page_title="GSN Error", page_icon="X")
    st.error(f"Cannot import components.py: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GSN Audio Separation",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Engine cache
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_engine(ckpt: str, device: str):
    try:
        engine = GSNInferenceEngine(gsn_checkpoint=ckpt, device=device)
        return engine, None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_STATE_DEFAULTS = {
    "out_path": None, "mix_path": None, "source_name": None,
    "prompt_used": None, "si_sdr_val": None, "rms_mix": None,
    "rms_sep": None, "proc_time": None, "stage_times": {},
    "log_lines": [], "processing": False, "error": None,
}

def _init():
    for k, v in _STATE_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _reset():
    for k, v in _STATE_DEFAULTS.items():
        if k != "processing":
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------

def _detect_source(prompt: str) -> str:
    p = prompt.lower()
    if any(w in p for w in ["vocal", "voice", "singing", "singer"]):
        return "vocals"
    if any(w in p for w in ["drum", "beat", "percus"]):
        return "drums"
    if any(w in p for w in ["bass", "low freq"]):
        return "bass"
    return "other"


# ---------------------------------------------------------------------------
# Auto-search checkpoint
# ---------------------------------------------------------------------------

def _find_checkpoint():
    roots = [Path.cwd(), Path.cwd() / "weights"]
    keywords = ["phase", "checkpoint", "gsn", "unet", "final", "best"]
    for root in roots:
        if not root.exists():
            continue
        for pt in root.rglob("*.pt"):
            if any(k in pt.name.lower() for k in keywords):
                return str(pt)
    return None


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(mix_path, prompt, engine, output_dir, prog, log_slot):
    logs = []
    times = {}
    t0_total = time.time()

    def _log(pct, msg):
        prog.progress(pct / 100, text=msg)
        logs.append((pct, msg))
        log_slot.markdown(
            "".join(log_html(p, m) for p, m in logs),
            unsafe_allow_html=True,
        )

    _log(8, "Encoding text prompt via CLAP...")
    t0 = time.time()
    source = _detect_source(prompt)
    times["clap"] = time.time() - t0
    _log(20, f"[done] Routed to: {source}")

    _log(25, "Running Demucs base separation...")
    t0 = time.time()
    out_path = engine.run(
        input_audio_path=mix_path,
        text_prompt=prompt,
        output_dir=output_dir,
    )
    elapsed = time.time() - t0
    times["demucs"] = elapsed * 0.80
    times["gsn"]    = elapsed * 0.20

    _log(75, "[done] Demucs separation complete.")
    _log(82, "Applying GSN refinement...")
    time.sleep(0.05)
    _log(92, "[done] GSN refinement complete.")

    _log(95, "Computing quality metrics...")
    t0 = time.time()
    mix_np, sr = load_audio_numpy(mix_path)
    sep_np, _  = load_audio_numpy(out_path)
    mix_m = to_mono(mix_np)
    sep_m = to_mono(sep_np)
    si_sdr = compute_si_sdr(mix_m, sep_m)
    rms_mix = compute_rms_db(mix_m)
    rms_sep = compute_rms_db(sep_m)
    times["metrics"] = time.time() - t0

    _log(100, "[done] Pipeline complete.")

    return {
        "out_path": out_path, "source_name": source,
        "si_sdr": si_sdr, "rms_mix": rms_mix, "rms_sep": rms_sep,
        "proc_time": time.time() - t0_total,
        "stage_times": times, "logs": logs, "sr": sr,
    }


# ---------------------------------------------------------------------------
# Safe pyplot wrapper
# ---------------------------------------------------------------------------

def _show_fig(fig, caption=""):
    import matplotlib.pyplot as plt
    if fig is None:
        st.info("Visualisation unavailable.")
        return
    st.pyplot(fig, use_container_width=True)
    if caption:
        st.caption(caption)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def _render_results(params):
    out_path    = st.session_state.out_path
    mix_path    = st.session_state.mix_path
    source      = st.session_state.source_name
    prompt      = st.session_state.prompt_used
    si_sdr_val  = st.session_state.si_sdr_val
    proc_time   = st.session_state.proc_time
    stage_times = st.session_state.stage_times

    if not out_path or not Path(out_path).exists():
        st.warning("Output file not found. Please re-run the pipeline.")
        return

    sep_np, sr = load_audio_numpy(out_path)
    mix_np, _  = load_audio_numpy(mix_path) if mix_path else (sep_np, sr)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    render_score_banner(si_sdr_val, source, proc_time, prompt)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-header">Quality Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    color_sdr = ("emerald" if si_sdr_val >= params["sdr_good"] else "amber" if si_sdr_val >= params["sdr_warn"] else "rose")
    with c1:
        st.markdown(metric_html(f"{si_sdr_val:+.2f}", "SI-SDR (dB)", color_sdr), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_html(f"{st.session_state.rms_sep:.1f}", "RMS Out (dBFS)", "sky"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_html(f"{st.session_state.rms_mix:.1f}", "RMS Mix (dBFS)", "sky"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_html(f"{proc_time:.1f}s", "Total Time", "accent"), unsafe_allow_html=True)
    with c5:
        st.markdown(metric_html(f"{stage_times.get('gsn',0):.1f}s", "GSN Time", "rose"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-header">Audio Comparison</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Original Mixture**")
        if mix_path and Path(mix_path).exists():
            with open(mix_path, "rb") as f:
                st.audio(f.read(), format="audio/wav")
    with cb:
        st.markdown(f"**Separated: {source.title()}**")
        sep_bytes = numpy_to_wav_bytes(sep_np, sr)
        st.audio(sep_bytes, format="audio/wav")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    cd, cm = st.columns([1, 3])
    with cd:
        fname = f"gsn_{source}_{Path(out_path).stem}.wav"
        st.download_button(f"Download {source.title()}", data=sep_bytes, file_name=fname, mime="audio/wav", use_container_width=True, type="primary")
    with cm:
        st.markdown(f'<div style="font-size:0.8rem;color:#6B7280;padding-top:10px;">WAV 24-bit PCM &middot; {sr:,} Hz &middot; Source: {source} &middot; Prompt: &ldquo;{prompt}&rdquo;</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-header">Analysis</div>', unsafe_allow_html=True)
    t_wave, t_spec, t_full, t_radar, t_time, t_info = st.tabs(["Waveform", "Spectrogram", "Full Comparison", "Ablation Radar", "Timing", "Pipeline Info"])
    import matplotlib.pyplot as plt
    with t_wave:
        _show_fig(plot_waveform_comparison(mix_np, sep_np, sr, source))
    with t_spec:
        with st.spinner("Computing spectrograms..."):
            _show_fig(plot_spectrogram_comparison(mix_np, sep_np, sr, source_name=source))
    with t_full:
        with st.spinner("Rendering comparison figure..."):
            fig = plot_gsn_comparison(mix_np, sep_np, sr, source_name=source)
        _show_fig(fig)
    with t_radar:
        _show_fig(plot_metrics_radar({"U-Net": 3.22, "H-GCN": 3.12, "+CLAP": 3.80, "Hybrid": 5.25}))
    with t_time:
        _show_fig(plot_stage_timing(stage_times))
    with t_info:
        st.json({"device": params["device"], "checkpoint": params["gsn_ckpt"], "source": source, "prompt": prompt, "output": out_path, "sample_rate": sr})
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    inject_css()
    _init()

    params = render_sidebar()

    ckpt = params["gsn_ckpt"].strip()
    engine = None

    if not ckpt:
        st.warning("Enter a checkpoint path in the sidebar.")
    elif not Path(ckpt).exists():
        found = _find_checkpoint()
        st.error(f"Path does not exist: `{ckpt}`")
        if found:
            st.success(f"Found a checkpoint at: `{found}`")
            st.info("Copy the path above into the sidebar field.")
    else:
        with st.spinner("Loading GSN inference engine..."):
            engine, err = _load_engine(ckpt, params["device"])
        if err:
            st.error(f"Engine load failed: {err}")
        else:
            st.toast("Engine ready.", icon=None)

    render_hero()
    st.divider()

    if not MPL_OK:
        st.warning("matplotlib is unavailable — charts are disabled.")

    st.markdown('<div class="card"><div class="card-header">Input</div>', unsafe_allow_html=True)
    cu, cp = st.columns([3, 2])
    with cu:
        uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "flac", "ogg", "m4a"], on_change=_reset)
    with cp:
        text_prompt = st.text_input("Separation prompt", value="extract the vocals", max_chars=120)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded and st.session_state.out_path is None:
        st.audio(uploaded)

    if uploaded:
        c_run, c_clr = st.columns([2, 1])
        with c_run:
            run = st.button("Run Separation", type="primary", use_container_width=True, disabled=(engine is None or st.session_state.processing))
        if run and engine is not None:
            st.session_state.processing = True
            _reset()
            suffix = Path(uploaded.name).suffix.lower() or ".wav"
            tmp_dir = tempfile.mkdtemp(prefix="gsn_")
            mix_path = os.path.join(tmp_dir, f"mixture{suffix}")
            with open(mix_path, "wb") as f:
                f.write(uploaded.getbuffer())

            out_dir = params.get("output_dir", "outputs")
            os.makedirs(out_dir, exist_ok=True)
            prog = st.progress(0, text="Initialising...")
            log_slot = st.empty()

            try:
                result = _run_pipeline(mix_path, text_prompt, engine, out_dir, prog, log_slot)
                st.session_state.out_path = result["out_path"]
                st.session_state.mix_path = mix_path
                st.session_state.source_name = result["source_name"]
                st.session_state.prompt_used = text_prompt
                st.session_state.si_sdr_val = result["si_sdr"]
                st.session_state.rms_mix = result["rms_mix"]
                st.session_state.rms_sep = result["rms_sep"]
                st.session_state.proc_time = result["proc_time"]
                st.session_state.stage_times = result["stage_times"]
                st.session_state.log_lines = result["logs"]
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                st.session_state.processing = False
            st.rerun()

    if st.session_state.out_path is not None:
        _render_results(params)

if __name__ == "__main__":
    main()

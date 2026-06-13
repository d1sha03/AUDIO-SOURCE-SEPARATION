"""
components.py
Reusable Streamlit UI components with professional dark theme.
"""

import streamlit as st
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --bg:       #0B0F19;
        --card:     #141926;
        --surface:  #1C2133;
        --border:   #2A3042;
        --text:     #D1D5DB;
        --muted:    #6B7280;
        --dim:      #4B5563;
        --accent:   #6366F1;
        --teal:     #14B8A6;
        --amber:    #F59E0B;
        --rose:     #F43F5E;
        --sky:      #38BDF8;
        --emerald:  #10B981;
    }

    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background: var(--bg) !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        color: var(--text) !important;
    }

    [data-testid="stSidebar"] {
        background: var(--card) !important;
        border-right: 1px solid var(--border) !important;
    }

    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 18px;
    }

    .card-header {
        font-size: 0.92rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border);
        letter-spacing: 0.2px;
    }

    .hero-title {
        text-align: center;
        padding: 28px 0 8px;
    }
    .hero-title h1 {
        font-size: 2.2rem;
        font-weight: 800;
        color: #F1F5F9;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }
    .hero-title .accent { color: var(--accent); }
    .hero-title p {
        font-size: 0.92rem;
        color: var(--muted);
        max-width: 580px;
        margin: 0 auto;
        line-height: 1.6;
    }
    .hero-tags {
        display: flex; gap: 8px; justify-content: center;
        flex-wrap: wrap; margin-top: 12px;
    }
    .tag {
        font-size: 0.72rem; font-weight: 600;
        padding: 3px 10px; border-radius: 6px;
        letter-spacing: 0.3px;
    }
    .tag-accent  { background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.3); color: var(--accent); }
    .tag-teal    { background: rgba(20,184,166,0.12);  border: 1px solid rgba(20,184,166,0.3);  color: var(--teal); }
    .tag-sky     { background: rgba(56,189,248,0.12);  border: 1px solid rgba(56,189,248,0.3);  color: var(--sky); }
    .tag-rose    { background: rgba(244,63,94,0.12);   border: 1px solid rgba(244,63,94,0.3);   color: var(--rose); }

    .metric {
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        padding: 12px 16px; border-radius: 10px;
        min-width: 90px; text-align: center;
    }
    .metric .val { font-size: 1.6rem; font-weight: 800; line-height: 1.1; }
    .metric .lbl {
        font-size: 0.65rem; color: var(--muted); margin-top: 4px;
        font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase;
    }
    .m-accent  { background: rgba(99,102,241,0.10);  border: 1px solid rgba(99,102,241,0.25); }
    .m-teal    { background: rgba(20,184,166,0.10);   border: 1px solid rgba(20,184,166,0.25); }
    .m-sky     { background: rgba(56,189,248,0.10);   border: 1px solid rgba(56,189,248,0.25); }
    .m-amber   { background: rgba(245,158,11,0.10);   border: 1px solid rgba(245,158,11,0.25); }
    .m-rose    { background: rgba(244,63,94,0.10);    border: 1px solid rgba(244,63,94,0.25); }
    .m-emerald { background: rgba(16,185,129,0.10);   border: 1px solid rgba(16,185,129,0.25); }
    .m-accent .val  { color: var(--accent); }
    .m-teal .val    { color: var(--teal); }
    .m-sky .val     { color: var(--sky); }
    .m-amber .val   { color: var(--amber); }
    .m-rose .val    { color: var(--rose); }
    .m-emerald .val { color: var(--emerald); }

    .score-banner {
        background: linear-gradient(135deg,
            rgba(99,102,241,0.08) 0%,
            rgba(20,184,166,0.06) 100%);
        border: 1px solid rgba(99,102,241,0.22);
        border-radius: 14px;
        padding: 20px 24px;
        text-align: center;
        margin: 14px 0;
    }
    .score-banner .number {
        font-size: 3.0rem;
        font-weight: 900;
        line-height: 1.0;
    }
    .score-banner .label {
        font-size: 0.72rem;
        color: var(--muted);
        margin-top: 4px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .score-banner .detail {
        font-size: 0.82rem;
        color: var(--muted);
        margin-top: 8px;
    }

    .log-entry {
        padding: 6px 12px;
        border-radius: 6px;
        border-left: 3px solid var(--border);
        background: rgba(255,255,255,0.02);
        margin-bottom: 4px;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
        color: var(--text);
    }
    .log-done   { border-left-color: var(--emerald); color: var(--emerald); }
    .log-active { border-left-color: var(--amber);   color: var(--amber);
                  animation: pulse-anim 1.1s ease-in-out infinite; }
    .log-error  { border-left-color: var(--rose);    color: var(--rose); }
    @keyframes pulse-anim { 0%,100%{opacity:1} 50%{opacity:0.5} }

    [data-testid="stFileUploader"] section {
        background: rgba(99,102,241,0.03) !important;
        border: 1.5px dashed rgba(99,102,241,0.25) !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: rgba(99,102,241,0.50) !important;
    }

    [data-testid="stButton"] > button[kind="primary"] {
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 0.5rem 1.6rem !important;
        transition: opacity .15s !important;
    }
    [data-testid="stButton"] > button[kind="primary"]:hover {
        opacity: 0.88 !important;
    }
    [data-testid="stButton"] > button[kind="secondary"] {
        background: var(--surface) !important;
        color: var(--muted) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, var(--accent), var(--teal)) !important;
    }

    [data-testid="stTabs"] [role="tab"] {
        color: var(--muted) !important; font-weight: 500 !important;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--text) !important;
        border-color: var(--accent) !important;
        font-weight: 700 !important;
    }

    audio { width: 100%; border-radius: 8px; }
    hr { border-color: var(--border) !important; margin: 22px 0 !important; }

    ::-webkit-scrollbar       { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: #3B4252; border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

def render_hero():
    st.markdown("""
    <div class="hero-title">
        <h1>GSN <span class="accent">Audio Separation</span></h1>
        <p>
            Text-guided hybrid pipeline combining Demucs,
            CLAP semantic routing, and a Graph-based Separation Network
            for targeted vocal refinement.
        </p>
        <div class="hero-tags">
            <span class="tag tag-accent">SI-SDR 5.25 dB</span>
            <span class="tag tag-sky">Demucs htdemucs</span>
            <span class="tag tag-teal">CLAP Routing</span>
            <span class="tag tag-rose">GSNComplex U-Net</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Score banner
# ---------------------------------------------------------------------------

def render_score_banner(si_sdr, source, time_s, prompt):
    if si_sdr >= 4.5:
        color, grade = "#10B981", "Excellent"
    elif si_sdr >= 3.0:
        color, grade = "#F59E0B", "Good"
    elif si_sdr >= 1.0:
        color, grade = "#F43F5E", "Fair"
    else:
        color, grade = "#6B7280", "Poor"

    st.markdown(f"""
    <div class="score-banner">
        <div class="number" style="color:{color};">{si_sdr:+.2f} dB</div>
        <div class="label">SI-SDR Quality Score &middot; {grade}</div>
        <div class="detail">
            Source: <strong>{source.upper()}</strong> &middot;
            Prompt: <em>&ldquo;{prompt}&rdquo;</em> &middot;
            Processed in <strong>{time_s:.1f}s</strong> &middot;
            SOTA ref: ~7.50 dB
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Metric badge
# ---------------------------------------------------------------------------

def metric_html(value, label, color="accent"):
    return f"""
    <div class="metric m-{color}">
        <div class="val">{value}</div>
        <div class="lbl">{label}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Log entry
# ---------------------------------------------------------------------------

def log_html(pct, message):
    if "complete" in message.lower() or message.startswith("[done]"):
        cls = "log-entry log-done"
    elif "error" in message.lower() or "fail" in message.lower():
        cls = "log-entry log-error"
    else:
        cls = "log-entry log-active"
    return f'<div class="{cls}">[{pct:3d}%] {message}</div>'


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(ckpt_default=""):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:14px 0 6px;">
            <div style="font-size:1.1rem;font-weight:800;
                         color:#D1D5DB;">GSN Parameters</div>
            <div style="font-size:0.72rem;color:#6B7280;margin-top:2px;">
                Hybrid Separation Engine
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Device ────────────────────────────────────────────
        import torch
        cuda_ok = torch.cuda.is_available()
        device = st.selectbox(
            "Compute Device",
            options=["cuda", "cpu"] if cuda_ok else ["cpu"],
            help="CUDA recommended for Demucs inference.",
        )
        if cuda_ok:
            props = torch.cuda.get_device_properties(0)
            vram  = props.total_memory / (1024**3)
            st.caption(f"{props.name}  ({vram:.1f} GB)")
        else:
            st.caption("No CUDA detected.")

        st.divider()

        # ── Checkpoint Selection ──────────────────────────────
        gsn_ckpt = _render_checkpoint_selector(ckpt_default)

        st.divider()

        # ── Quality thresholds ────────────────────────────────
        sdr_warn = st.slider("Warn below (dB)", 0.0, 5.0, 2.0, 0.5)
        sdr_good = st.slider("Good above (dB)", 0.0, 8.0, 4.5, 0.5)

        st.divider()

        # ── Advanced ──────────────────────────────────────────
        with st.expander("Advanced"):
            output_dir = st.text_input("Output directory", value="outputs")

        st.divider()

        # ── Info card ─────────────────────────────────────────
        st.markdown("""
        <div style="font-size:0.72rem;color:#4B5563;line-height:1.6;">
            <strong style="color:#6B7280;">Pipeline</strong><br>
            Demucs &rarr; CLAP routing &rarr; GSNComplex<br><br>
            <strong style="color:#6B7280;">Dataset</strong><br>
            MUSDB18 &middot; 100 train / 36 test<br><br>
            <strong style="color:#6B7280;">Results</strong><br>
            U-Net 3.22 &middot; H-GCN 3.12<br>
            +CLAP 3.80 &middot;
            <span style="color:#6366F1;font-weight:700;">
                Hybrid 5.25 dB
            </span>
        </div>
        """, unsafe_allow_html=True)

    return {
        "device":     device,
        "gsn_ckpt":   gsn_ckpt,
        "sdr_warn":   sdr_warn,
        "sdr_good":   sdr_good,
        "output_dir": output_dir if "output_dir" in dir() else "outputs",
    }


# ---------------------------------------------------------------------------
# Checkpoint scanner and selector
# ---------------------------------------------------------------------------

def _scan_checkpoints():
    """
    Scan common project directories for .pt checkpoint files.
    Returns a dict:  { "display_name": "full_path", ... }
    """
    import os
    from pathlib import Path

    app_dir = Path(__file__).resolve().parent
    search_roots = [
        app_dir,
        app_dir.parent,
        Path(os.getcwd()),
        Path(os.getcwd()).parent,
    ]

    # Deduplicate roots
    seen_roots = set()
    unique_roots = []
    for r in search_roots:
        resolved = str(r.resolve())
        if resolved not in seen_roots and r.exists():
            seen_roots.add(resolved)
            unique_roots.append(r)

    found = {}
    seen_files = set()

    for root in unique_roots:
        try:
            for pt_file in root.rglob("*.pt"):
                resolved = str(pt_file.resolve())
                if resolved in seen_files:
                    continue
                seen_files.add(resolved)

                # Build a readable display name
                display = _make_display_name(pt_file, root)
                found[display] = resolved
        except PermissionError:
            continue

    return found


def _make_display_name(pt_path, root):
    """
    Create a human-readable label for a checkpoint file.
    Example: "Phase 1 Best  (best_model.pt, 42.3 MB)"
    """
    from pathlib import Path

    name   = pt_path.stem.lower()
    parent = pt_path.parent.name.lower()
    size   = pt_path.stat().st_size / (1024 * 1024)

    # Detect phase from folder or filename
    phase = ""
    for keyword, label in [
        ("phase1", "Phase 1"),
        ("phase2", "Phase 2"),
        ("phase3", "Phase 3"),
        ("phase4", "Phase 4"),
        ("phase5", "Phase 5"),
        ("phase6", "Phase 6"),
    ]:
        if keyword in name or keyword in parent:
            phase = label
            break

    # Detect model type from filename
    model_type = ""
    type_map = {
        "best":       "Best",
        "final":      "Final",
        "checkpoint":  "Checkpoint",
        "unet":       "U-Net",
        "gsn":        "GSN",
        "clap":       "CLAP",
        "hybrid":     "Hybrid",
        "baseline":   "Baseline",
    }
    for key, label in type_map.items():
        if key in name:
            model_type = label
            break

    # Detect SI-SDR from filename if present (e.g. "3.22dB")
    import re
    sdr_match = re.search(r"(\d+\.\d+)\s*d[bB]", pt_path.name)
    sdr_str = f" [{sdr_match.group(1)} dB]" if sdr_match else ""

    # Compose label
    parts = [p for p in [phase, model_type] if p]
    label = " — ".join(parts) if parts else "Model"
    label += sdr_str
    label += f"  ({pt_path.name}, {size:.1f} MB)"

    return label


def _render_checkpoint_selector(ckpt_default=""):
    """
    Render checkpoint selection UI.
    Returns the selected checkpoint path as a string.
    """
    st.markdown(
        '<div style="font-size:0.85rem;font-weight:700;'
        'color:#D1D5DB;margin-bottom:8px;">GSN Checkpoint</div>',
        unsafe_allow_html=True,
    )

    # Scan for available checkpoints
    checkpoints = _scan_checkpoints()

    if checkpoints:
        # Dropdown selector
        options = list(checkpoints.keys())
        paths   = list(checkpoints.values())

        # Find default index
        default_idx = 0
        if ckpt_default:
            for i, p in enumerate(paths):
                if ckpt_default in p:
                    default_idx = i
                    break

        selected_label = st.selectbox(
            "Select checkpoint",
            options=options,
            index=default_idx,
            label_visibility="collapsed",
            help="Auto-detected .pt files from your project directory.",
        )

        selected_path = checkpoints[selected_label]

        # Show path and status
        st.markdown(
            f'<div style="font-size:0.72rem;color:#10B981;'
            f'margin-top:4px;">File found</div>'
            f'<div style="font-size:0.68rem;color:#4B5563;'
            f'word-break:break-all;margin-top:2px;">'
            f'{selected_path}</div>',
            unsafe_allow_html=True,
        )

        # Option to enter custom path
        with st.expander("Enter path manually"):
            custom = st.text_input(
                "Custom checkpoint path",
                value="",
                placeholder="C:\\path\\to\\your\\model.pt",
                label_visibility="collapsed",
            )
            if custom.strip():
                import os
                if os.path.exists(custom.strip()):
                    st.markdown(
                        '<span style="font-size:0.72rem;color:#10B981;">'
                        'File found</span>',
                        unsafe_allow_html=True,
                    )
                    return custom.strip()
                else:
                    st.markdown(
                        '<span style="font-size:0.72rem;color:#F43F5E;">'
                        'File not found</span>',
                        unsafe_allow_html=True,
                    )

        return selected_path

    else:
        # No checkpoints found — manual entry only
        st.markdown(
            '<div style="font-size:0.78rem;color:#F59E0B;'
            'margin-bottom:8px;">No .pt files detected in project.</div>',
            unsafe_allow_html=True,
        )

        manual_path = st.text_input(
            "Checkpoint path",
            value=ckpt_default or "",
            placeholder="C:\\path\\to\\best_model.pt",
            label_visibility="collapsed",
        )

        if manual_path.strip():
            import os
            exists = os.path.exists(manual_path.strip())
            color  = "#10B981" if exists else "#F43F5E"
            label  = "File found" if exists else "File not found"
            st.markdown(
                f'<span style="font-size:0.72rem;color:{color};">'
                f'{label}</span>',
                unsafe_allow_html=True,
            )
            return manual_path.strip()

        st.caption(
            "Run in terminal to find checkpoints:\n"
            'Get-ChildItem -Recurse -Filter "*.pt"'
        )

        return ""
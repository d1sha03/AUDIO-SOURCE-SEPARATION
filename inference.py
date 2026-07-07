import os
import sys
import torch
import torchaudio
from pathlib import Path

# Ensures Python can find your 'src' folder
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from demucs.apply import apply_model
from demucs.pretrained import get_model
from src.models.unet import UNetConfig
from src.models.gsn_complex import GSNComplex
from src.models.clap_encoder import CLAPTextEncoder
from src.audio.complex_dsp import ComplexSTFT

class GSNInferenceEngine:
    def __init__(self, gsn_checkpoint, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing Pipeline on {self.device}...")

        # 1. Load Demucs (Base Separator)
        self.demucs = get_model("htdemucs").to(self.device).eval()

        # 2. Load CLAP (Text Encoder)
        self.clap = CLAPTextEncoder()

        # 3. Setup GSN Configuration
        config = UNetConfig(
            depth=4,
            base_channels=24,
            pool_freq=True
        )

        # 4. Initialize GSN Model Architecture
        print("[*] Initializing GSN Complex Architecture...")
        self.gsn = GSNComplex(config).to(self.device)

        # 5. Initialize DSP
        self.stft = ComplexSTFT().to(self.device)

        # 6. Load Custom Weights
        if os.path.exists(gsn_checkpoint):
            print(f"[*] Loading weights from: {gsn_checkpoint}")
            checkpoint = torch.load(gsn_checkpoint, map_location=self.device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            if "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]

            # Use strict=False to handle potential version mismatches
            self.gsn.load_state_dict(state_dict, strict=False)
            print("[✓] GSN Weights loaded successfully.")
        else:
            print(f"[!] Checkpoint not found at: {gsn_checkpoint}. Running without refinement.")
            self.gsn_active = False
            return

        self.gsn.eval()
        self.gsn_active = True

    def run(self, input_audio_path, text_prompt, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)
        import soundfile as sf; mix_np, sr = sf.read(input_audio_path, always_2d=True); mix = torch.from_numpy(mix_np.T).float()

        # Step A: Base Separation (Demucs)
        with torch.no_grad():
            # stems: [4, 2, T] (drums, bass, other, vocals)
            stems = apply_model(self.demucs, mix.unsqueeze(0).to(self.device))[0]

        if not self.gsn_active:
             # Fallback to simple Demucs routing
             is_vocal_request = any(w in text_prompt.lower() for w in ["vocal", "voice", "singing"])
             if is_vocal_request:
                 final_output = stems[3]
             else:
                 stem_idx = 0 if "drum" in text_prompt.lower() else 1 if "bass" in text_prompt.lower() else 2
                 final_output = stems[stem_idx]
        else:
            # Step B: Semantic Routing
            is_vocal_request = any(w in text_prompt.lower() for w in ["vocal", "voice", "singing"])

            # Step C: Refinement
            if is_vocal_request:
                target_audio = stems[3] # Vocals stem [2, T]
                with torch.no_grad():
                    text_emb = self.clap.encode(text_prompt, device=self.device)

                    # Transform to complex [1, 4, F, Tf]
                    tgt_complex = self.stft.transform(target_audio.unsqueeze(0))

                    # Process L and R channels
                    # tgt_complex is [1, 4, F, Tf] -> [L_re, L_im, R_re, R_im]
                    mask_l = self.gsn(tgt_complex[:, 0:2], text_embedding=text_emb) # [1, 2, F, Tf]
                    mask_r = self.gsn(tgt_complex[:, 2:4], text_embedding=text_emb) # [1, 2, F, Tf]

                    full_mask = torch.cat([mask_l, mask_r], dim=1) # [1, 4, F, Tf]

                    # Apply complex mask
                    refined_complex = self.stft.apply_complex_mask(tgt_complex, full_mask)

                    # Inverse STFT
                    final_output = self.stft.inverse(refined_complex, length=target_audio.shape[-1]).squeeze(0)
            else:
                stem_idx = 0 if "drum" in text_prompt.lower() else 1 if "bass" in text_prompt.lower() else 2
                final_output = stems[stem_idx]

        # Step D: Save Result
        out_path = os.path.join(output_dir, f"separated_{Path(input_audio_path).stem}.wav")
        import soundfile as sf; sf.write(out_path, final_output.cpu().numpy().T, sr)
        return out_path

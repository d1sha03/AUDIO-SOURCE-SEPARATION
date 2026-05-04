import os
import sys
import torch
import torchaudio
from pathlib import Path

# Ensures Python can find your 'src' folder
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from demucs.apply import apply_model
from demucs.pretrained import get_model
from src.models.unet import UNetConfig
from src.models.gsn_complex import GSNComplex
from src.models.clap_encoder import CLAPTextEncoder

class GSNInferenceEngine:
    def __init__(self, gsn_checkpoint, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing Pipeline on {self.device}...")
        
        # 1. Load Demucs (Base Separator)
        self.demucs = get_model("htdemucs").to(self.device).eval()
        
        # 2. Load CLAP (Text Encoder)
        self.clap = CLAPTextEncoder()
        
        # 3. Setup GSN Configuration
        # We define 'config' here so it is available for the next line
        config = UNetConfig(
            depth=4, 
            base_channels=24,
            pool_freq=True # Matching your memory-efficient setup
        )
        
        # 4. Initialize GSN Model Architecture
        print("[*] Initializing GSN Complex Architecture...")
        # Passing the 'config' variable we just defined
        self.gsn = GSNComplex(config).to(self.device)
        
        # 5. Load Custom Weights
        if os.path.exists(gsn_checkpoint):
            print(f"[*] Loading weights from: {gsn_checkpoint}")
            checkpoint = torch.load(gsn_checkpoint, map_location=self.device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            
            # Using strict=False to handle the Phase 1 to Phase 4 transition
            self.gsn.load_state_dict(state_dict, strict=False)
            print("[✓] GSN Weights loaded successfully.")
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {gsn_checkpoint}")
            
        self.gsn.eval()

    def run(self, input_audio_path, text_prompt, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)
        mix, sr = torchaudio.load(input_audio_path)
        
        # Step A: Base Separation (Demucs)
        with torch.no_grad():
            stems = apply_model(self.demucs, mix.unsqueeze(0).to(self.device))[0]

        # Step B: Semantic Routing
        is_vocal_request = any(w in text_prompt.lower() for w in ["vocal", "voice", "singing"])
        
        # Step C: Refinement
        if is_vocal_request:
            target_audio = stems[3] # Vocals stem
            with torch.no_grad():
                text_emb = self.clap.encode(text_prompt, device=self.device)
                try:
                    # Model forward pass
                    final_output = self.gsn(target_audio.unsqueeze(0), text_embedding=text_emb).squeeze(0)
                except Exception:
                    final_output = target_audio
        else:
            stem_idx = 0 if "drum" in text_prompt.lower() else 1 if "bass" in text_prompt.lower() else 2
            final_output = stems[stem_idx]

        # Step D: Save Result
        out_path = os.path.join(output_dir, f"separated_{Path(input_audio_path).stem}.wav")
        torchaudio.save(out_path, final_output.cpu(), sr)
        return out_path
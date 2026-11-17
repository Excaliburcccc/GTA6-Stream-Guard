"""Local zero-shot scene triage. Scores are similarities, not probabilities.

This branch only returns a whole-scene decision. Small objects, brief cuts,
offscreen audio and the intent of a player need other checks.
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path
from threading import Lock


_shared_guard = None
_shared_lock = Lock()
_model_load_lock = Lock()

MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "clip-vit-base-patch32"

# Competing ordinary contexts prevent e.g. a patrol car, swimsuit or red light
# alone from being treated as a reason to hide an entire scene. Local detectors
# still run on frames that pass this branch.
ORDINARY = {
    "street": ("people on a city sidewalk, walking and talking", "a normal street scene with buildings and pedestrians"),
    "driving": ("driving a car past traffic on the highway", "a dashboard, city traffic and a parked police car"),
    "beach": ("a wide view of a public beach with tourists in swimsuits", "a seafront with beach umbrellas, a pool and holidaymakers"),
    "fitness": ("athletes in sportswear lifting weights in a gym", "people playing a sport or exercising"),
    "interior": ("people talking inside an apartment or office", "an ordinary kitchen with someone getting a drink"),
    "nightlife": ("fully dressed guests socializing in a bar under colored lights", "a concert with stage lights and a crowd in ordinary clothing"),
    "uniform": ("police officers patrolling a city street", "a person in uniform standing near a vehicle"),
    "range": ("recreational target practice at a shooting range", "a character holding a gun away from people"),
    "outdoors": ("a panoramic city skyline or open countryside", "a lake, woodland, mountains or a boat on the water"),
    "interface": ("a game menu or map with interface icons", "a loading screen, title card or game settings menu"),
    "night_skyline": ("the exterior of a city building at night with neon signs and streetlights", "a night skyline with illuminated office towers and highways"),
    "conversation": ("a close-up of a speaking person's face, in a colored light", "people having a face-to-face conversation or talking on a phone or radio"),
    "corridor": ("a quiet hallway with someone walking to a door", "a person walking peacefully in an apartment building stairwell"),
}

RESTRICTED = {
    "scene_severe_injury": ("graphic closeup of a mutilated body and severe open wounds", "dismemberment, exposed organs or extensive gore"),
    "scene_brutality": ("sustained brutal beating of a helpless civilian", "a gruesome torture scene with a captive victim"),
    "scene_attack_officers": ("an armed attacker firing at uniformed police officers", "a violent beating of a uniformed law enforcement officer"),
    "scene_sexual_venue": ("an adult strip club with pole dancers and sexualized advertising", "a sexually explicit stage show with erotic dancers"),
    "scene_sexual_activity": ("an explicit sexual encounter between people", "a close-up of explicit sexual activity"),
    "scene_drug_activity": ("people manufacturing illegal drugs in a clandestine drug lab", "a person visibly using illegal drugs in a drug den"),
    "scene_close_combat": ("a third-person gunfight, an armed shooter aiming and firing down an apartment corridor", "a person shooting a victim at close range in an office or a hallway"),
    "scene_adult_stage": ("a striptease club stage with a stripper pole and neon erotic decor", "an adult cabaret, a pole dancer performing for a tipping audience"),
    "scene_erotic_decor": ("a sculpture of naked buttocks above a stage in a strip club", "sexualized art showing naked women decorating an adult venue"),
}


@dataclass(frozen=True)
class SceneDecision:
    block: bool
    category: str
    margin: float
    available: bool


class SceneGuard:
    def __init__(self, device="auto", model_path=None, margin=0.015, *, scorer=None):
        """A scorer, if supplied for testing, returns per-category cosine scores."""
        self.margin = float(margin)
        if not math.isfinite(self.margin) or not 0 <= self.margin < 1:
            raise ValueError("scene margin must be finite and in [0, 1)")
        self.error = ""
        self.device = device
        self.available = False
        self._scorer = scorer
        self.labels = tuple(ORDINARY) + tuple(RESTRICTED)
        if scorer is None:
            path = Path(model_path or os.environ.get("GTA6_SCENE_MODEL") or DEFAULT_MODEL_PATH)
            try:
                if not path.is_dir():
                    raise FileNotFoundError(f"Missing scene model at {path}. Run download_scene_model.py.")
                with _model_load_lock:
                    self._load(path, device)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                return
        self.available = True

    def _load(self, path, device):
        # These variables are read on first import of huggingface_hub.
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._torch = torch
        self._processor = CLIPProcessor.from_pretrained(path, local_files_only=True, use_fast=False)
        self._model = CLIPModel.from_pretrained(path, local_files_only=True).eval().to(device)
        groups = list(ORDINARY.values()) + list(RESTRICTED.values())
        phrases = ["a video game screenshot showing " + phrase for group in groups for phrase in group]
        tokens = self._processor.tokenizer(phrases, padding=True, return_tensors="pt").to(device)
        normalize = torch.nn.functional.normalize
        with torch.inference_mode():
            embeddings = normalize(self._model.get_text_features(**tokens).float(), dim=-1)
            grouped = embeddings.split([len(group) for group in groups])
            self._text = normalize(torch.stack([group.mean(dim=0) for group in grouped]), dim=-1)
        self._scorer = self._cosine_scores

    def _cosine_scores(self, frame_bgr):
        from PIL import Image, ImageOps

        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype.name != "uint8":
            raise ValueError("Expected an 8-bit BGR frame with three channels.")
        rgb = Image.fromarray(frame_bgr[:, :, ::-1])
        # CLIP's stock center crop discards both outer edges of wide gameplay.
        # Pad to square first, preserving the full field of view.
        size = self._model.config.vision_config.image_size
        square = ImageOps.pad(rgb, (size, size), method=Image.Resampling.BICUBIC, color=(123, 117, 104))
        pixels = self._processor.image_processor(square, return_tensors="pt").pixel_values.to(self.device)
        with self._torch.inference_mode():
            features = self._model.get_image_features(pixel_values=pixels).float()
            unit = self._torch.nn.functional.normalize(features, dim=-1)
            scores = (unit @ self._text.T)[0].cpu().tolist()
        return dict(zip(self.labels, scores, strict=True))

    def analyze(self, frame_bgr):
        if self.available:
            try:
                scores = self._scorer(frame_bgr)
                if not all(math.isfinite(scores[label]) for label in self.labels):
                    raise ValueError("Invalid scene score.")
                category = max(RESTRICTED, key=scores.__getitem__)
                gap = float(scores[category] - max(scores[label] for label in ORDINARY))
                if gap > self.margin:
                    return SceneDecision(True, category, gap, True)
                if gap >= -self.margin:
                    return SceneDecision(True, "scene_uncertain", gap, True)
                return SceneDecision(False, "scene_clear", gap, True)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self.available = False
        return SceneDecision(True, "scene_unavailable", 0.0, False)


def shared_scene_guard():
    """Share the read-only CLIP encoder across the local crop detectors."""
    global _shared_guard
    with _shared_lock:
        if _shared_guard is None:
            loaded = SceneGuard()
            if loaded.available:
                _shared_guard = loaded
            return loaded
        return _shared_guard
# app/muq_embed.py

import time
import shutil
import subprocess
import os
import yaml
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import torch.nn.functional as TF
import joblib


from muq import MuQMuLan
from .tag_bank import (
    FAMILY_LABELS, FAMILY_PROMPTS,
    MOOD_LABELS, MOOD_PROMPTS,
    FAMILY_FALLBACK_GENRE,
    GENRE_ALL_LABELS, GENRE_ALL_PROMPTS, GENRE_TO_FAMILY,
    STAGE2,
)

# -----------------------
# Config / helpers
# -----------------------

def load_cfg():
    cfg_path = os.getenv("MUSICAI_CONFIG", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    db_dsn = os.getenv("MUSICAI_DB_DSN")
    if db_dsn:
        cfg["db_dsn"] = db_dsn
    device = os.getenv("MUSICAI_DEVICE")
    if device:
        cfg["device"] = device
    return cfg

def load_audio_torch(path: str, target_sr: int = 24000):
    # Prefer soundfile (libsndfile) to avoid FFmpeg ABI issues
    try:
        import soundfile as sf
        y, sr = sf.read(path, always_2d=False, dtype="float32")
        if y.ndim == 2:
            y = y.mean(axis=1)
        wav = torch.from_numpy(np.ascontiguousarray(y))  # [T] CPU
    except Exception as e:
        # Fallback: torchaudio (FFmpeg/sox backends may be required for m4a/opus)
        try:
            wav, sr = torchaudio.load(path)  # [C,T]
            if wav.ndim == 2:
                wav = wav.mean(dim=0)
        except Exception as e2:
            # Final fallback: ffmpeg CLI if available
            try:
                if shutil.which("ffmpeg") is None:
                    raise RuntimeError("ffmpeg not found in PATH")
                cmd = [
                    "ffmpeg",
                    "-nostdin",
                    "-i",
                    path,
                    "-f",
                    "f32le",
                    "-ac",
                    "1",
                    "-ar",
                    str(int(target_sr)),
                    "-",
                ]
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if proc.returncode != 0 or not proc.stdout:
                    err = proc.stderr.decode("utf-8", "ignore")[-500:]
                    raise RuntimeError(f"ffmpeg decode failed: {err}")
                y = np.frombuffer(proc.stdout, dtype=np.float32)
                wav = torch.from_numpy(np.ascontiguousarray(y))
                sr = int(target_sr)
            except Exception as e3:
                raise RuntimeError(
                    f"Failed to decode audio: {path} (torchaudio: {e2}; ffmpeg: {e3})"
                ) from e

    if sr != target_sr:
        wav = F.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
        sr = target_sr
    return wav.contiguous(), sr

# -----------------------
# Engine
# -----------------------

class MuQEngine:
    def __init__(self, device: str, model_path: str | None = None, offline: bool = False):
        if device == "xpu":
            if not hasattr(torch, "xpu") or not torch.xpu.is_available():
                raise RuntimeError("device='xpu' requested but torch.xpu.is_available() is False")
        self.device = device

        if offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        model_id = model_path or "OpenMuQ/MuQ-MuLan-large"
        kwargs = {}
        if offline:
            kwargs["local_files_only"] = True
        if model_path:
            safepath = os.path.join(model_path, "model.safetensors")
            binpath = os.path.join(model_path, "pytorch_model.bin")
            if not os.path.exists(safepath) and os.path.exists(binpath):
                try:
                    from safetensors.torch import save_file
                    state = torch.load(binpath, map_location="cpu")
                    save_file(state, safepath)
                    if os.path.exists(safepath):
                        os.remove(binpath)
                except Exception:
                    pass
        self.mulan = MuQMuLan.from_pretrained(model_id, **kwargs).to(self.device).eval()

        # Cache for arbitrary text embeddings (rarely used after we precompute banks)
        self._text_cache = {}
        self._genre_clf = None

        # Precompute banks ONCE (big speedup)
        self.family_labels = list(FAMILY_LABELS)
        self.family_bank = self._embed_label_bank(self.family_labels, FAMILY_PROMPTS)

        self.mood_labels = list(MOOD_LABELS)
        self.mood_bank = self._embed_label_bank(self.mood_labels, MOOD_PROMPTS)

        # stage2 per family: {family: (labels, bank_tensor)}
        self.stage2 = {}
        for fam, (labels, prompts_map) in STAGE2.items():
            labels = list(labels)
            bank = self._embed_label_bank(labels, prompts_map)
            self.stage2[fam] = (labels, bank)

        # Global genre bank (cross-family)
        self.genre_all_labels = list(GENRE_ALL_LABELS)
        self.genre_all_bank = self._embed_label_bank(self.genre_all_labels, GENRE_ALL_PROMPTS)

    def _load_genre_clf(self, path: str):
        if self._genre_clf is not None:
            return
        payload = joblib.load(path)
        if not isinstance(payload, dict) or "model" not in payload or "mlb" not in payload:
            raise ValueError("Invalid genre classifier file; expected dict with 'model' and 'mlb'.")
        self._genre_clf = payload

    @torch.no_grad()
    def _embed_texts_list(self, texts):
        """
        Embed + L2-normalize. Cached by exact tuple(texts).
        Returned tensor is kept on the model device.
        """
        key = tuple(texts)
        if key in self._text_cache:
            return self._text_cache[key]
        te = self.mulan(texts=texts)  # [M,D]
        te = torch.nn.functional.normalize(te, dim=-1)
        self._text_cache[key] = te
        return te

    def _embed_label_bank(self, labels, prompts_map):
        """
        Build a label bank where each label can have one or multiple prompts.
        Multiple prompts are averaged into a single normalized embedding.
        """
        flat_prompts = []
        spans = []
        for label in labels:
            prompts = prompts_map[label]
            if isinstance(prompts, str):
                prompts = [prompts]
            elif not isinstance(prompts, (list, tuple)):
                raise TypeError(f"prompts_map[{label}] must be str or list/tuple of str")
            start = len(flat_prompts)
            flat_prompts.extend(prompts)
            spans.append((start, len(flat_prompts)))

        te = self._embed_texts_list(flat_prompts)  # [P,D]
        bank = []
        for start, end in spans:
            v = te[start:end].mean(dim=0, keepdim=True)
            v = torch.nn.functional.normalize(v, dim=-1)
            bank.append(v)
        return torch.cat(bank, dim=0)

    # -----------------------
    # Embedding
    # -----------------------

    @torch.inference_mode()
    def embed_track(self, path: str, sr: int = 24000, chunk_s: float = 10.0, hop_s: float = 5.0, active_keep: int = 4):
        wav_cpu, sr = load_audio_torch(path, target_sr=sr)  # CPU [T]
        duration_s = float(wav_cpu.numel() / sr)

        chunk = int(chunk_s * sr)
        hop = int(hop_s * sr)

        if wav_cpu.numel() <= chunk:
            wavs_cpu = wav_cpu.unsqueeze(0)  # [1,T]
        else:
            # Vectorized window energy: mean(x^2) over each chunk, stride=hop
            x2 = (wav_cpu.float().square()).view(1, 1, -1)  # [1,1,T]
            energies = TF.avg_pool1d(x2, kernel_size=chunk, stride=hop).squeeze(0).squeeze(0)  # [F]

            k = min(active_keep, energies.numel())
            top = torch.topk(energies, k=k).indices
            top, _ = torch.sort(top)  # keep chronological order

            starts = (top * hop).tolist()
            wavs_cpu = torch.stack([wav_cpu[s:s + chunk] for s in starts], dim=0)  # [N,chunk]

        # Optional: pin memory to speed H2D copy
        if self.device == "xpu":
            wavs_cpu = wavs_cpu.contiguous().pin_memory()
            wavs = wavs_cpu.to(self.device, non_blocking=True)
        else:
            wavs = wavs_cpu.to(self.device)

        # Mixed precision on XPU (often faster)
        if self.device == "xpu":
            with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
                embs = self.mulan(wavs=wavs)  # [N,D]
        else:
            embs = self.mulan(wavs=wavs)

        embs = torch.nn.functional.normalize(embs, dim=-1)
        pooled = torch.nn.functional.normalize(embs.mean(dim=0, keepdim=True), dim=-1)
        return pooled, duration_s, wav_cpu  # return wav_cpu so you can reuse it

    def _select_active_chunks_torch(self, wav_cpu: torch.Tensor, sr: int, chunk_s: float, hop_s: float, keep: int):
        chunk = int(chunk_s * sr)
        hop = int(hop_s * sr)

        if wav_cpu.numel() <= chunk:
            return wav_cpu.unsqueeze(0), [wav_cpu.numel() / sr]  # [1,T], durations list

        # window energy: mean(x^2) over each chunk, stride=hop
        x2 = (wav_cpu.float().square()).view(1, 1, -1)  # [1,1,T]
        energies = TF.avg_pool1d(x2, kernel_size=chunk, stride=hop).squeeze(0).squeeze(0)  # [F]

        k = min(int(keep), int(energies.numel()))
        top = torch.topk(energies, k=k).indices
        top, _ = torch.sort(top)

        starts = (top * hop).tolist()
        chunks = torch.stack([wav_cpu[s:s + chunk] for s in starts], dim=0)  # [k,chunk]
        return chunks, [wav_cpu.numel() / sr]

    @torch.inference_mode()
    def embed_tracks(
        self,
        paths,
        sr: int = 24000,
        chunk_s: float = 10.0,
        hop_s: float = 5.0,
        active_keep: int = 4,
        microbatch_chunks: int = 64,   # how many chunks per forward pass
        num_workers: int = 0,          # unused; kept for API compatibility
        return_timings: bool = False,
    ):
        """
        Returns list of (pooled_emb [1,D] on device, duration_s float, wav_cpu torch.Tensor)
        pooled is per-track mean over its chunks.
        """
        # 1) CPU: decode/resample + select chunks per track
        t_load_start = time.perf_counter()
        track_chunks = []  # list of CPU tensors [k,chunk]
        track_wavs = []    # keep wav_cpu for rms/bpm reuse
        durations = []

        for p in paths:
            wav_cpu, sr2 = load_audio_torch(p, target_sr=sr)
            # sr2 should equal sr after resample
            chunks_cpu, durs = self._select_active_chunks_torch(wav_cpu, sr, chunk_s, hop_s, active_keep)

            track_chunks.append(chunks_cpu)
            track_wavs.append(wav_cpu)
            durations.append(float(durs[0]))
        t_load_end = time.perf_counter()

        # 2) Pack all chunks and remember boundaries
        t_pack_start = time.perf_counter()
        sizes = [c.shape[0] for c in track_chunks]            # chunks per track
        offsets = [0]
        for s in sizes:
            offsets.append(offsets[-1] + int(s))

        all_chunks_cpu = torch.cat(track_chunks, dim=0).contiguous()  # [B,chunk]

        # Optional: pin for faster H2D
        if self.device == "xpu":
            all_chunks_cpu = all_chunks_cpu.pin_memory()
        t_pack_end = time.perf_counter()

        # 3) Run model in chunk-microbatches to control memory
        t_infer_start = time.perf_counter()
        all_embs = []
        B = all_chunks_cpu.shape[0]

        for i in range(0, B, int(microbatch_chunks)):
            mb_cpu = all_chunks_cpu[i:i + int(microbatch_chunks)]
            mb = mb_cpu.to(self.device, non_blocking=True)

            if self.device == "xpu":
                with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
                    embs = self.mulan(wavs=mb)
            else:
                embs = self.mulan(wavs=mb)

            embs = torch.nn.functional.normalize(embs, dim=-1)
            all_embs.append(embs)

        all_embs = torch.cat(all_embs, dim=0)  # [B,D] on device
        t_infer_end = time.perf_counter()

        # 4) Pool per track
        t_pool_start = time.perf_counter()
        results = []
        for ti in range(len(paths)):
            a, b = offsets[ti], offsets[ti + 1]
            embs_t = all_embs[a:b]  # [k,D]
            pooled = torch.nn.functional.normalize(embs_t.mean(dim=0, keepdim=True), dim=-1)  # [1,D]
            results.append((pooled, durations[ti], track_wavs[ti]))
        t_pool_end = time.perf_counter()

        if return_timings:
            timings = {
                "load_s": t_load_end - t_load_start,
                "pack_s": t_pack_end - t_pack_start,
                "infer_s": t_infer_end - t_infer_start,
                "pool_s": t_pool_end - t_pool_start,
                "chunks": int(B),
            }
            return results, timings

        return results

    # -----------------------
    # Classification
    # -----------------------

    @staticmethod
    def _decide_torch(labels, sims: torch.Tensor, threshold: float, margin: float, allow_fallback: bool = True):
        """
        sims: [M] torch tensor (on device)
        Returns: (label_or_none, best, margin, confident_bool)
        """
        k = 2 if sims.numel() >= 2 else 1
        vals, idx = torch.topk(sims, k=k)
        best = float(vals[0])
        second = float(vals[1]) if k == 2 else -1e9
        m = best - second
        i = int(idx[0])
        decided = (best >= threshold) and (m >= margin)
        if decided:
            return labels[i], best, m, True
        if allow_fallback:
            return labels[i], best, m, False
        return None, best, m, False

    @torch.no_grad()
    def classify(self, emb: torch.Tensor, cfg: dict):
        """
        emb: torch.Tensor [1,D] on device, already normalized
        """
        family_allow_fallback = bool(cfg.get("family_allow_fallback", True))
        genre_allow_fallback = bool(cfg.get("genre_allow_fallback", True))
        mood_allow_fallback = bool(cfg.get("mood_allow_fallback", True))
        genre_global = bool(cfg.get("genre_global", True))
        family_from_genre = bool(cfg.get("family_from_genre", True))
        return_topk = bool(cfg.get("return_topk", False))
        topk_n = int(cfg.get("topk_n", 5))
        genre_clf_path = cfg.get("genre_clf_path")
        genre_clf_min_score = float(cfg.get("genre_clf_min_score", 0.45))

        # Stage 1: family
        fam_sims = self.mulan.calc_similarity(emb, self.family_bank).squeeze(0)  # [M]
        family, family_best, family_margin, family_conf = self._decide_torch(
            self.family_labels,
            fam_sims,
            float(cfg.get("family_threshold", 0.10)),
            float(cfg.get("family_margin", 0.01)),
            allow_fallback=family_allow_fallback,
        )
        if (not family_conf) and (not family_allow_fallback):
            family = cfg.get("family_fallback_label", "other")

        # Mood
        mood_sims = self.mulan.calc_similarity(emb, self.mood_bank).squeeze(0)
        mood, mood_best, mood_margin, mood_conf = self._decide_torch(
            self.mood_labels,
            mood_sims,
            float(cfg.get("mood_threshold", 0.10)),
            float(cfg.get("mood_margin", 0.01)),
            allow_fallback=mood_allow_fallback,
        )
        if (not mood_conf) and (not mood_allow_fallback):
            mood = cfg.get("mood_fallback_label")

        # Optional: classifier-based genre override
        clf_genre = None
        clf_score = None
        clf_margin = None
        if genre_clf_path:
            try:
                self._load_genre_clf(genre_clf_path)
                X = emb.squeeze(0).detach().cpu().numpy().reshape(1, -1)
                model = self._genre_clf["model"]
                mlb = self._genre_clf["mlb"]
                proba = model.predict_proba(X)
                proba = np.asarray(proba)
                if proba.ndim == 2 and proba.shape[1] == len(mlb.classes_):
                    probs = proba[0]
                    top2 = np.argsort(-probs)[:2]
                    clf_genre = str(mlb.classes_[top2[0]])
                    clf_score = float(probs[top2[0]])
                    clf_margin = float(probs[top2[0]] - probs[top2[1]]) if len(top2) > 1 else float(probs[top2[0]])
            except Exception:
                clf_genre = None

        # Stage 2: genre within family
        if genre_global:
            labels = self.genre_all_labels
            sims = self.mulan.calc_similarity(emb, self.genre_all_bank).squeeze(0)
            genre, genre_best, genre_margin, genre_conf = self._decide_torch(
                labels,
                sims,
                float(cfg.get("genre_threshold", 0.12)),
                float(cfg.get("genre_margin", 0.02)),
                allow_fallback=genre_allow_fallback,
            )
            if (not genre_conf) and (not genre_allow_fallback):
                genre = FAMILY_FALLBACK_GENRE.get(family, cfg.get("genre_fallback_label", "other"))
            if family_from_genre and (genre in GENRE_TO_FAMILY):
                family = GENRE_TO_FAMILY[genre]
            genre_labels = labels
            genre_sims = sims
        else:
            if family in self.stage2:
                labels, bank = self.stage2[family]
                sims = self.mulan.calc_similarity(emb, bank).squeeze(0)
                genre, genre_best, genre_margin, genre_conf = self._decide_torch(
                    labels,
                    sims,
                    float(cfg.get("genre_threshold", 0.12)),
                    float(cfg.get("genre_margin", 0.02)),
                    allow_fallback=genre_allow_fallback,
                )
                if (not genre_conf) and (not genre_allow_fallback):
                    genre = FAMILY_FALLBACK_GENRE.get(family, cfg.get("genre_fallback_label", family))
                genre_labels = labels
                genre_sims = sims
            else:
                genre, genre_best, genre_margin, genre_conf = family, family_best, family_margin, family_conf
                genre_labels = self.family_labels
                genre_sims = fam_sims

        genre_score_val = float(genre_best)
        genre_margin_val = float(genre_margin)
        genre_conf_val = bool(genre_conf)

        if genre is None:
            genre = FAMILY_FALLBACK_GENRE.get(family, "other")

        # Override with classifier if confident
        if clf_genre and clf_score is not None and clf_score >= genre_clf_min_score:
            genre = clf_genre
            genre_score_val = float(clf_score)
            genre_margin_val = float(clf_margin) if clf_margin is not None else 0.0
            genre_conf_val = True
            if family_from_genre and (genre in GENRE_TO_FAMILY):
                family = GENRE_TO_FAMILY[genre]

        out = {
            "family": family,
            "family_score": float(family_best),
            "family_margin": float(family_margin),
            "family_confident": bool(family_conf),

            "genre": genre,
            "genre_score": genre_score_val,
            "genre_margin": genre_margin_val,
            "genre_confident": genre_conf_val,

            "mood": mood,
            "mood_score": float(mood_best),
            "mood_margin": float(mood_margin),
            "mood_confident": bool(mood_conf),
        }
        if return_topk:
            k = min(int(topk_n), int(fam_sims.numel()))
            vals, idx = torch.topk(fam_sims, k=k)
            out["family_topk"] = [(self.family_labels[int(i)], float(v)) for v, i in zip(vals, idx)]

            k = min(int(topk_n), int(mood_sims.numel()))
            vals, idx = torch.topk(mood_sims, k=k)
            out["mood_topk"] = [(self.mood_labels[int(i)], float(v)) for v, i in zip(vals, idx)]

            k = min(int(topk_n), int(genre_sims.numel()))
            vals, idx = torch.topk(genre_sims, k=k)
            out["genre_topk"] = [(genre_labels[int(i)], float(v)) for v, i in zip(vals, idx)]

        return out



def rms_only_from_wav(wav_cpu: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(wav_cpu.float().square())).item())

def bpm_from_wav_fast(wav_cpu: torch.Tensor, sr: int, max_seconds: int = 60, down_sr: int = 12000) -> float:
    import librosa

    # Use only first N seconds (or you can pass the “active” region instead)
    n = min(wav_cpu.numel(), sr * max_seconds)
    y = wav_cpu[:n].cpu().numpy()

    if down_sr and down_sr < sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=down_sr)
        sr = down_sr

    return float(librosa.beat.tempo(y=y, sr=sr)[0])

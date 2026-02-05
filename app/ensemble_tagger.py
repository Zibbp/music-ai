import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio.functional as F

from .tag_bank import (
    FAMILY_LABELS,
    FAMILY_PROMPTS,
    MOOD_LABELS,
    MOOD_PROMPTS,
    STAGE2,
)


class EnsembleTagger:
    """
    Multi-model tagger that combines:
      - MuQ text-audio similarity (existing)
      - Optional CLAP zero-shot audio classification
      - Optional PANNs AudioSet tagger mapped to your labels

    It produces probabilities for family/genre/mood and then applies
    threshold + margin decisions.
    """

    def __init__(self, engine, cfg: dict):
        self.engine = engine
        self.cfg = cfg

        self.enable_clap = bool(cfg.get("clap_enable", False))
        self.enable_panns = bool(cfg.get("panns_enable", False))

        # weights per model (sum not required)
        self.weights = {
            "muq": float(cfg.get("ensemble_weight_muq", 0.6)),
            "clap": float(cfg.get("ensemble_weight_clap", 0.3)),
            "panns": float(cfg.get("ensemble_weight_panns", 0.1)),
        }

        # lazy init for optional models
        self._clap = None
        self._panns = None

    # -----------------------
    # Public API
    # -----------------------

    @torch.no_grad()
    def classify(
        self,
        pooled_emb: torch.Tensor,
        wav_cpu: torch.Tensor,
        sr: int,
        chunk_embs: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """
        Returns tag dict consistent with existing schema.
        """
        # 1) MuQ probabilities (family/mood + stage2 genre)
        muq_probs = self._muq_probs(pooled_emb, chunk_embs)

        # 2) Optional aux models
        clap_probs = self._clap_probs(wav_cpu, sr) if self.enable_clap else {}
        panns_probs = self._panns_probs(wav_cpu, sr) if self.enable_panns else {}

        # 3) Combine probabilities
        family_probs = self._combine_probs(
            FAMILY_LABELS,
            [
                (muq_probs.get("family"), self.weights["muq"]),
                (clap_probs.get("family"), self.weights["clap"]),
                (panns_probs.get("family"), self.weights["panns"]),
            ],
        )
        mood_probs = self._combine_probs(
            MOOD_LABELS,
            [
                (muq_probs.get("mood"), self.weights["muq"]),
                (clap_probs.get("mood"), self.weights["clap"]),
                (panns_probs.get("mood"), self.weights["panns"]),
            ],
        )

        # Stage2 genre depends on family
        family_label, family_best, family_margin, family_conf = self._decide(
            FAMILY_LABELS,
            family_probs,
            float(self.cfg.get("family_threshold_ensemble", 0.20)),
            float(self.cfg.get("family_margin_ensemble", 0.05)),
        )

        if family_label in STAGE2:
            genre_labels, _ = STAGE2[family_label]
            genre_probs = self._combine_probs(
                genre_labels,
                [
                    (muq_probs.get("genre", {}).get(family_label), self.weights["muq"]),
                    (clap_probs.get("genre", {}).get(family_label), self.weights["clap"]),
                    (panns_probs.get("genre", {}).get(family_label), self.weights["panns"]),
                ],
            )
            genre_label, genre_best, genre_margin, genre_conf = self._decide(
                genre_labels,
                genre_probs,
                float(self.cfg.get("genre_threshold_ensemble", 0.20)),
                float(self.cfg.get("genre_margin_ensemble", 0.05)),
            )
        else:
            genre_label, genre_best, genre_margin, genre_conf = (
                family_label,
                family_best,
                family_margin,
                family_conf,
            )

        mood_label, mood_best, mood_margin, mood_conf = self._decide(
            MOOD_LABELS,
            mood_probs,
            float(self.cfg.get("mood_threshold_ensemble", 0.18)),
            float(self.cfg.get("mood_margin_ensemble", 0.04)),
        )

        return {
            "family": family_label,
            "family_score": float(family_best),
            "family_margin": float(family_margin),
            "family_confident": bool(family_conf),

            "genre": genre_label,
            "genre_score": float(genre_best),
            "genre_margin": float(genre_margin),
            "genre_confident": bool(genre_conf),

            "mood": mood_label,
            "mood_score": float(mood_best),
            "mood_margin": float(mood_margin),
            "mood_confident": bool(mood_conf),
        }

    # -----------------------
    # MuQ helpers
    # -----------------------

    @torch.no_grad()
    def _muq_probs(
        self,
        pooled_emb: torch.Tensor,
        chunk_embs: Optional[torch.Tensor],
    ) -> Dict[str, object]:
        """
        Returns dict with probabilities for family/mood and per-family genre.
        """
        use_chunks = chunk_embs is not None and bool(self.cfg.get("classify_by_chunks", False))
        agg = str(self.cfg.get("chunk_sim_agg", "mean"))
        agg_topk_ratio = float(self.cfg.get("chunk_sim_topk_ratio", 0.6))

        if use_chunks:
            emb_for_sim = chunk_embs  # [K,D]
            fam_sims_k = self.engine.mulan.calc_similarity(emb_for_sim, self.engine.family_bank)
            fam_sims = self.engine._aggregate_chunk_sims(fam_sims_k, agg=agg, topk_ratio=agg_topk_ratio)

            mood_sims_k = self.engine.mulan.calc_similarity(emb_for_sim, self.engine.mood_bank)
            mood_sims = self.engine._aggregate_chunk_sims(mood_sims_k, agg=agg, topk_ratio=agg_topk_ratio)
        else:
            fam_sims = self.engine.mulan.calc_similarity(pooled_emb, self.engine.family_bank).squeeze(0)
            mood_sims = self.engine.mulan.calc_similarity(pooled_emb, self.engine.mood_bank).squeeze(0)

        fam_probs = self._softmax(fam_sims, temp=float(self.cfg.get("muq_softmax_temp", 0.05)))
        mood_probs = self._softmax(mood_sims, temp=float(self.cfg.get("muq_softmax_temp", 0.05)))

        genre_probs_by_family = {}
        for fam, (labels, bank) in self.engine.stage2.items():
            if use_chunks:
                sims_k = self.engine.mulan.calc_similarity(chunk_embs, bank)
                sims = self.engine._aggregate_chunk_sims(sims_k, agg=agg, topk_ratio=agg_topk_ratio)
            else:
                sims = self.engine.mulan.calc_similarity(pooled_emb, bank).squeeze(0)
            probs = self._softmax(sims, temp=float(self.cfg.get("muq_softmax_temp", 0.05)))
            genre_probs_by_family[fam] = (list(labels), probs)

        return {
            "family": (list(FAMILY_LABELS), fam_probs),
            "mood": (list(MOOD_LABELS), mood_probs),
            "genre": genre_probs_by_family,
        }

    # -----------------------
    # CLAP helpers
    # -----------------------

    def _lazy_init_clap(self):
        if self._clap is not None:
            return
        try:
            from transformers import pipeline
        except Exception as e:  # pragma: no cover - optional dependency
            self.enable_clap = False
            self._clap = e
            return

        device = self.cfg.get("clap_device", "cpu")
        if device == "xpu":
            device = "cpu"

        model_name = self.cfg.get("clap_model", "laion/clap-htsat-fused")
        self._clap = pipeline(
            task="zero-shot-audio-classification",
            model=model_name,
            device=device,
        )
        self._clap_sr = getattr(getattr(self._clap, "feature_extractor", None), "sampling_rate", None)

    def _clap_probs(self, wav_cpu: torch.Tensor, sr: int) -> Dict[str, object]:
        self._lazy_init_clap()
        if not callable(self._clap):
            return {}

        target_sr = int(self._clap_sr) if getattr(self, "_clap_sr", None) else None
        audio_np, _ = self._prepare_aux_audio(wav_cpu, sr, target_sr=target_sr)

        def _score(prompts: List[str], labels: List[str]) -> Tuple[List[str], np.ndarray]:
            out = self._clap(
                audio_np,
                candidate_labels=prompts,
            )
            # pipeline returns list of dicts sorted by score
            score_map = {d["label"]: float(d["score"]) for d in out}
            probs = np.array([score_map.get(p, 0.0) for p in prompts], dtype=np.float32)
            probs = self._normalize_probs(probs)
            return labels, probs

        fam_prompts = [FAMILY_PROMPTS[l] for l in FAMILY_LABELS]
        mood_prompts = [MOOD_PROMPTS[l] for l in MOOD_LABELS]

        fam_labels, fam_probs = _score(fam_prompts, list(FAMILY_LABELS))
        mood_labels, mood_probs = _score(mood_prompts, list(MOOD_LABELS))

        genre_probs_by_family = {}
        for fam, (labels, prompts_map) in STAGE2.items():
            prompts = [prompts_map[l] for l in labels]
            glabels, gprobs = _score(prompts, list(labels))
            genre_probs_by_family[fam] = (glabels, gprobs)

        return {
            "family": (fam_labels, fam_probs),
            "mood": (mood_labels, mood_probs),
            "genre": genre_probs_by_family,
        }

    # -----------------------
    # PANNs helpers
    # -----------------------

    def _lazy_init_panns(self):
        if self._panns is not None:
            return
        try:
            from panns_inference import AudioTagging, labels as panns_labels
        except Exception as e:  # pragma: no cover - optional dependency
            self.enable_panns = False
            self._panns = e
            return

        device = self.cfg.get("panns_device", "cpu")
        if device == "xpu":
            device = "cpu"

        self._panns = {
            "model": AudioTagging(checkpoint_path=None, device=device),
            "labels": panns_labels,
        }

    def _panns_probs(self, wav_cpu: torch.Tensor, sr: int) -> Dict[str, object]:
        self._lazy_init_panns()
        if not isinstance(self._panns, dict):
            return {}

        audio_np, sr2 = self._prepare_aux_audio(wav_cpu, sr, target_sr=32000)
        audio_np = audio_np[None, :]  # batch

        clipwise_output, _ = self._panns["model"].inference(audio_np)
        scores = np.asarray(clipwise_output[0], dtype=np.float32)
        labels = self._panns["labels"]

        # Map AudioSet labels -> your labels
        fam_probs = self._map_panns_family(scores, labels)
        mood_probs = self._map_panns_mood(scores, labels)
        genre_probs = self._map_panns_genre(scores, labels)

        return {
            "family": (list(FAMILY_LABELS), fam_probs),
            "mood": (list(MOOD_LABELS), mood_probs),
            "genre": genre_probs,
        }

    # -----------------------
    # Common helpers
    # -----------------------

    def _prepare_aux_audio(
        self,
        wav_cpu: torch.Tensor,
        sr: int,
        target_sr: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Select active chunks, optionally resample, and limit duration.
        Returns numpy mono audio and sample rate.
        """
        # reuse engine's chunk selector for "active" sections
        chunk_s = float(self.cfg.get("aux_chunk_s", 10.0))
        hop_s = float(self.cfg.get("aux_hop_s", 5.0))
        keep = int(self.cfg.get("aux_active_keep", 6))
        keep_uniform = int(self.cfg.get("aux_active_keep_uniform", 2))

        chunks, _ = self.engine._select_active_chunks_torch(
            wav_cpu, sr, chunk_s, hop_s, keep, keep_uniform=keep_uniform
        )

        audio = chunks.flatten()

        max_seconds = float(self.cfg.get("aux_max_seconds", 30.0))
        max_samples = int(max_seconds * sr)
        if audio.numel() > max_samples:
            audio = audio[:max_samples]

        if target_sr and int(target_sr) != int(sr):
            audio = F.resample(audio.unsqueeze(0), sr, int(target_sr)).squeeze(0)
            sr = int(target_sr)

        audio_np = audio.detach().cpu().numpy().astype(np.float32, copy=False)
        return audio_np, int(sr)

    def _softmax(self, sims: torch.Tensor, temp: float) -> np.ndarray:
        t = float(temp) if float(temp) > 0 else 0.05
        x = (sims / t).detach().float().cpu()
        x = torch.softmax(x, dim=-1)
        return x.numpy()

    def _normalize_probs(self, probs: np.ndarray) -> np.ndarray:
        s = float(np.sum(probs))
        if not math.isfinite(s) or s <= 0:
            return np.ones_like(probs, dtype=np.float32) / max(1, probs.size)
        return probs / s

    def _combine_probs(
        self,
        labels: List[str],
        sources: List[Tuple[Optional[Tuple[List[str], np.ndarray]], float]],
    ) -> np.ndarray:
        """
        sources: list of ((labels, probs), weight). Some may be None.
        """
        out = np.zeros(len(labels), dtype=np.float32)
        total_w = 0.0
        for src, w in sources:
            if not src or w <= 0:
                continue
            src_labels, src_probs = src
            idx = {l: i for i, l in enumerate(src_labels)}
            for i, l in enumerate(labels):
                j = idx.get(l)
                if j is not None:
                    out[i] += float(src_probs[j]) * float(w)
            total_w += float(w)
        if total_w > 0:
            out = out / total_w
        return out

    def _decide(
        self,
        labels: List[str],
        probs: np.ndarray,
        threshold: float,
        margin: float,
    ) -> Tuple[Optional[str], float, float, bool]:
        if probs.size == 0:
            return None, -1.0, -1.0, False
        idx = int(np.argmax(probs))
        best = float(probs[idx])
        # margin vs second best
        if probs.size >= 2:
            second = float(np.partition(probs, -2)[-2])
        else:
            second = -1.0
        m = best - second
        decided = (best >= float(threshold)) and (m >= float(margin))
        return labels[idx], best, m, decided

    # -----------------------
    # PANNs label mapping
    # -----------------------

    def _map_panns_family(self, scores: np.ndarray, labels: List[str]) -> np.ndarray:
        mapping = {
            "electronic": [
                "Electronic music", "Dance music", "House music", "Techno",
                "Trance music", "Dubstep", "Electronica", "Drum and bass",
            ],
            "dnb": ["Drum and bass", "Jungle"],
            "metal": ["Heavy metal", "Metal", "Death metal", "Black metal"],
            "rock_punk": ["Rock music", "Punk rock", "Grunge", "Hard rock"],
            "hiphop_rnb": ["Hip hop music", "Rap music", "Rapping", "R&B", "Soul music"],
            "classical_jazz": ["Classical music", "Orchestra", "Piano", "Jazz", "Blues"],
            "soundtrack_world": ["Soundtrack music", "Music of Africa", "Latin music", "World music"],
            "other": [],
        }
        return self._map_labels_to_probs(scores, labels, mapping, FAMILY_LABELS)

    def _map_panns_mood(self, scores: np.ndarray, labels: List[str]) -> np.ndarray:
        mapping = {
            "upbeat": ["Happy music"],
            "energetic": ["Exciting music"],
            "dark": ["Scary music"],
            "calm": ["Calm music"],
            "sad": ["Sad music"],
            "chill": ["Chillout music", "Ambient music"],
            "aggressive": ["Angry music"],
            "heavy": ["Heavy metal"],
        }
        return self._map_labels_to_probs(scores, labels, mapping, MOOD_LABELS)

    def _map_panns_genre(self, scores: np.ndarray, labels: List[str]) -> Dict[str, Tuple[List[str], np.ndarray]]:
        # Only map high-signal genres
        genre_map = {
            "electronic": {
                "techno": ["Techno"],
                "house": ["House music"],
                "trance": ["Trance music"],
                "dubstep": ["Dubstep"],
                "hard_dance": ["Hardcore techno"],
                "breaks": ["Breakbeat"],
                "edm_other": ["Electronic music", "Dance music"],
            },
            "dnb": {
                "dnb_jungle": ["Jungle"],
                "dnb_other": ["Drum and bass"],
            },
            "metal": {
                "death": ["Death metal"],
                "black": ["Black metal"],
                "heavy_metal": ["Heavy metal"],
                "metal_other": ["Metal"],
            },
            "rock_punk": {
                "hard_rock": ["Hard rock"],
                "punk": ["Punk rock"],
                "grunge": ["Grunge"],
                "rock_other": ["Rock music"],
            },
            "hiphop_rnb": {
                "rnb": ["R&B"],
                "hiphop_other": ["Hip hop music", "Rap music", "Rapping"],
            },
            "classical_jazz": {
                "classical": ["Classical music"],
                "piano_solo": ["Piano"],
                "jazz": ["Jazz"],
                "blues": ["Blues"],
                "cj_other": ["Orchestra"],
            },
            "soundtrack_world": {
                "film_score": ["Soundtrack music"],
                "world": ["World music", "Music of Africa", "Latin music"],
                "stw_other": ["Soundtrack music"],
            },
        }

        out = {}
        for fam, (labels_list, _) in STAGE2.items():
            mapping = genre_map.get(fam, {})
            probs = self._map_labels_to_probs(scores, labels, mapping, list(labels_list))
            out[fam] = (list(labels_list), probs)
        return out

    def _map_labels_to_probs(
        self,
        scores: np.ndarray,
        labels: List[str],
        mapping: Dict[str, List[str]],
        target_labels: List[str],
    ) -> np.ndarray:
        idx = {l: i for i, l in enumerate(labels)}
        probs = np.zeros(len(target_labels), dtype=np.float32)
        for i, tl in enumerate(target_labels):
            sources = mapping.get(tl, [])
            if not sources:
                continue
            vals = []
            for s in sources:
                j = idx.get(s)
                if j is not None:
                    vals.append(float(scores[j]))
            if vals:
                probs[i] = max(vals)
        return self._normalize_probs(probs)

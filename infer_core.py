#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""常驻 KWS-CTC 推理引擎，供 WebUI 复用。"""

import copy
import json
import math
import os
import sys
import tempfile
import threading
import wave
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "wekws"
EXAMPLE = REPO / "examples" / "hi_xiaowen" / "s0"
EXP_DIR = EXAMPLE / "exp" / "ds_tcn_ctc"

# Keep in sync with wekws/model/loss.py:ctc_prefix_beam_search().
SCORE_BEAM_SIZE = 3
PROB_THRESHOLD = 0.05

sys.path.insert(0, str(REPO))

from wekws.dataset.init_dataset import init_dataset  # noqa: E402
from wekws.model.kws_model import init_model  # noqa: E402
from wekws.model.loss import ctc_prefix_beam_search  # noqa: E402
from wekws.utils.checkpoint import load_checkpoint  # noqa: E402
from wenet.text.char_tokenizer import CharTokenizer  # noqa: E402


def is_sublist(main_list, check_list):
    if len(main_list) < len(check_list):
        return -1
    if len(main_list) == len(check_list):
        return 0 if main_list == check_list else -1
    for i in range(len(main_list) - len(check_list)):
        if main_list[i] == check_list[0]:
            for j in range(len(check_list)):
                if main_list[i + j] != check_list[j]:
                    break
            else:
                return i
    return -1


def wav_duration(wav_path):
    try:
        import soundfile as sf

        return float(sf.info(str(wav_path)).duration)
    except Exception:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())


def list_epoch_checkpoints():
    ckpts = []
    for p in EXP_DIR.glob("*.pt"):
        if p.stem.isdigit():
            ckpts.append((int(p.stem), p.resolve()))
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def list_named_checkpoints():
    """Return non-epoch checkpoints such as avg_30.pt, excluding init.pt."""
    ckpts = []
    for p in EXP_DIR.glob("*.pt"):
        if p.stem.isdigit() or p.stem == "init":
            continue
        ckpts.append((p.name, p.resolve()))
    ckpts.sort(key=lambda x: x[0])
    return ckpts


class KwsEngine:
    def __init__(self,
                 config=None,
                 checkpoint="auto",
                 dict_dir=None,
                 gpu=0,
                 keywords="嗨小问,你好问问",
                 batch_size=1,
                 jit_model=False):
        self.config_path = Path(config or EXP_DIR / "config.yaml").resolve()
        self.dict_dir = Path(dict_dir or EXAMPLE / "dict").resolve()
        self.gpu = int(gpu)
        self.batch_size = max(1, int(batch_size))
        self.jit_model = bool(jit_model)
        self.lock = threading.RLock()

        if not self.config_path.exists():
            raise FileNotFoundError(f"config not found: {self.config_path}")
        if not self.dict_dir.exists():
            raise FileNotFoundError(f"dict not found: {self.dict_dir}")

        if self.jit_model:
            # torch.jit 导出的模型按仓库现有逻辑只支持 CPU
            self.device = torch.device("cpu")
        elif self.gpu >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(self.gpu)
            self.device = torch.device(f"cuda:{self.gpu}")
        else:
            self.device = torch.device("cpu")

        self._load_config()
        self._build_tokenizer()
        self._build_test_conf()

        self.tmp_dir = Path(tempfile.mkdtemp(prefix="kws_webui_", dir="/tmp/opencode"))
        self.model = None
        self.current_checkpoint = None

        self.set_keywords(keywords)
        self.set_checkpoint(checkpoint)

    def _load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as fin:
            self.configs = yaml.load(fin, Loader=yaml.FullLoader)
        self.frame_shift_ms = int(
            self.configs["dataset_conf"].get("fbank_conf", {}).get("frame_shift", 10)
        )

        cmvn = self.configs["model"].get("cmvn")
        if cmvn and cmvn.get("cmvn_file"):
            raw = Path(cmvn["cmvn_file"]).expanduser()
            if not raw.is_absolute():
                candidates = [
                    EXAMPLE / raw,
                    self.config_path.parent / raw,
                    ROOT / raw,
                ]
                for cand in candidates:
                    if cand.exists():
                        cmvn["cmvn_file"] = str(cand.resolve())
                        break
                else:
                    raise FileNotFoundError(f"cmvn file not found: {raw}")

    def _build_tokenizer(self):
        self.tokenizer = CharTokenizer(
            str(self.dict_dir / "dict.txt"),
            str(self.dict_dir / "words.txt"),
            unk="<filler>",
            split_with_space=True,
        )
        self.id_to_token = {}
        with open(self.dict_dir / "dict.txt", "r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.rstrip("\n").rsplit(" ", 1)
                if len(parts) != 2:
                    continue
                token, raw_id = parts
                try:
                    token_id = int(raw_id)
                except ValueError:
                    continue
                if token_id < 0:
                    continue
                # dict.txt contains both "sil 0" and "<blk> 0"; prefer a
                # human-readable blank label for debug output.
                if token_id not in self.id_to_token or token == "<blk>":
                    self.id_to_token[token_id] = token

    def _token_label(self, token_id):
        return self.id_to_token.get(int(token_id), f"<id:{int(token_id)}>")

    def _build_test_conf(self):
        self.test_conf = copy.deepcopy(self.configs["dataset_conf"])
        filter_conf = self.test_conf["filter_conf"]
        filter_conf["max_length"] = 102400
        filter_conf["min_length"] = 0
        filter_conf["token_max_length"] = 10240
        filter_conf["token_min_length"] = 1
        filter_conf["min_output_input_ratio"] = 1e-6
        filter_conf["max_output_input_ratio"] = 1

        self.test_conf["speed_perturb"] = False
        self.test_conf["spec_aug"] = False
        self.test_conf["shuffle"] = False
        feats_type = self.test_conf.get("feats_type", "fbank")
        self.test_conf[f"{feats_type}_conf"]["dither"] = 0.0
        self.test_conf["batch_conf"]["batch_size"] = self.batch_size

    def set_keywords(self, keywords):
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        if not keywords_list:
            raise ValueError("keywords is empty")

        with self.lock:
            self.keywords_text = ",".join(keywords_list)
            self.keywords_token = {}
            self.keywords_idxset = {0}

            for keyword in keywords_list:
                strs, indexes = self.tokenizer.tokenize(" ".join(list(keyword)))
                indexes = tuple(indexes)
                self.keywords_token[keyword] = {
                    "token_id": indexes,
                    "token_str": "".join(f"{i} " for i in indexes),
                    "chars": list(strs),
                }
                self.keywords_idxset.update(indexes)

    def resolve_checkpoint(self, checkpoint):
        if checkpoint in (None, "", "auto"):
            ckpts = list_epoch_checkpoints()
            if not ckpts:
                raise RuntimeError(
                    f"No epoch checkpoint found in {EXP_DIR}. "
                    "Train a few epochs or select a checkpoint explicitly."
                )
            return ckpts[-1][1]

        raw = Path(checkpoint).expanduser()
        candidates = []
        if raw.is_absolute():
            candidates.append(raw.resolve())
        else:
            # WebUI 下拉框传的是 "5.pt" 这样的文件名，优先在 checkpoint 目录查找。
            candidates.extend([
                (EXP_DIR / raw).resolve(),
                (Path.cwd() / raw).resolve(),
                (self.config_path.parent / raw).resolve(),
            ])

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}. Tried: "
            + ", ".join(str(p) for p in candidates)
        )

    def set_checkpoint(self, checkpoint="auto", force=False):
        ckpt = self.resolve_checkpoint(checkpoint)
        with self.lock:
            if not force and self.current_checkpoint == ckpt and self.model is not None:
                return self.current_checkpoint

            if self.jit_model:
                self.model = torch.jit.load(str(ckpt), map_location="cpu")
            else:
                if self.model is None:
                    self.model = init_model(self.configs["model"]).to(self.device)
                load_checkpoint(self.model, str(ckpt))

            self.model.eval()
            self.current_checkpoint = ckpt
            return ckpt

    def _make_temp_list(self, wav_path):
        wav_path = Path(wav_path).expanduser().resolve()
        if not wav_path.exists():
            raise FileNotFoundError(f"wav not found: {wav_path}")

        data_list = self.tmp_dir / "current.data.list"
        sample = {
            "key": wav_path.stem,
            # score_ctc 不依赖 target 解码，只要求 token 合法。
            "txt": "嗨",
            "duration": round(wav_duration(wav_path), 6),
            "wav": str(wav_path),
        }
        with open(data_list, "w", encoding="utf-8") as fout:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
        return data_list

    def _normalize_data_list(self, data_list_path):
        data_list_path = Path(data_list_path).expanduser().resolve()
        if not data_list_path.exists():
            raise FileNotFoundError(f"data_list not found: {data_list_path}")

        normalized = self.tmp_dir / f"{data_list_path.stem}.normalized.list"
        with open(data_list_path, "r", encoding="utf-8") as fin, \
                open(normalized, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                wav = Path(sample["wav"]).expanduser()
                if not wav.is_absolute():
                    wav = (data_list_path.parent / wav).resolve()
                sample["wav"] = str(wav)
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
        return normalized

    def _build_dataset(self, data_list):
        return init_dataset(
            data_list_file=str(data_list),
            conf=self.test_conf,
            tokenizer=self.tokenizer,
            split="test",
        )

    def _frame_topk_details(self, score):
        """Per-frame top-k debug details from the same posterior used by CTC search."""
        details = []
        top_k = min(SCORE_BEAM_SIZE, score.size(1))
        for frame in range(score.size(0)):
            probs, ids = score[frame].topk(top_k)
            raw_topk = []
            kept = []
            for prob, token_id in zip(probs.tolist(), ids.tolist()):
                item = {
                    "token_id": int(token_id),
                    "token": self._token_label(token_id),
                    "prob": round(float(prob), 6),
                }
                raw_topk.append(item)
                if (float(prob) > PROB_THRESHOLD
                        and int(token_id) in self.keywords_idxset):
                    kept.append(dict(item))
            details.append({
                "frame": frame,
                "time_ms": frame * self.frame_shift_ms,
                "raw_topk": raw_topk,
                "kept": kept,
            })
        return details

    def _results_from_batch(self, batch, include_frame_topk=False):
        keys = batch["keys"]
        feats = batch["feats"].to(self.device)
        lengths = batch["feats_lengths"].to(self.device)

        with torch.no_grad():
            logits, _ = self.model(feats)

        logits = logits.softmax(2).cpu()
        results = []

        for i, key in enumerate(keys):
            score = logits[i][: int(lengths[i].item())]
            hyps = ctc_prefix_beam_search(score, lengths[i].cpu(), self.keywords_idxset)

            hit_keyword = None
            hit_score = 1.0
            token_probs = []
            start = 0
            end = 0

            for one_hyp in hyps:
                prefix_ids = one_hyp[0]
                prefix_nodes = one_hyp[2]
                for word, info in self.keywords_token.items():
                    lab = info["token_id"]
                    offset = is_sublist(prefix_ids, lab)
                    if offset != -1:
                        hit_keyword = word
                        start = prefix_nodes[offset]["frame"]
                        end = prefix_nodes[offset + len(lab) - 1]["frame"]
                        chars = info.get("chars") or list(word)
                        for idx in range(offset, offset + len(lab)):
                            node = prefix_nodes[idx]
                            token_id = int(node["token"])
                            pos = idx - offset
                            token = chars[pos] if pos < len(chars) else self._token_label(token_id)
                            prob = float(node["prob"])
                            token_probs.append({
                                "token": token,
                                "token_id": token_id,
                                "prob": round(prob, 6),
                                "frame": int(node["frame"]),
                                "time_ms": int(node["frame"]) * self.frame_shift_ms,
                            })
                            hit_score *= prob
                        break
                if hit_keyword is not None:
                    hit_score = math.sqrt(hit_score)
                    break

            result = {
                "key": key,
                "detected": hit_keyword is not None,
                "keyword": hit_keyword,
                "score": round(float(hit_score), 6) if hit_keyword else None,
                "start_frame": start,
                "end_frame": end,
                "duration": end - start,
                "checkpoint": str(self.current_checkpoint),
                "keywords": self.keywords_text,
                "token_probs": token_probs,
            }
            if include_frame_topk:
                result["frame_topk"] = self._frame_topk_details(score)
            results.append(result)

        return results

    def detect(self, wav_path, include_frame_topk=False):
        with self.lock:
            if self.model is None:
                self.set_checkpoint("auto")

            data_list = self._make_temp_list(wav_path)
            dataset = self._build_dataset(data_list)
            batch = next(iter(dataset))
            return self._results_from_batch(
                batch, include_frame_topk=include_frame_topk)[0]

    def detect_list(self, data_list_path, include_frame_topk=False):
        with self.lock:
            if self.model is None:
                self.set_checkpoint("auto")

            data_list = self._normalize_data_list(data_list_path)
            dataset = self._build_dataset(data_list)
            results = []
            for batch in dataset:
                results.extend(
                    self._results_from_batch(batch, include_frame_topk=include_frame_topk))
            return results

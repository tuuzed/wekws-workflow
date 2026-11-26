#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dump ONNX posteriors and Python reference scores for Android decoder tests."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
import yaml

ANDROID_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ANDROID_ROOT.parent / "wekws"
WORK_ROOT = ANDROID_ROOT.parent / "work"
sys.path.insert(0, str(ANDROID_ROOT.parent))
sys.path.insert(0, str(ANDROID_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from wekws.model.kws_model import init_model  # noqa: E402
from wekws.model.loss import ctc_prefix_beam_search  # noqa: E402
from wekws.utils.checkpoint import load_checkpoint  # noqa: E402


def is_sublist(main_list, check_list):
    if len(main_list) < len(check_list):
        return -1
    if len(main_list) == len(check_list):
        return 0 if main_list == check_list else -1
    for i in range(len(main_list) - len(check_list)):
        for j in range(len(check_list)):
            if main_list[i + j] != check_list[j]:
                break
        else:
            return i
    return -1


def python_score(probs, token_set, keywords):
    hyps = ctc_prefix_beam_search(torch.from_numpy(probs), torch.tensor([probs.shape[0]]),
                                  set(token_set))
    for hyp in hyps:
        prefix_ids = list(hyp[0])
        nodes = hyp[2]
        for keyword in keywords:
            lab = keyword["tokens"]
            offset = is_sublist(prefix_ids, lab)
            if offset == -1:
                continue
            score = 1.0
            for idx in range(offset, offset + len(lab)):
                score *= nodes[idx]["prob"]
            score = math.sqrt(score)
            return {
                "keyword": keyword["text"],
                "score": round(float(score), 6),
                "start_frame": nodes[offset]["frame"],
                "end_frame": nodes[offset + len(lab) - 1]["frame"],
            }
    return None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--onnx", default=str(ANDROID_ROOT / "app/src/main/assets/kws.onnx"))
    parser.add_argument("--config_json",
                        default=str(ANDROID_ROOT / "app/src/main/assets/config.json"))
    parser.add_argument("--train_config",
                        default=str(WORK_ROOT / "exp/hi_xiaowen/ds_tcn_ctc/config.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out_dir", default=str(ANDROID_ROOT / "tests/out"))
    return parser.parse_args()


def resolve_cmvn(config_path, cmvn):
    raw = Path(cmvn["cmvn_file"]).expanduser()
    if raw.is_absolute():
        cmvn["cmvn_file"] = str(raw.resolve())
        return
    for candidate in [
        config_path.parent.parent.parent / raw,
        config_path.parent.parent / raw,
        config_path.parent / raw,
        WORK_ROOT / "data/hi_xiaowen" / Path(raw).name,
        WORK_ROOT / "data/hi_xiaowen/train/global_cmvn",
        REPO_ROOT / raw,
    ]:
        if candidate.exists():
            cmvn["cmvn_file"] = str(candidate.resolve())
            return
    raise FileNotFoundError(raw)


def main():
    args = get_args()
    wav_path = Path(args.wav).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config_json, "r", encoding="utf-8") as fin:
        runtime_config = json.load(fin)
    with open(args.train_config, "r", encoding="utf-8") as fin:
        train_config = yaml.load(fin, Loader=yaml.FullLoader)

    # torchaudio.load() returns normalized float, then processor.py multiplies
    # by (1 << 15). soundfile with dtype=float32 gives the same normalization.
    pcm, sample_rate = sf.read(str(wav_path), dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    if sample_rate != runtime_config["sample_rate"]:
        raise ValueError(f"expected {runtime_config['sample_rate']} Hz, got {sample_rate}")
    waveform = torch.from_numpy(pcm).unsqueeze(0)  # (1, samples)
    waveform = waveform * (1 << 15)
    import torchaudio.compliance.kaldi as kaldi
    feats = kaldi.fbank(
        waveform,
        num_mel_bins=runtime_config["feature_dim"],
        frame_length=runtime_config["frame_length_ms"],
        frame_shift=runtime_config["frame_shift_ms"],
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=sample_rate,
    )
    feats = feats.unsqueeze(0)  # (1, T, D)
    np.save(out_dir / "feats.npy", feats[0].numpy())

    vocab_size = runtime_config["vocab_size"]
    hidden_dim = runtime_config["hidden_dim"]
    cache_len = runtime_config["cache_len"]
    cache = np.zeros((1, hidden_dim, cache_len), dtype=np.float32)

    ort_session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    onnx_outputs = ort_session.run(None, {
        "input": feats.numpy().astype(np.float32),
        "cache": cache,
    })
    probs = onnx_outputs[0][0]  # (T, V)

    np.save(out_dir / "probs.npy", probs)
    probs.astype(np.float32).tofile(out_dir / "probs.bin")

    ref = python_score(probs, runtime_config["keywords_token_set"],
                       runtime_config["keywords"])
    print("onnx probs shape:", probs.shape)
    print("python score:", json.dumps(ref, ensure_ascii=False))

    # Compare ONNX output to the PyTorch checkpoint when available.
    checkpoint = args.checkpoint
    if checkpoint:
        import copy
        train_config = copy.deepcopy(train_config)
        cmvn = train_config["model"].get("cmvn")
        if cmvn and cmvn.get("cmvn_file"):
            resolve_cmvn(Path(args.train_config), cmvn)
        model = init_model(train_config["model"])
        model.forward = model.forward_softmax
        load_checkpoint(model, checkpoint)
        model.eval()
        with torch.no_grad():
            torch_out, _ = model(feats, torch.from_numpy(cache))
        max_diff = float(torch.max(torch.abs(torch_out[0] - torch.from_numpy(probs))))
        print("pytorch/onnx max diff:", max_diff)

        from infer_core import KwsEngine
        engine = KwsEngine(checkpoint=checkpoint, gpu=-1,
                           keywords=",".join(k["text"] for k in runtime_config["keywords"]))
        py_result = engine.detect(str(wav_path))
        print("infer_core result:", json.dumps(py_result, ensure_ascii=False))


if __name__ == "__main__":
    main()

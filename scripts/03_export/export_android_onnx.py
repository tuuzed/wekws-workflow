#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the WeKWS CTC checkpoint to a dynamic-batch ONNX model for Android.

The exported ONNX model is loaded directly by ONNX Runtime Mobile as kws.onnx.
No .ort conversion is required.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPTS_ROOT.parent
WEKWS_ROOT = WORKSPACE_ROOT / "wekws"
WORK_ROOT = WORKSPACE_ROOT / "work"
ANDROID_ROOT = WORKSPACE_ROOT / "android"
sys.path.insert(0, str(WEKWS_ROOT))

from wenet.text.char_tokenizer import CharTokenizer  # noqa: E402
from wekws.model.kws_model import init_model  # noqa: E402
from wekws.utils.checkpoint import load_checkpoint  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(description="Export WeKWS CTC to ONNX")
    parser.add_argument(
        "--checkpoint",
        default=str(WORK_ROOT / "exp/hi_xiaowen/ds_tcn_ctc/avg_30.pt"),
        help="checkpoint to export; deployment is fixed to avg_30.pt",
    )
    parser.add_argument(
        "--config",
        default=str(WORK_ROOT / "exp/hi_xiaowen/ds_tcn_ctc/config.yaml"),
        help="training config yaml",
    )
    parser.add_argument(
        "--dict",
        default=str(WORK_ROOT / "dict/hi_xiaowen"),
        help="dictionary directory",
    )
    parser.add_argument(
        "--onnx_model",
        default=str(ANDROID_ROOT / "app/src/main/assets/kws.onnx"),
        help="output ONNX path",
    )
    parser.add_argument(
        "--config_json",
        default=str(ANDROID_ROOT / "app/src/main/assets/config.json"),
        help="output runtime config json",
    )
    parser.add_argument(
        "--keywords_txt",
        default=str(ANDROID_ROOT / "app/src/main/assets/keywords.txt"),
        help="output human-readable keyword token file",
    )
    parser.add_argument(
        "--id_to_token",
        default=str(ANDROID_ROOT / "app/src/main/assets/id_to_token.json"),
        help="output token-id to token-text map for debug display",
    )
    parser.add_argument(
        "--keywords",
        default="嗨小问,你好问问",
        help="comma-separated keywords",
    )
    parser.add_argument(
        "--keyword_thresholds",
        required=True,
        help="comma-separated keyword=threshold pairs, e.g. '嗨小问=0.303,你好问问=0.039'",
    )
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--batch", type=int, default=2, help="dummy batch for validation")
    parser.add_argument("--frames", type=int, default=100, help="dummy frames for validation")
    return parser.parse_args()


def resolve_cmvn(config_path, cmvn):
    raw = Path(cmvn["cmvn_file"]).expanduser()
    if raw.is_absolute():
        cmvn["cmvn_file"] = str(raw.resolve())
        return

    candidates = [
        config_path.parent.parent.parent / raw,
        config_path.parent.parent / raw,
        config_path.parent / raw,
        WEKWS_ROOT / raw,
        WORK_ROOT / raw,
        Path.cwd() / raw,
    ]
    # Training artifacts moved from the recipe root into work/, so the old
    # relative path data/train/global_cmvn now lives one level deeper.
    if raw.parts and raw.parts[0] == "data":
        candidates.append(
            WORK_ROOT / "data" / "hi_xiaowen" / Path(*raw.parts[1:])
        )
    for candidate in candidates:
        if candidate.exists():
            cmvn["cmvn_file"] = str(candidate.resolve())
            return
    raise FileNotFoundError(f"cmvn file not found: {raw}")


def add_metadata(model, metadata):
    for key, value in metadata.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = str(value)


def parse_keyword_thresholds(raw):
    if not raw:
        raise ValueError(
            "--keyword_thresholds is required, e.g. '嗨小问=0.303,你好问问=0.039'"
        )
    result = {}
    for item in raw.split(","):
        if "=" not in item:
            raise ValueError(f"invalid keyword threshold item: {item}")
        keyword, value = item.split("=", 1)
        keyword = keyword.strip()
        if not keyword:
            raise ValueError(f"empty keyword in threshold item: {item}")
        result[keyword] = float(value)
    return result


def main():
    args = get_args()
    keyword_thresholds = parse_keyword_thresholds(args.keyword_thresholds)

    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    onnx_path = Path(args.onnx_model).resolve()
    config_json_path = Path(args.config_json).resolve()
    keywords_txt_path = Path(args.keywords_txt).resolve()
    id_to_token_path = Path(args.id_to_token).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    with open(config_path, "r", encoding="utf-8") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    configs = copy.deepcopy(configs)
    cmvn = configs["model"].get("cmvn")
    if cmvn and cmvn.get("cmvn_file"):
        resolve_cmvn(config_path, cmvn)

    feature_dim = int(configs["model"]["input_dim"])
    vocab_size = int(configs["model"]["output_dim"])
    hidden_dim = int(configs["model"]["hidden_dim"])
    padding = int(configs["model"]["backbone"]["padding"]) if "padding" in configs["model"]["backbone"] else 0

    if padding == 0:
        kernel_size = int(configs["model"]["backbone"].get("kernel_size", 8))
        num_layers = int(configs["model"]["backbone"].get("num_layers", 4))
        padding = sum((kernel_size - 1) * (2 ** i) for i in range(num_layers))
        configs["model"]["backbone"]["padding"] = padding

    criterion = configs.get("training_config", {}).get("criterion", "max_pooling")
    model = init_model(configs["model"])
    if criterion == "ctc":
        # CTC inference consumes posterior probabilities directly.
        model.forward = model.forward_softmax
    load_checkpoint(model, str(checkpoint))
    model.eval()

    cache_len = model.backbone.padding
    assert cache_len == padding, (cache_len, padding)

    dummy_input = torch.randn(1, args.frames, feature_dim, dtype=torch.float)
    cache = torch.zeros(1, hidden_dim, cache_len, dtype=torch.float)

    dynamic_axes = {
        "input": {0: "N", 1: "T"},
        "cache": {0: "N"},
        "output": {0: "N", 1: "T"},
        "r_cache": {0: "N"},
    }

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch_output = model(dummy_input, cache)
        torch.onnx.export(
            model,
            (dummy_input, cache),
            str(onnx_path),
            input_names=["input", "cache"],
            output_names=["output", "r_cache"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            verbose=False,
            do_constant_folding=True,
        )

    metadata = {
        "model_type": "ctc" if criterion == "ctc" else "max_pooling",
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "cache_len": cache_len,
        "vocab_size": vocab_size,
        "sample_rate": int(configs["dataset_conf"]["resample_conf"]["resample_rate"]),
    }
    onnx_model = onnx.load(str(onnx_path))
    add_metadata(onnx_model, metadata)
    onnx.save(onnx_model, str(onnx_path))
    onnx.checker.check_model(str(onnx_path))

    # Keyword token ids must use the same tokenizer logic as score_ctc.py.
    tokenizer = CharTokenizer(
        str(Path(args.dict) / "dict.txt"),
        str(Path(args.dict) / "words.txt"),
        unk="<filler>",
        split_with_space=True,
    )
    id_to_token = {}
    with open(Path(args.dict) / "dict.txt", "r", encoding="utf-8") as fin:
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
            if token_id not in id_to_token or token == "<blk>":
                id_to_token[token_id] = token

    keywords = []
    keyword_token_set = {0}
    configured_keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    for keyword in configured_keywords:
        if keyword not in keyword_thresholds:
            raise ValueError(f"missing trigger_threshold for keyword: {keyword}")
        chars, indexes = tokenizer.tokenize(" ".join(list(keyword)))
        indexes = [int(i) for i in indexes]
        keywords.append({
            "text": keyword,
            "chars": list(chars),
            "tokens": indexes,
            "trigger_threshold": keyword_thresholds[keyword],
        })
        keyword_token_set.update(indexes)

    unknown_keywords = sorted(set(keyword_thresholds) - set(configured_keywords))
    if unknown_keywords:
        raise ValueError(
            "trigger_threshold given for unknown keyword(s): "
            + ", ".join(unknown_keywords)
        )

    runtime_config = {
        "model_path": onnx_path.name,
        "model_type": metadata["model_type"],
        "sample_rate": metadata["sample_rate"],
        "feature_dim": feature_dim,
        "frame_length_ms": int(configs["dataset_conf"]["fbank_conf"]["frame_length"]),
        "frame_shift_ms": int(configs["dataset_conf"]["fbank_conf"]["frame_shift"]),
        "hidden_dim": hidden_dim,
        "cache_len": cache_len,
        "vocab_size": vocab_size,
        "blank_id": 0,
        "keywords": keywords,
        "keywords_token_set": sorted(keyword_token_set),
        "ctc": {
            "score_beam_size": 3,
            "path_beam_size": 20,
            "prob_threshold": 0.05,
            "isclose_abs_tol": 1e-6,
            "cooldown_ms": 1000,
            "reset_decoder_after_trigger": True,
            "reset_cache_after_trigger": False,
        },
        "runtime": {
            "default_channels": 2,
            "chunk_frames": 80,
            "intra_op_threads": 1,
            "inter_op_threads": 1,
        },
        "debug": {
            "dump_frame_topk": False,
            "frame_topk_size": 3,
            "max_debug_frames": 2000,
        },
    }

    config_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_json_path, "w", encoding="utf-8") as fout:
        json.dump(runtime_config, fout, ensure_ascii=False, indent=2)
        fout.write("\n")

    keywords_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(keywords_txt_path, "w", encoding="utf-8") as fout:
        for item in keywords:
            token_str = " ".join(str(t) for t in item["tokens"])
            fout.write(f"{item['text']} {token_str}\n")

    id_to_token_path.parent.mkdir(parents=True, exist_ok=True)
    with open(id_to_token_path, "w", encoding="utf-8") as fout:
        json.dump(id_to_token, fout, ensure_ascii=False, indent=2)
        fout.write("\n")

    # Validate ONNX with dynamic batch and cache batch.
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    batch = max(1, args.batch)
    input_np = torch.randn(batch, args.frames, feature_dim).numpy()
    cache_np = np.zeros((batch, hidden_dim, cache_len), dtype=np.float32)
    ort_outputs = ort_session.run(
        None, {"input": input_np, "cache": cache_np}
    )
    out_shape = ort_outputs[0].shape
    cache_shape = ort_outputs[1].shape
    assert out_shape == (batch, args.frames, vocab_size), out_shape
    assert cache_shape == (batch, hidden_dim, cache_len), cache_shape

    with torch.no_grad():
        torch_ref = model(torch.from_numpy(input_np), torch.from_numpy(cache_np))
    max_diff = float(torch.max(torch.abs(torch_ref[0] - torch.from_numpy(ort_outputs[0]))))
    assert max_diff < 3e-5, max_diff

    print("ONNX export succeeded")
    print(f"  onnx:      {onnx_path}")
    print(f"  config:    {config_json_path}")
    print(f"  keywords:  {keywords_txt_path}")
    print(f"  id2token:  {id_to_token_path}")
    print(f"  metadata:  {metadata}")
    print(f"  max_diff:  {max_diff:.8f}")


if __name__ == "__main__":
    main()

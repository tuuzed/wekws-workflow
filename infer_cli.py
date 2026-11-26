#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KWS-CTC 命令行推理工具。

推理逻辑全部复用 infer_core.KwsEngine。
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from infer_core import KwsEngine  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(description="KWS CTC CLI inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wav", type=str, help="single wav file for inference")
    group.add_argument("--data_list", type=str, help="data.list for batch inference")

    parser.add_argument("--config", type=str, default=None, help="model config yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="auto",
        help="checkpoint path, or 'auto' to use latest epoch",
    )
    parser.add_argument("--dict", type=str, default=None, help="dict directory")
    parser.add_argument(
        "--keywords",
        type=str,
        default="嗨小问,你好问问",
        help="keywords separated by comma",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--batch_size", type=int, default=64, help="inference batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="kept for compatibility; core does not use DataLoader workers")
    parser.add_argument(
        "--score_file",
        type=str,
        default=str(ROOT / "infer_cli" / "score.txt"),
        help="output score file",
    )
    parser.add_argument(
        "--detail_file",
        type=str,
        default=None,
        help="write JSONL details with per-token probs",
    )
    parser.add_argument(
        "--dump_frame_topk",
        action="store_true",
        help="include per-frame top-k probs in --detail_file",
    )
    parser.add_argument("--jit_model", action="store_true", help="load a torch.jit model")
    return parser.parse_args()


def write_score_file(results, score_file):
    score_file = Path(score_file).expanduser().resolve()
    score_file.parent.mkdir(parents=True, exist_ok=True)

    with open(score_file, "w", encoding="utf-8") as fout:
        for result in results:
            key = result["key"]
            if result["detected"]:
                score = result["score"] if result["score"] is not None else 0.0
                fout.write(f"{key} detected {result['keyword']} {score:.3f}\n")
            else:
                fout.write(f"{key} rejected\n")
    return score_file


def write_detail_file(results, detail_file):
    if not detail_file:
        return None
    detail_file = Path(detail_file).expanduser().resolve()
    detail_file.parent.mkdir(parents=True, exist_ok=True)
    with open(detail_file, "w", encoding="utf-8") as fout:
        for result in results:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    return detail_file


def print_summary(results, score_file, engine, detail_file=None):
    detected = [r for r in results if r["detected"]]
    rejected = [r for r in results if not r["detected"]]

    print("\n===== Inference summary =====")
    print(f"checkpoint: {engine.current_checkpoint}")
    print(f"keywords:   {engine.keywords_text}")
    print(f"score_file: {score_file}")
    if detail_file:
        print(f"detail_file: {detail_file}")
    print(f"total:      {len(results)}")
    print(f"detected:   {len(detected)}")
    print(f"rejected:   {len(rejected)}")

    if detected:
        counter = Counter(r["keyword"] for r in detected)
        print("keyword counts:")
        for keyword, count in counter.items():
            print(f"  {keyword}: {count}")
        print("details (first 20):")
        for result in detected[:20]:
            print(
                f"  {result['key']}: detected {result['keyword']} "
                f"score={result['score']}"
            )
            token_probs = result.get("token_probs") or []
            if token_probs:
                detail = " ".join(
                    f"{p['token']}={p['prob']:.6f}" for p in token_probs
                )
                print(f"      tokens: {detail}")
    print("=============================\n")


def main():
    args = get_args()

    if args.dump_frame_topk and not args.detail_file:
        raise SystemExit("--dump_frame_topk requires --detail_file")

    if args.num_workers != 4:
        print("note: --num_workers is kept for compatibility and is currently unused")

    engine = KwsEngine(
        config=args.config,
        checkpoint=args.checkpoint,
        dict_dir=args.dict,
        gpu=args.gpu,
        keywords=args.keywords,
        batch_size=args.batch_size,
        jit_model=args.jit_model,
    )

    include_frame_topk = bool(args.dump_frame_topk)
    if args.wav:
        results = [engine.detect(args.wav, include_frame_topk=include_frame_topk)]
    else:
        results = engine.detect_list(
            args.data_list, include_frame_topk=include_frame_topk)

    score_file = write_score_file(results, args.score_file)
    detail_file = write_detail_file(results, args.detail_file)
    print_summary(results, score_file, engine, detail_file=detail_file)


if __name__ == "__main__":
    main()

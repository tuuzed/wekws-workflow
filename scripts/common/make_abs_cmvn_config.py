#!/usr/bin/env python3
"""Copy a trained config.yaml with an absolute CMVN path.

Training saves `model.cmvn.cmvn_file` as a recipe-relative path such as
`data/train/global_cmvn`. Downstream tools (score/export/infer) would otherwise
resolve it against the current working directory, which no longer exists once
the artifacts move to work/. Keep the training artifact untouched and emit a
derived config instead.
"""

import argparse

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('dst')
    parser.add_argument('cmvn')
    args = parser.parse_args()

    with open(args.src, encoding='utf-8') as fin:
        config = yaml.safe_load(fin)
    config.setdefault('model', {}).setdefault('cmvn', {})['cmvn_file'] = args.cmvn
    with open(args.dst, 'w', encoding='utf-8') as fout:
        yaml.safe_dump(config, fout, sort_keys=False, allow_unicode=True)
    print(f'wrote {args.dst}')


if __name__ == '__main__':
    main()

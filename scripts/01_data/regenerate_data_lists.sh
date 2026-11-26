#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"

OLD_PREFIX="$WEKWS_ROOT/examples/hi_xiaowen/s0/data/"
NEW_PREFIX="$EXAMPLE_DATA/"

for split in train dev test; do
    d="$EXAMPLE_DATA/$split"
    kws_require_file "$d/wav.scp"
    kws_require_file "$d/text"
    kws_require_file "$d/wav.dur"

    sed "s#$OLD_PREFIX#$NEW_PREFIX#g" "$d/wav.scp" > "$d/wav.scp.tmp"
    mv "$d/wav.scp.tmp" "$d/wav.scp"

    # wav_to_duration.sh leaves intermediate slices containing the old recipe
    # path. They are not needed once wav.dur/data.list exist.
    rm -rf "$d/log"

    "$PYTHON_BIN" "$WEKWS_ROOT/tools/make_list.py" \
        "$d/wav.scp" "$d/text" "$d/wav.dur" "$d/data.list"
done

if grep -R -n "$OLD_PREFIX" "$EXAMPLE_DATA"; then
    kws_die "old upstream data path still present in $EXAMPLE_DATA"
fi

"$PYTHON_BIN" - <<'PY'
import json
import os

for split in ('train', 'dev', 'test'):
    path = os.path.join(os.environ['EXAMPLE_DATA'], split, 'data.list')
    with open(path, encoding='utf-8') as fin:
        for lineno, line in enumerate(fin, 1):
            item = json.loads(line)
            wav = item['wav']
            if not wav.startswith(os.environ['EXAMPLE_DATA'] + os.sep):
                raise SystemExit(f'{path}:{lineno}: wav outside work: {wav}')
            if not os.path.isfile(wav):
                raise SystemExit(f'{path}:{lineno}: wav missing: {wav}')
    print(f'{path}: ok')
PY

#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"

TOKENS="$MOBVOI_TRANSCRIPTION_DIR/tokens.txt"
kws_require_file "$TOKENS"

mkdir -p "$EXAMPLE_DICT"

# Same transformation as the upstream recipe: token ids are 1-based in
# tokens.txt; the tokenizer dict is 0-based. `&` is the unknown/filler token.
awk '{print $1, $2-1}' "$TOKENS" > "$EXAMPLE_DICT/dict.txt"
sed -i 's/& 1/<filler> 1/' "$EXAMPLE_DICT/dict.txt"

printf '<SILENCE>\n<EPS>\n<BLK>\n' > "$EXAMPLE_DICT/words.txt"

"$PYTHON_BIN" - <<'PY'
import os

dict_path = os.path.join(os.environ['EXAMPLE_DICT'], 'dict.txt')
expected = {
    '你': 1027,
    '嗨': 1302,
    '好': 1401,
    '小': 1462,
    '问': 2494,
    '<filler>': 1,
}
actual = {}
with open(dict_path, encoding='utf-8') as fin:
    for line in fin:
        parts = line.split()
        if len(parts) == 2:
            actual[parts[0]] = int(parts[1])
for token, token_id in expected.items():
    if actual.get(token) != token_id:
        raise SystemExit(f'{dict_path}: expected {token}={token_id}, got {actual.get(token)}')
if '&' in actual:
    raise SystemExit(f'{dict_path}: raw & token still present')
print('dict token ids: ok')
PY

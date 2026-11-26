#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/onnxruntime"
VERSION="${ORT_VERSION:-1.20.0}"
URL="https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${VERSION}/onnxruntime-android-${VERSION}.aar"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading ONNX Runtime Android ${VERSION}"
curl -fL --retry 3 "$URL" -o "$TMP/ort.aar"
unzip -q -o "$TMP/ort.aar" -d "$TMP/aar"

rm -rf "$DEST/include" "$DEST/lib"
mkdir -p "$DEST/include" "$DEST/lib"
cp -a "$TMP/aar/headers/." "$DEST/include/"

for abi in arm64-v8a armeabi-v7a x86 x86_64; do
    mkdir -p "$DEST/lib/$abi"
    cp "$TMP/aar/jni/$abi/libonnxruntime.so" "$DEST/lib/$abi/"
done

echo "ORT Mobile installed to $DEST"
echo "  headers: $DEST/include"
echo "  libs:    $DEST/lib/<abi>/libonnxruntime.so"

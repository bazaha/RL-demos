#!/usr/bin/env bash
# Stage the exported Core ML model into the iOS app target, and verify that the
# app and the exporter still agree about it.
#
# Two blockers made this script necessary, both of which it now catches:
#
#   1. The app's .mlpackage was committed with only its Manifest.json -- the
#      bare `data/` line in .gitignore matched the package's internal `Data/`
#      directory (macOS git is case-insensitive), so the model spec and the
#      21 MB of weights were never in git. `coremlc` failed on every clone.
#   2. The exporter was rewritten (commit 9be47b3) to name its input tensor "x"
#      while the app still fed a feature named "board". Nothing checked, so the
#      break only showed up as a red badge at runtime.
#
# Usage:
#   scripts/refresh_ios_model.sh            export if needed, then stage
#   scripts/refresh_ios_model.sh --check    verify only (used as a build phase)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="GomokuAZ_b1"
SRC="$ROOT/results/coreml_export/$MODEL.mlpackage"
DST_DIR="$ROOT/ios/Gomoku15/Models"
DST="$DST_DIR/$MODEL.mlpackage"
CKPT="${CAL_CKPT:-$ROOT/results/gomoku_ckpt_p15/iter040.pt}"
# basename of the checkpoint the staged model must have come from
CKPT_TAG="$(basename "${CAL_CKPT:-results/gomoku_ckpt_p15/iter040.pt}")"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

die() { echo "error: $*" >&2; exit 1; }

# --- the app and the exporter must agree on the input feature name -----------
# A mismatch here is invisible until the first prediction throws at runtime.
exporter_name=$(sed -n 's/.*ct\.TensorType(name="\([^"]*\)".*/\1/p' \
    "$ROOT/scripts/export_gomoku_coreml.py" | head -1)
swift_names=$(sed -n 's/.*inputDescriptionsByName\["\([^"]*\)"\].*/\1/p;
                      s/.*dictionary: \["\([^"]*\)": MLFeatureValue.*/\1/p' \
    "$ROOT/ios/GomokuEngine/Sources/GomokuEngine/AZNet.swift" | sort -u)
[ -n "$exporter_name" ] || die "could not read the input tensor name out of scripts/export_gomoku_coreml.py"
[ -n "$swift_names" ] || die "could not read the input feature name out of AZNet.swift"
for n in $swift_names; do
    [ "$n" = "$exporter_name" ] || die \
        "input feature name mismatch: exporter emits \"$exporter_name\", AZNet.swift asks for \"$n\""
done
echo "input feature name agrees: \"$exporter_name\""

# --- export if we have nothing to stage --------------------------------------
if [ $CHECK_ONLY -eq 0 ] && [ ! -d "$SRC" ]; then
    [ -f "$CKPT" ] || die "no exported model and no checkpoint at $CKPT"
    echo "no $SRC; exporting from $(basename "$CKPT")"
    ( cd "$ROOT" && AZ_BOARD=15 AZ_CH=192 AZ_BLOCKS=12 \
        uv run python scripts/export_gomoku_coreml.py )
fi

# --- stage -------------------------------------------------------------------
if [ $CHECK_ONLY -eq 0 ]; then
    [ -d "$SRC" ] || die "$SRC missing"
    mkdir -p "$DST_DIR"
    rm -rf "$DST"
    cp -R "$SRC" "$DST"
    echo "staged -> ${DST#"$ROOT"/}"
    # The reference vectors must move with the model: they are what the app's
    # boot self-test compares against, and a model from one export paired with
    # vectors from another is a red badge nobody asked for.
    if [ -f "$(dirname "$SRC")/testvec.json" ]; then
        cp "$(dirname "$SRC")/testvec.json" "$ROOT/ios/Gomoku15/Resources/testvec.json"
        echo "staged -> ios/Gomoku15/Resources/testvec.json"
    fi
fi

# --- verify the staged package is complete -----------------------------------
# Manifest.json alone is what shipped before; it compiles to nothing.
[ -d "$DST" ] || die "$DST missing -- run scripts/refresh_ios_model.sh"
spec="$DST/Data/com.apple.CoreML/model.mlmodel"
[ -f "$spec" ] || die "$MODEL.mlpackage has no model spec (Data/com.apple.CoreML/model.mlmodel) -- \
the package is incomplete; run scripts/refresh_ios_model.sh"
# `|| true`: under `set -e` + pipefail a failing find aborts the script before
# the die below can explain why
weights=$(find "$DST/Data/com.apple.CoreML/weights" -type f -size +1M 2>/dev/null | head -1 || true)
[ -n "$weights" ] || die "$MODEL.mlpackage has no weight blob -- \
the package is incomplete; run scripts/refresh_ios_model.sh"

# Everything below reads the STAGED ARTIFACT, not the sources. The source-to-
# source check above catches a rename made in both files; only the spec itself
# catches a rename made in the sources without re-exporting -- the normal case,
# since results/coreml_export/** is gitignored.
report=$(python3 "$ROOT/scripts/inspect_mlpackage.py" "$DST" "$exporter_name" "$CKPT_TAG" || true)
case "$report" in
    OK*)   echo "${report#OK }" ;;
    FAIL*) die "${report#FAIL }" ;;
    *)     die "could not inspect $DST (python3 said: ${report:-nothing})" ;;
esac

size=$(du -sh "$DST" | cut -f1)
echo "staged model ok: ${DST#"$ROOT"/} ($size)"

# --- drift warning -----------------------------------------------------------
if [ -d "$SRC" ] && [ "$SRC/Manifest.json" -nt "$DST/Manifest.json" ]; then
    echo "warning: $SRC is newer than the staged copy; run scripts/refresh_ios_model.sh" >&2
fi

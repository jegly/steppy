#!/usr/bin/env bash
# Build a self-contained Steppy .deb (ACE-Step 1.5 engine).
#
#   ./build_deb.sh 32gb     standard  — fp32-ish, up to 10 min tracks
#   ./build_deb.sh 16gb     low-mem   — bf16 + int8, 3 min cap, MemoryHigh
#
# App code comes from this folder. The python runtime is assembled fresh:
# soundforge/runtime is the *relocatable* interpreter, but its site-packages
# carry torch 2.12 for stable-audio-3 while ACE-Step pins 2.10 — so we take the
# interpreter and graft ACE-Step's own site-packages on top.
set -euo pipefail

VARIANT="${1:-32gb}"
case "$VARIANT" in 16gb|32gb) ;; *) echo "usage: $0 [16gb|32gb]" >&2; exit 1;; esac

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
PKG=steppy
VERSION="${VERSION:-1.0.0}"
ARCH=amd64
STAGE="$HERE/build/${PKG}_${VERSION}-${VARIANT}_${ARCH}"
OPT="$STAGE/opt/$PKG"
DEB="$HERE/${PKG}_${VERSION}-${VARIANT}_${ARCH}.deb"

BASE_RUNTIME="${BASE_RUNTIME:-$ROOT/runtime}"
ACESTEP_SRC="${ACESTEP_SRC:-$HOME/acestep-test/ACE-Step-1.5}"
ACESTEP_SP="${ACESTEP_SP:-$HOME/acestep-test/venv/lib/python3.12/site-packages}"

# dpkg-deb stages the compressed archive in $TMPDIR. /tmp is a tmpfs (RAM) on
# these machines and this archive is ~7GB — keep it on real disk.
export TMPDIR="$HERE/build/tmp"; mkdir -p "$TMPDIR"

echo ">> [1/9] verify inputs"
for p in "$BASE_RUNTIME/bin/python3.12" "$ACESTEP_SRC/acestep" "$ACESTEP_SP" \
         "$ACESTEP_SRC/checkpoints/acestep-v15-turbo" \
         "$ACESTEP_SRC/checkpoints/Qwen3-Embedding-0.6B" \
         "$ACESTEP_SRC/checkpoints/vae" \
         "$ACESTEP_SRC/checkpoints/acestep-5Hz-lm-0.6B"; do
  [ -e "$p" ] || { echo "FATAL: missing $p" >&2; exit 1; }
done
for f in gui.py worker_acestep.py theme.py config.py; do
  [ -f "$HERE/$f" ] || { echo "FATAL: missing $HERE/$f" >&2; exit 1; }
done
echo "   ok"

echo ">> [2/9] clean stage"
rm -rf "$STAGE"
mkdir -p "$OPT"/{icons,fonts,acestep} "$STAGE/DEBIAN" "$STAGE/usr/bin" \
         "$STAGE/usr/share/applications" \
         "$STAGE/usr/share/icons/hicolor/512x512/apps" \
         "$STAGE/usr/share/icons/hicolor/scalable/apps"

echo ">> [3/9] app code, fonts, icons"
cp "$HERE"/gui.py "$HERE"/worker_acestep.py "$HERE"/theme.py "$HERE"/config.py "$OPT"/
cp "$HERE"/fonts/*.ttf "$OPT/fonts/" 2>/dev/null || true
mkdir -p "$OPT/fonts/licenses"
cp "$HERE"/fonts/licenses/*.txt "$OPT/fonts/licenses/" 2>/dev/null || true
cp "$HERE"/LICENSE "$HERE"/THIRD-PARTY-LICENSES.md "$OPT/" 2>/dev/null || true
cp "$HERE"/icons/*.svg "$OPT/icons/" 2>/dev/null || true

echo ">> [4/9] assemble the runtime (interpreter + ACE-Step deps)"
cp -a "$BASE_RUNTIME" "$OPT/runtime"
[ -L "$OPT/runtime" ] && { echo "FATAL: runtime is a symlink" >&2; exit 1; }
SP="$OPT/runtime/lib/python3.12/site-packages"
find "$SP" -mindepth 1 -maxdepth 1 ! -name 'pip*' ! -name 'setuptools*' \
     ! -name 'pkg_resources' ! -name '_distutils_hack*' ! -name 'wheel*' \
     -exec rm -rf {} + 2>/dev/null || true
cp -a "$ACESTEP_SP/." "$SP/"
find "$OPT/runtime" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "   runtime: $(du -sh "$OPT/runtime" | cut -f1)"

echo ">> [5/9] verify the STAGED runtime imports everything"
"$OPT/runtime/bin/python3.12" - <<'PY'
import sys
mods = ["torch","torchaudio","transformers","diffusers","einops","safetensors",
        "soundfile","numpy","scipy","matplotlib","PIL","loguru","torchao"]
bad=[]
for m in mods:
    try: __import__(m)
    except Exception as e: bad.append(f"{m}({type(e).__name__})")
if bad: print("   FATAL missing:", ", ".join(bad)); sys.exit(1)
import torch; print(f"   staged runtime OK — torch {torch.__version__}")
PY

echo ">> [6/9] ACE-Step package + checkpoints (1.7B LM deliberately excluded)"
cp -a "$ACESTEP_SRC/acestep" "$OPT/acestep/acestep"
for f in cli.py pyproject.toml LICENSE; do
  [ -e "$ACESTEP_SRC/$f" ] && cp -a "$ACESTEP_SRC/$f" "$OPT/acestep/" || true
done
mkdir -p "$OPT/acestep/checkpoints"
for d in acestep-v15-turbo Qwen3-Embedding-0.6B vae acestep-5Hz-lm-0.6B config.json README.md; do
  [ -e "$ACESTEP_SRC/checkpoints/$d" ] && cp -a "$ACESTEP_SRC/checkpoints/$d" "$OPT/acestep/checkpoints/"
done
find "$OPT/acestep" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

# ACE-Step hard-codes the 1.7B LM in MAIN_MODEL_COMPONENTS and DEFAULT_LM_MODEL.
# We ship the 0.6B instead (1.23GB vs 3.45GB, and the 1.7B could not finish a
# lyric in 20+ min on CPU). Without this patch check_main_model_exists() fails,
# and ACE-Step decides the model is missing and tries to DOWNLOAD it.
MD="$OPT/acestep/acestep/model_downloader.py"
sed -i 's/"acestep-5Hz-lm-1\.7B",     # Default LM model (1.7B)/"acestep-5Hz-lm-0.6B",     # shipped LM (see build_deb.sh)/' "$MD"
sed -i 's/^DEFAULT_LM_MODEL = "acestep-5Hz-lm-1\.7B"/DEFAULT_LM_MODEL = "acestep-5Hz-lm-0.6B"/' "$MD"
grep -q '"acestep-5Hz-lm-0.6B",     # shipped LM' "$MD" || { echo "FATAL: MAIN_MODEL_COMPONENTS patch did not apply" >&2; exit 1; }
grep -q '^DEFAULT_LM_MODEL = "acestep-5Hz-lm-0.6B"' "$MD" || { echo "FATAL: DEFAULT_LM_MODEL patch did not apply" >&2; exit 1; }
echo "   patched model_downloader.py -> 0.6B LM, no download path"
rm -rf "$OPT/acestep/checkpoints"/*/.cache "$OPT/acestep/checkpoints/.cache" 2>/dev/null || true
echo "   checkpoints: $(du -sh "$OPT/acestep/checkpoints" | cut -f1)"

echo ">> [7/9] launcher / desktop / icons"
cp "$HERE/packaging/steppy.$VARIANT.launcher" "$STAGE/usr/bin/$PKG"
chmod 755 "$STAGE/usr/bin/$PKG"
cp "$HERE/packaging/com.steppy.app.desktop" "$STAGE/usr/share/applications/"
cp "$HERE/packaging/com.steppy.app.png" "$STAGE/usr/share/icons/hicolor/512x512/apps/"
cp "$HERE/icons/steppy-tray-symbolic.svg" "$STAGE/usr/share/icons/hicolor/scalable/apps/" 2>/dev/null || true
cp "$HERE/packaging/postinst" "$STAGE/DEBIAN/postinst"; chmod 755 "$STAGE/DEBIAN/postinst"
sed 's/^/   /' "$STAGE/usr/bin/$PKG" | head -6

echo ">> [8/9] control"
find "$OPT" "$STAGE/usr" -type d -exec chmod a+rx {} +
find "$OPT" "$STAGE/usr" -type f -exec chmod a+r  {} +
chmod 755 "$STAGE/usr/bin/$PKG" "$OPT/runtime/bin/"* 2>/dev/null || true
if [ "$VARIANT" = "16gb" ]; then
  H="Steppy (16GB low-memory build) - offline AI music generator with lyrics"
  B=" LOW-MEMORY VARIANT: the DiT is int8-quantised and loaded in bfloat16, the
 5Hz lyric LM runs in a short-lived subprocess so it is never resident beside
 the DiT, and track length is capped at 3 minutes. Peak ~8GB. The launcher sets
 MemoryHigh so a long render throttles instead of triggering systemd-oomd."
else
  H="Steppy (32GB standard build) - offline AI music generator with lyrics"
  B=" STANDARD VARIANT: full precision, unchunked attention and track lengths up
 to 10 minutes. Peak ~12GB. Requires a 32GB-RAM machine."
fi
cat > "$STAGE/DEBIAN/control" <<CTL
Package: $PKG
Version: $VERSION-$VARIANT
Architecture: $ARCH
Maintainer: Steppy <jjjegly@gmail.com>
Installed-Size: $(du -sk "$STAGE" | cut -f1)
Depends: python3, python3-gi, gir1.2-gtk-4.0, gir1.2-adw-1, gstreamer1.0-plugins-base, gstreamer1.0-plugins-good
Section: sound
Priority: optional
Description: $H
 Offline text-to-audio desktop app built on ACE-Step 1.5. Generates full songs
 with sung lyrics from a text prompt; the bundled 5Hz language model can write
 the lyrics for you. Model, Python runtime and all dependencies are bundled -
 no GPU and no network required.
$B
CTL

echo ">> [9/9] build (TMPDIR=$TMPDIR)"
echo "   payload: $(du -sh "$STAGE" | cut -f1)"
rm -f "$DEB"
dpkg-deb --root-owner-group -Zgzip -z1 --build "$STAGE" "$DEB"
rm -rf "$STAGE"
echo
ls -la "$DEB" | awk '{printf " built: %s  (%.2f GiB)\n", $9, $5/1073741824}'
echo " verify: dpkg-deb -c '$DEB' >/dev/null && echo OK"

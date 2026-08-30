# Third-party licences

Steppy bundles the following components. All are permissively licensed and
compatible with redistribution and commercial use.

## Models and weights

| component | licence | source |
|---|---|---|
| ACE-Step 1.5 (code) | MIT — © 2026 ACEStep | github.com/ace-step/ACE-Step-1.5 |
| ACE-Step 1.5 (weights) | MIT | huggingface.co/ACE-Step/Ace-Step1.5 |
| acestep-5Hz-lm-0.6B | MIT | huggingface.co/ACE-Step/acestep-5Hz-lm-0.6B |
| Qwen3-Embedding-0.6B | Apache 2.0 | huggingface.co/Qwen/Qwen3-Embedding-0.6B |

## Python runtime and libraries

| component | licence |
|---|---|
| CPython 3.12 | Python Software Foundation License |
| PyTorch, torchaudio | BSD-3-Clause |
| transformers, diffusers, safetensors, accelerate | Apache 2.0 |
| numpy | BSD-3-Clause (with 0BSD, MIT, Zlib components) |
| scipy | BSD-3-Clause |
| soundfile | BSD-3-Clause |
| matplotlib | matplotlib licence (BSD-compatible) |
| Pillow | MIT-CMU |
| loguru | MIT |
| einops | MIT |
| torchao | BSD-3-Clause |

## Fonts

Bundled under the SIL Open Font License 1.1:

- DotGothic16, Gugi, Orbitron, Press Start 2P, VT323,
  Playfair Display, Space Grotesk, Geist Pixel

Each family's licence text ships in [`fonts/licenses/`](fonts/licenses/), one
file per family, retrieved from Google Fonts. Each has been verified against the
copyright string embedded in the corresponding `.ttf`.

The OFL permits bundling and redistribution. Its one constraint that matters
here: Reserved Font Names may not be reused for modified versions of the fonts.
Steppy ships them unmodified, so this does not apply.

## Not bundled

Steppy does **not** include stable-audio-3 or its weights. Those are governed by
the Stability AI Community License, which carries a USD $1,000,000 annual revenue
limit and attribution requirements, and the weights are gated on HuggingFace.
Steppy uses ACE-Step exclusively.

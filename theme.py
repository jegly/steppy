"""Theme engine for Steppy.

Palette manifests compiled into GTK CSS through a single user-priority
``Gtk.CssProvider``, layered over libadwaita's ``StyleManager`` light/dark base.
Live switching, custom accent override, density, glow.

The core technique (ported from the tesseract GUI): instead of styling every
widget by hand, each palette redefines libadwaita's named CSS variables
(``--window-bg-color``, ``--accent-bg-color``, ``--card-bg-color`` …) at
``:root``. Every Adwaita widget already reads those, so ~18 colours re-skin the
whole app. Requires libadwaita >= 1.6 (we ship against 1.9).

User themes drop into ``~/.config/steppy/themes/*.toml`` with the same
field schema as ``Palette``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, fields

try:  # Python 3.11+ stdlib; system python here is 3.14
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


@dataclass
class Palette:
    id: str = "follow-system"
    name: str = "Follow System"
    dark: bool = False
    # Pure libadwaita look (no palette override), only accent applies.
    follow_system: bool = True

    window_bg: str = ""
    view_bg: str = ""
    surface: str = ""
    surface_alt: str = ""
    headerbar: str = ""
    sidebar: str = ""
    card: str = ""
    popover: str = ""
    text: str = ""
    text_dim: str = ""
    accent: str = ""
    accent_fg: str = ""
    accent2: str = ""  # secondary accent (chips, gradients, neon highlights)
    success: str = ""
    warning: str = ""
    error: str = ""
    border: str = ""

    radius: int = 12  # corner radius (px) for cards/dialogs; buttons use pill
    glow: bool = False  # neon glow box-shadows (cyberpunk)
    serif: bool = False  # serif accent font for body (vintage)


_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_COLOR_FIELDS = {
    "window_bg", "view_bg", "surface", "surface_alt", "headerbar", "sidebar",
    "card", "popover", "text", "text_dim", "accent", "accent_fg", "accent2",
    "success", "warning", "error", "border",
}


def _sanitize_theme_data(data):
    """Validate a user theme manifest before it reaches Palette/CSS.

    Color fields are spliced verbatim into GTK CSS in compile_css(), so a
    malformed value (accidental or crafted) could inject arbitrary CSS
    declarations. Reject anything that isn't a plain #rrggbb hex color;
    the field then falls back to the Palette dataclass default.
    """
    out = {}
    for k, v in data.items():
        if k in _COLOR_FIELDS:
            if isinstance(v, str) and _HEX_COLOR_RE.match(v):
                out[k] = v
            continue
        out[k] = v
    return out


def _p(id, name, *, dark, win, view, surface, alt, header, side, card, pop,
       text, dim, accent, accent_fg, accent2, ok, warn, err, border,
       radius=12, glow=False, serif=False):
    return Palette(
        id=id, name=name, dark=dark, follow_system=False,
        window_bg=win, view_bg=view, surface=surface, surface_alt=alt,
        headerbar=header, sidebar=side, card=card, popover=pop,
        text=text, text_dim=dim, accent=accent, accent_fg=accent_fg,
        accent2=accent2, success=ok, warning=warn, error=err, border=border,
        radius=radius, glow=glow, serif=serif,
    )


def builtin_themes():
    return [
        Palette(),  # follow-system
        _p("grass", "Grass", dark=True,
           win="#13773d", view="#0f6234", surface="#1c8a4a", alt="#239a55",
           header="#0f6234", side="#126b38", card="#188044", pop="#1c8a4a",
           text="#fff0a5", dim="#bcd6a0",
           accent="#e7b000", accent_fg="#13773d", accent2="#7fd9b0",
           ok="#9bea6a", warn="#e7b000", err="#cf3a2a", border="#2a9a5e"),
        _p("dracula", "Dracula", dark=True,
           win="#282a36", view="#21222c", surface="#343746", alt="#3c3f51",
           header="#21222c", side="#262833", card="#313342", pop="#343746",
           text="#f8f8f2", dim="#9ea8c7",
           accent="#bd93f9", accent_fg="#1c1d26", accent2="#ff79c6",
           ok="#50fa7b", warn="#f1fa8c", err="#ff5555", border="#44475a"),
        _p("catppuccin-latte", "Catppuccin Latte", dark=False,
           win="#eff1f5", view="#ffffff", surface="#e6e9ef", alt="#dce0e8",
           header="#e6e9ef", side="#e9ecf2", card="#ffffff", pop="#eff1f5",
           text="#4c4f69", dim="#6c6f85",
           accent="#8839ef", accent_fg="#ffffff", accent2="#ea76cb",
           ok="#40a02b", warn="#df8e1d", err="#d20f39", border="#ccd0da"),
        _p("catppuccin-frappe", "Catppuccin Frappé", dark=True,
           win="#303446", view="#292c3c", surface="#414559", alt="#51576d",
           header="#292c3c", side="#2e3244", card="#3b3f54", pop="#414559",
           text="#c6d0f5", dim="#a5adce",
           accent="#ca9ee6", accent_fg="#232634", accent2="#f4b8e4",
           ok="#a6d189", warn="#e5c890", err="#e78284", border="#51576d"),
        _p("catppuccin-macchiato", "Catppuccin Macchiato", dark=True,
           win="#24273a", view="#1e2030", surface="#363a4f", alt="#494d64",
           header="#1e2030", side="#222539", card="#2f3247", pop="#363a4f",
           text="#cad3f5", dim="#a5adcb",
           accent="#c6a0f6", accent_fg="#181926", accent2="#f5bde6",
           ok="#a6da95", warn="#eed49f", err="#ed8796", border="#494d64"),
        _p("catppuccin-mocha", "Catppuccin Mocha", dark=True,
           win="#1e1e2e", view="#181825", surface="#313244", alt="#45475a",
           header="#181825", side="#1c1c2c", card="#2a2a3c", pop="#313244",
           text="#cdd6f4", dim="#a6adc8",
           accent="#cba6f7", accent_fg="#11111b", accent2="#f5c2e7",
           ok="#a6e3a1", warn="#f9e2af", err="#f38ba8", border="#45475a"),
        _p("vintage-light", "Vintage Light", dark=False,
           win="#f6efe1", view="#fbf6ea", surface="#efe5d0", alt="#e7dabf",
           header="#efe5d0", side="#f1e9d7", card="#fbf6ea", pop="#f3ecdc",
           text="#46392b", dim="#7a6a55",
           accent="#b07d3a", accent_fg="#fff8ec", accent2="#4f7c74",
           ok="#5f7d4f", warn="#b07d3a", err="#a14d3a", border="#d8c8a8",
           radius=14, serif=True),
        _p("neon-tessera", "Neon Tessera", dark=True,
           win="#0a0e14", view="#070a10", surface="#11161f", alt="#161d29",
           header="#0a0e14", side="#0d1118", card="#10151e", pop="#131923",
           text="#d8e6f2", dim="#7e93a8",
           accent="#00e5ff", accent_fg="#03131a", accent2="#ff2ec4",
           ok="#00ff9c", warn="#ffc400", err="#ff3860", border="#1d2735",
           radius=10, glow=True),
        _p("adventure-time", "Adventure Time", dark=True,
           win="#1f1d45", view="#17152f", surface="#2a2755", alt="#34306a",
           header="#17152f", side="#1b1940", card="#252253", pop="#2a2755",
           text="#f8dcc0", dim="#a39ac4",
           accent="#e7741e", accent_fg="#1f1d45", accent2="#5cf9ff",
           ok="#4ab118", warn="#e7b000", err="#bd0013", border="#3a356f"),
        _p("borland", "Borland", dark=True,
           win="#0000a4", view="#000084", surface="#0a1ab0", alt="#1730c0",
           header="#000084", side="#00118f", card="#0817ac", pop="#0a1ab0",
           text="#ffff80", dim="#b6b6e6",
           accent="#ffff4e", accent_fg="#0000a4", accent2="#4fe9fc",
           ok="#4efa78", warn="#ffff4e", err="#ff5959", border="#2a40c4",
           radius=8),
        _p("c64", "Commodore 64", dark=True,
           win="#40318d", view="#352978", surface="#4d3ea0", alt="#5a4bb0",
           header="#352978", side="#3a2e85", card="#473a98", pop="#4d3ea0",
           text="#cabdf2", dim="#9385c9",
           accent="#bfce72", accent_fg="#40318d", accent2="#67b6bd",
           ok="#55a049", warn="#bfce72", err="#883932", border="#5648a8",
           radius=8),
        _p("fairy-floss-dark", "Fairy Floss Dark", dark=True,
           win="#3b364c", view="#332f42", surface="#4a4564", alt="#56506f",
           header="#332f42", side="#3d3850", card="#453f5c", pop="#4a4564",
           text="#f8f8f2", dim="#c5bdda",
           accent="#ffb8d1", accent_fg="#3b364c", accent2="#c5a3ff",
           ok="#c2ffdf", warn="#ffea00", err="#ff857f", border="#564f6f",
           radius=14),
        _p("flat", "Flat", dark=True,
           win="#2c3e50", view="#243342", surface="#34495e", alt="#3e5870",
           header="#243342", side="#2a3a4a", card="#324356", pop="#34495e",
           text="#ecf0f1", dim="#a4b5c4",
           accent="#3498db", accent_fg="#ffffff", accent2="#9b59b6",
           ok="#2ecc71", warn="#f1c40f", err="#e74c3c", border="#3e5066"),
        _p("gogh", "Gogh — Starry Night", dark=True,
           win="#0d1b34", view="#0a1628", surface="#14264a", alt="#1b3260",
           header="#0a1628", side="#0f1d38", card="#122243", pop="#14264a",
           text="#e8eeff", dim="#94a8cc",
           accent="#f4cd3a", accent_fg="#0d1b34", accent2="#5b8dd9",
           ok="#6bbf59", warn="#f4cd3a", err="#d9603b", border="#21345f"),
        _p("gruvbox-material", "Gruvbox Material", dark=True,
           win="#282828", view="#1f1f1f", surface="#32302f", alt="#3c3836",
           header="#1f1f1f", side="#252423", card="#2f2d2c", pop="#32302f",
           text="#d4be98", dim="#a89984",
           accent="#d8a657", accent_fg="#282828", accent2="#7daea3",
           ok="#a9b665", warn="#d8a657", err="#ea6962", border="#45403d"),
        _p("homebrew", "Homebrew", dark=True,
           win="#000000", view="#050505", surface="#0c140c", alt="#122012",
           header="#000000", side="#040804", card="#0a120a", pop="#0c140c",
           text="#00d000", dim="#1f8a1f",
           accent="#00ff00", accent_fg="#001500", accent2="#00d8b2",
           ok="#00c800", warn="#9a9a00", err="#c80000", border="#103810",
           radius=8, glow=True),
        _p("ocean", "Ocean", dark=True,
           win="#2b303b", view="#232831", surface="#343d46", alt="#3e4855",
           header="#232831", side="#2a2f39", card="#313844", pop="#343d46",
           text="#c0c5ce", dim="#8b95a4",
           accent="#8fa1b3", accent_fg="#1b2027", accent2="#b48ead",
           ok="#a3be8c", warn="#ebcb8b", err="#bf616a", border="#3e4855"),
        _p("kokuban", "Kokuban", dark=True,
           win="#1f3526", view="#192c1f", surface="#274030", alt="#2f4c39",
           header="#192c1f", side="#1d3123", card="#243c2d", pop="#274030",
           text="#f0f0e8", dim="#a9c2af",
           accent="#f2e9c8", accent_fg="#1f3526", accent2="#f2b4b4",
           ok="#a8d8a0", warn="#f0e68c", err="#f2a0a0", border="#315040"),
        _p("mono-cyan", "Mono Cyan", dark=True,
           win="#081414", view="#040e0e", surface="#0e1f1f", alt="#143030",
           header="#040e0e", side="#0a1818", card="#0c1c1c", pop="#0e1f1f",
           text="#c8f0f0", dim="#5c9a9a",
           accent="#00d0d0", accent_fg="#021616", accent2="#5ce0e0",
           ok="#00d0a0", warn="#80e0e0", err="#e08585", border="#163838",
           radius=10, glow=True),
    ]


def all_themes(themes_dir=None):
    """Built-ins + user manifests (``*.toml``) from the themes dir."""
    themes = builtin_themes()
    if themes_dir and tomllib and os.path.isdir(themes_dir):
        valid = {f.name for f in fields(Palette)}
        for fn in sorted(os.listdir(themes_dir)):
            if not fn.endswith(".toml"):
                continue
            try:
                with open(os.path.join(themes_dir, fn), "rb") as fh:
                    data = tomllib.load(fh)
            except Exception:
                continue
            data = _sanitize_theme_data(data)
            p = Palette(**{k: v for k, v in data.items() if k in valid})
            if not p.id:
                p.id = fn[:-5]
            p.follow_system = False
            themes.append(p)
    return themes


def find_theme(theme_id, themes_dir=None):
    for t in all_themes(themes_dir):
        if t.id == theme_id:
            return t
    return Palette()


def _alpha(hex_str, a):
    h = hex_str.lstrip("#")
    if len(h) != 6:
        return f"alpha(currentColor, {a})"
    try:
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return f"alpha(currentColor, {a})"
    return f"rgba({r}, {g}, {b}, {a:.2f})"


def compile_css(p: Palette, accent_override="", glow_intensity=60,
                density="comfortable", font=""):
    """Compile a palette + user prefs into a GTK CSS string."""
    accent = accent_override if _HEX_COLOR_RE.match(accent_override or "") else p.accent

    css = []

    # ---- palette: override libadwaita's named CSS variables (adw >= 1.6) ----
    if not p.follow_system:
        afg = p.accent_fg
        css.append(f""":root {{
  --window-bg-color: {p.window_bg};
  --window-fg-color: {p.text};
  --view-bg-color: {p.view_bg};
  --view-fg-color: {p.text};
  --headerbar-bg-color: {p.headerbar};
  --headerbar-fg-color: {p.text};
  --headerbar-backdrop-color: {p.window_bg};
  --sidebar-bg-color: {p.sidebar};
  --sidebar-fg-color: {p.text};
  --sidebar-backdrop-color: {p.window_bg};
  --secondary-sidebar-bg-color: {p.sidebar};
  --secondary-sidebar-fg-color: {p.text};
  --card-bg-color: {p.card};
  --card-fg-color: {p.text};
  --dialog-bg-color: {p.surface};
  --dialog-fg-color: {p.text};
  --popover-bg-color: {p.popover};
  --popover-fg-color: {p.text};
  --success-color: {p.success};
  --success-bg-color: {p.success};
  --success-fg-color: {afg};
  --warning-color: {p.warning};
  --warning-bg-color: {p.warning};
  --warning-fg-color: {afg};
  --error-color: {p.error};
  --error-bg-color: {p.error};
  --error-fg-color: {afg};
  --destructive-color: {p.error};
  --destructive-bg-color: {p.error};
  --destructive-fg-color: {afg};
}}""")

    if accent:
        afg = p.accent_fg if p.accent_fg else "#ffffff"
        css.append(f""":root {{
  --accent-bg-color: {accent};
  --accent-fg-color: {afg};
  --accent-color: {accent};
}}""")

    # ---- shape language: rounded cards, pill controls, soft elevation ----
    radius = p.radius
    shadow = ("0 1px 3px rgba(0,0,0,0.42), 0 4px 14px rgba(0,0,0,0.28)" if p.dark
              else "0 1px 3px rgba(60,50,40,0.10), 0 4px 16px rgba(60,50,40,0.08)")
    border = p.border if p.border else "alpha(currentColor, 0.12)"
    dim_src = p.text_dim if p.text_dim else "#888888"
    accent_c = accent if accent else "var(--accent-bg-color)"
    a2solid = p.accent2 if p.accent2 else "var(--accent-color)"
    ok = p.success if p.success else "var(--success-color)"
    warn = p.warning if p.warning else "var(--warning-color)"
    err = p.error if p.error else "var(--error-color)"
    dim = p.text_dim if p.text_dim else "alpha(currentColor, 0.6)"
    acc_src = accent if accent else "#3584e4"
    a2_src = p.accent2 if p.accent2 else "#3584e4"

    css.append(f"""
.as-card {{
  background-color: var(--card-bg-color);
  border-radius: {radius}px;
  box-shadow: {shadow};
  border: 1px solid {border};
  border-top: 2px solid {_alpha(a2_src, 0.55)};
  padding: 16px;
}}
.as-elevated {{ box-shadow: {shadow}; }}
button.pill {{ border-radius: 999px; padding-left: 22px; padding-right: 22px; }}
button {{ border-radius: 10px; }}
entry, spinbutton {{ border-radius: 10px; }}
.boxed-list {{ border-radius: {radius}px; }}
.as-chip {{
  border-radius: 999px;
  padding: 2px 10px;
  font-size: 0.82em;
  font-weight: 600;
  background-color: {_alpha(dim_src, 0.18)};
  color: var(--window-fg-color);
}}
.as-chip.ready {{ background-color: {_alpha(p.success if p.success else "#2ec27e", 0.16)}; color: {ok}; }}
.as-chip.busy {{ background-color: {_alpha(p.warning if p.warning else "#e5a50a", 0.16)}; color: {warn}; }}
.as-chip.danger {{ background-color: {_alpha(p.error if p.error else "#e01b24", 0.16)}; color: {err}; }}
.as-dim {{ color: {dim}; }}
.as-mono {{ font-family: monospace; font-size: 0.88em; color: {a2solid}; }}
.as-hero {{ font-weight: 800; font-size: 1.7em; color: {accent_c}; }}
.as-section-title {{ font-weight: 700; font-size: 1.06em; color: {a2solid}; }}
/* multi-colour roles so a theme reads as a full palette, not one accent */
levelbar block.filled {{ background-color: {accent_c}; }}
levelbar block.high {{ background-color: {ok}; }}
levelbar block.low {{ background-color: {warn}; }}
progressbar progress {{ background-color: {a2solid}; }}
checkbutton check:checked, checkbutton radio:checked {{ background-color: {a2solid}; }}
spinner {{ color: {accent_c}; }}
switch:checked {{ background-color: {accent_c}; }}
scale highlight {{ background-color: {accent_c}; }}
scale slider {{ background-color: {accent_c}; }}
.as-drop-zone {{
  border: 2px dashed {border};
  border-radius: {radius}px;
  padding: 28px;
  background-color: {_alpha(p.surface_alt if p.surface_alt else "#808080", 0.25)};
  transition: border-color 160ms ease, background-color 160ms ease;
}}
.as-drop-zone.hover {{ border-color: {accent_c}; background-color: {_alpha(acc_src, 0.07)}; }}
.as-help {{
  border-radius: {radius}px;
  padding: 10px 12px;
  background-color: {_alpha(acc_src, 0.10)};
  border: 1px solid {_alpha(acc_src, 0.20)};
}}
.as-help label {{ font-size: 0.92em; }}
""")

    # ---- neon glow (cyberpunk) ----
    if p.glow:
        g = min(glow_intensity, 100) / 100.0
        acc = accent if accent else p.accent
        css.append(f"""
.as-card {{
  box-shadow: 0 0 {int(10 + 14 * g)}px {_alpha(acc, 0.10 + 0.10 * g)}, 0 1px 3px rgba(0,0,0,0.5);
  border: 1px solid {_alpha(acc, 0.30)};
}}
button.suggested-action {{
  box-shadow: 0 0 {int(6 + 10 * g)}px {_alpha(acc, 0.25 + 0.25 * g)};
  text-shadow: 0 0 6px {_alpha(acc, 0.25 + 0.25 * g)};
}}
headerbar {{ border-bottom: 1px solid {_alpha(acc, 0.25)}; }}
.as-hero {{ color: {acc}; text-shadow: 0 0 12px {_alpha(acc, 0.45)}; }}
progressbar progress {{ box-shadow: 0 0 8px {_alpha(acc, 0.45)}; }}
levelbar block.filled {{ background-color: {acc}; box-shadow: 0 0 6px {_alpha(acc, 0.45)}; }}
""")

    # ---- vintage serif accent (body only) ----
    if p.serif:
        css.append("""
.as-dim, .as-help label {
  font-family: "Source Serif Pro", "Noto Serif", "Georgia", serif;
}
""")

    # ---- heading style ----
    css.append("""
.as-hero, .as-section-title, .heading,
windowtitle > .title, window > headerbar .title,
.title-1, .title-2, .title-3, .title-4 {
  letter-spacing: 0.3px;
}
""")

    # ---- density ----
    if density == "compact":
        css.append("""
listbox row { min-height: 30px; }
headerbar { min-height: 38px; }
button { padding-top: 2px; padding-bottom: 2px; }
""")

    # ---- font override ----
    if font:
        css.append(f'window {{ font-family: "{font}"; }}\n')

    return "\n".join(css)

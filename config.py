"""Persistent appearance + app settings for Steppy.

Stored as JSON in ``$XDG_CONFIG_HOME/steppy/settings.json``. User theme
manifests live alongside in ``themes/`` (see ``theme.all_themes``).
"""
from __future__ import annotations

import json
import os

APP_DIRNAME = "steppy"
_OLD_DIRNAME = "frequency"


def _migrate_old_config():
    """One-time COPY of ~/.config/frequency -> ~/.config/steppy so
    themes, settings, history and prompt pools carry over — a copy, not a move,
    because both apps are co-installable and the old one keeps its own config."""
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    old, new = os.path.join(base, _OLD_DIRNAME), os.path.join(base, APP_DIRNAME)
    if os.path.isdir(old) and not os.path.exists(new):
        try:
            import shutil
            shutil.copytree(old, new)
        except OSError:
            pass

DEFAULTS = {
    "theme": "grass",          # default look
    "accent": "",               # "" = use theme accent; else "#rrggbb" override
    "glow_intensity": 60,       # 0..100, only matters for glow themes
    "density": "comfortable",   # or "compact"
    "font": "",                 # "" = system; else a font family name
    "export_format": "wav",    # default Save/Export format: wav|flac|mp3|ogg
    "close_to_tray": True,      # closing the window keeps the app in the tray
    "lock_enabled": False,      # in-app lock screen (PIN)
    "lock_hash": "",            # "salt$sha256(salt+pin)"; empty = no PIN set
}


def config_dir():
    _migrate_old_config()
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, APP_DIRNAME)


def themes_dir():
    return os.path.join(config_dir(), "themes")


def _settings_path():
    return os.path.join(config_dir(), "settings.json")


def load():
    s = dict(DEFAULTS)
    try:
        with open(_settings_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            s.update({k: data[k] for k in DEFAULTS if k in data})
    except (OSError, ValueError):
        pass
    return s


def save(settings):
    try:
        os.makedirs(config_dir(), exist_ok=True)
        with open(_settings_path(), "w", encoding="utf-8") as fh:
            json.dump({k: settings.get(k, DEFAULTS[k]) for k in DEFAULTS},
                      fh, indent=2)
    except OSError:
        pass

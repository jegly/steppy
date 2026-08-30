#!/usr/bin/env python3
"""
Frequency — native GTK4 / libadwaita desktop GUI (runs on system Python).

White-labeled offline text-to-audio. Drives the inference worker (bundled torch
venv) over a subprocess pipe; the model loads once and stays resident.

UI mirrors the Stable Audio 3 HF Space's two-tab layout (Simple / Advanced):
prompt+negative, duration, steps/CFG, sampler params, output params, init audio,
inpainting, output spectrogram, send-to-init/inpaint. Medium model only.

Theming: a single user-priority Gtk.CssProvider fed by theme.compile_css(),
layered over libadwaita's StyleManager. See theme.py / config.py.
"""
import atexit
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk, Gio, Gdk  # noqa: E402

import config  # noqa: E402
import theme  # noqa: E402

APP_NAME = "Steppy"
APP_ID = "com.steppy.app"

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def _bundled_fonts():
    """(pretty name, path) for every bundled OFL font — drop a TTF in fonts/ to add one."""
    out = []
    try:
        for fn in sorted(os.listdir(FONTS_DIR)):
            if fn.lower().endswith((".ttf", ".otf")):
                name = re.sub(r"-?(Regular|VariableFont[_\w]*|Italic-VariableFont[_\w]*)", "",
                              os.path.splitext(fn)[0])
                name = re.sub(r"(?<!^)(?=[A-Z])", " ", name).replace("_", " ").strip() or fn
                if fn.rsplit(".", 1)[0].lower().find("italic") >= 0:
                    name += " Italic"
                out.append((name, os.path.join(FONTS_DIR, fn)))
    except OSError:
        pass
    return out

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.environ.get("SF_VENV_PY", os.path.join(HERE, "nd-convert-venv/bin/python"))
WORKER = os.environ.get("SF_WORKER", os.path.join(HERE, "worker_acestep.py"))
MODEL_DIR = os.environ.get("SF_MODEL_DIR", os.path.join(HERE, "medium"))
SA3_REPO = os.environ.get("SF_SA3_REPO", os.path.join(HERE, "stable-audio-3"))
MAX_SECONDS = 600   # ACE-Step hard limit (10 min)

# Private (mode 0700), unpredictable scratch dir for this run — avoids the
# symlink-race risk of writing to fixed /tmp/steppy_* filenames that
# another local user could pre-create.
TMP_DIR = tempfile.mkdtemp(prefix="steppy-")
atexit.register(shutil.rmtree, TMP_DIR, ignore_errors=True)


def tmp_path(name):
    return os.path.join(TMP_DIR, name)

PLACEHOLDER = ("A dream-like Synthpop instrumental that would accompany a "
               "dream-sequence in a surrealist movie 120 BPM")
SAMPLERS = ["pingpong", "euler", "rk4", "dpmpp"]

# Curated presets for the two highest-RPM YouTube music niches (Focus/Deep-Work and
# Sleep/Meditation/Wellness) plus other proven functional-music formats. Each carries
# several prompt variations so a long-form render stays varied (not a flagged loop).
PRESETS = [
    {"name": "Focus / Deep-Work", "segment": 300, "steps": 8, "cfg": 1.0, "sampler": "pingpong",
     "prompts": [
        "minimal ambient focus music, steady warm synth pads, subtle low pulse, no drums, "
        "unobtrusive and hypnotic, deep concentration, calm and spacious",
        "deep work soundscape, soft evolving textures, gentle analog drone, slow movement, "
        "no melody hooks, sustained calm, distraction-free study",
        "modern classical minimalism, sparse soft piano motifs over warm strings, slow and "
        "steady, meditative concentration, clean and quiet",
        "ambient techno for focus, very soft steady pulse, deep warm bass hum, airy pads, "
        "no vocals, flow state, understated and continuous"]},
    {"name": "Sleep / Meditation / Wellness", "segment": 320, "steps": 8, "cfg": 1.0, "sampler": "pingpong",
     "prompts": [
        "soft ambient sleep music, slow evolving warm pads, gentle and soothing, deep "
        "relaxation, no percussion, peaceful and weightless",
        "meditation soundscape, warm drone, distant soft bells, slow breathing pace, "
        "healing and calming, spacious reverb, tranquil",
        "deep sleep ambient, low warm tones, very slow swells, dark and cozy, no rhythm, "
        "dreamy and enveloping, restful",
        "wellness spa music, soft flowing pads, gentle nature-like textures, serene and "
        "delicate, slow and continuous, deeply relaxing"]},
    {"name": "Lofi Study Beats", "segment": 240, "steps": 8, "cfg": 1.0, "sampler": "pingpong",
     "prompts": [
        "warm lofi hip hop beat, mellow jazzy chords, soft vinyl crackle, relaxed boom-bap "
        "drums, cozy study vibe, 75 BPM",
        "chill lofi instrumental, dusty piano, smooth bassline, laid-back drums, rainy "
        "afternoon mood, 70 BPM",
        "nostalgic lofi, soft electric piano, gentle swing groove, warm tape saturation, "
        "calm and reflective, 72 BPM"]},
    {"name": "Ambient / Drone", "segment": 340, "steps": 8, "cfg": 1.0, "sampler": "pingpong",
     "prompts": [
        "ethereal ambient drone, slowly evolving cinematic textures, vast and spacious, "
        "calm and immersive, no rhythm",
        "deep space ambient, sustained warm pads, glacial movement, weightless and serene, "
        "expansive reverb"]},
    {"name": "Rain + Piano", "segment": 300, "steps": 8, "cfg": 1.0, "sampler": "pingpong",
     "prompts": [
        "gentle solo piano with soft rain ambience, melancholic and calm, sparse and "
        "reflective, intimate and slow",
        "tender piano melody over steady rainfall, warm and nostalgic, quiet late-night "
        "mood, soft and unhurried"]},
    {"name": "Synthwave Drive", "segment": 240, "steps": 8, "cfg": 1.2, "sampler": "pingpong",
     "prompts": [
        "retro synthwave, nostalgic analog synths, steady driving beat, neon night drive, "
        "80s, cinematic and cool",
        "outrun synthwave instrumental, pulsing bass arpeggio, dreamy lead, midnight "
        "highway mood, 100 BPM"]},
]


def _spin(lo, hi, step, value, digits=0):
    r = Adw.SpinRow.new_with_range(lo, hi, step)
    r.set_value(value)
    if digits:
        r.set_digits(digits)
    return r


def _textview(min_height=72):
    tv = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR, top_margin=8,
                      bottom_margin=8, left_margin=8, right_margin=8)
    sw = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER)
    sw.set_min_content_height(min_height)
    sw.set_child(tv)
    sw.add_css_class("card")
    return sw, tv


def _widgets(obj, *names):
    """Only the widgets that actually exist.

    Tabs whose engine features ACE-Step cannot provide (Continue, Long-form,
    Loop) are not built, so their widgets are never created. Live code paths
    must not assume they are there.
    """
    return [w for w in (getattr(obj, n, None) for n in names) if w is not None]


def _tv_text(tv):
    b = tv.get_buffer()
    return b.get_text(b.get_start_iter(), b.get_end_iter(), False).strip()


class _Tray:
    """Minimal StatusNotifierItem (+dbusmenu) so closing the window parks the
    app in the top-bar tray (like cascade). Pure Gio.DBus — GTK4 has no tray
    API and the appindicator libraries would drag GTK3 into the process."""

    _SNI_XML = """<node><interface name="org.kde.StatusNotifierItem">
      <property name="Category" type="s" access="read"/>
      <property name="Id" type="s" access="read"/>
      <property name="Title" type="s" access="read"/>
      <property name="Status" type="s" access="read"/>
      <property name="IconName" type="s" access="read"/>
      <property name="IconThemePath" type="s" access="read"/>
      <property name="Menu" type="o" access="read"/>
      <property name="ItemIsMenu" type="b" access="read"/>
      <property name="ToolTip" type="(sa(iiay)ss)" access="read"/>
      <signal name="NewToolTip"/>
      <method name="Activate"><arg type="i"/><arg type="i"/></method>
      <method name="SecondaryActivate"><arg type="i"/><arg type="i"/></method>
      <method name="ContextMenu"><arg type="i"/><arg type="i"/></method>
      <method name="Scroll"><arg type="i"/><arg type="s"/></method>
    </interface></node>"""

    _MENU_XML = """<node><interface name="com.canonical.dbusmenu">
      <property name="Version" type="u" access="read"/>
      <property name="Status" type="s" access="read"/>
      <method name="GetLayout">
        <arg type="i" direction="in"/><arg type="i" direction="in"/>
        <arg type="as" direction="in"/>
        <arg type="u" direction="out"/><arg type="(ia{sv}av)" direction="out"/>
      </method>
      <method name="GetGroupProperties">
        <arg type="ai" direction="in"/><arg type="as" direction="in"/>
        <arg type="a(ia{sv})" direction="out"/>
      </method>
      <method name="Event">
        <arg type="i" direction="in"/><arg type="s" direction="in"/>
        <arg type="v" direction="in"/><arg type="u" direction="in"/>
      </method>
      <method name="AboutToShow"><arg type="i" direction="in"/>
        <arg type="b" direction="out"/></method>
    </interface></node>"""

    def __init__(self, app):
        self.app = app
        self.ok = False
        self.status_text = "Idle"
        self._rev = 1
        self._name = f"org.kde.StatusNotifierItem-{os.getpid()}-1"
        try:
            self._bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            sni = Gio.DBusNodeInfo.new_for_xml(self._SNI_XML).interfaces[0]
            menu = Gio.DBusNodeInfo.new_for_xml(self._MENU_XML).interfaces[0]
            self._bus.register_object("/StatusNotifierItem", sni,
                                      self._sni_call, self._sni_get, None)
            self._bus.register_object("/MenuBar", menu,
                                      self._menu_call, self._menu_get, None)
            Gio.bus_own_name_on_connection(self._bus, self._name,
                                           Gio.BusNameOwnerFlags.NONE,
                                           self._on_name, None)
        except Exception:
            self.ok = False

    def _on_name(self, _bus, _name):
        try:
            self._bus.call_sync(
                "org.kde.StatusNotifierWatcher", "/StatusNotifierWatcher",
                "org.kde.StatusNotifierWatcher", "RegisterStatusNotifierItem",
                GLib.Variant("(s)", (self._name,)), None,
                Gio.DBusCallFlags.NONE, 2000, None)
            self.ok = True
        except Exception:
            self.ok = False   # no watcher (extension off) -> close quits normally

    def _sni_get(self, _c, _s, _p, _i, prop):
        vals = {"Category": ("s", "ApplicationStatus"),
                "Id": ("s", "steppy"),
                "Title": ("s", APP_NAME),
                "Status": ("s", "Active"),
                "IconName": ("s", "steppy-tray-symbolic"),
                "IconThemePath": ("s", os.path.join(HERE, "icons")),
                "Menu": ("o", "/MenuBar"),
                "ItemIsMenu": ("b", False)}
        t, v = vals.get(prop, ("s", ""))
        return GLib.Variant(t, v)

    def _sni_call(self, _c, _s, _p, _i, method, _params, inv):
        if method in ("Activate", "SecondaryActivate"):
            self.app._tray_toggle()
        inv.return_value(None)

    # id 4 = live status (disabled/informational), then Show/Hide, Stop, Quit
    def _items(self):
        return [(4, self.status_text, False), (5, None, False),
                (1, "Show / hide window", True), (6, "Stop current render", True),
                (2, None, False), (3, "Quit Frequency", True)]

    def _item_props(self, iid):
        for i, label, enabled in self._items():
            if i == iid:
                if label is None:
                    return {"type": GLib.Variant("s", "separator")}
                return {"label": GLib.Variant("s", label),
                        "enabled": GLib.Variant("b", enabled)}
        return {}

    def update_status(self, text):
        """Refresh the tray's status line + tooltip; called from the GUI thread."""
        new = text or "Idle"
        if new == self.status_text:
            return
        self.status_text = new
        if not self.ok:
            return
        self._rev += 1
        try:
            self._bus.emit_signal(None, "/MenuBar", "com.canonical.dbusmenu",
                                  "LayoutUpdated", GLib.Variant("(ui)", (self._rev, 0)))
            self._bus.emit_signal(None, "/StatusNotifierItem",
                                  "org.kde.StatusNotifierItem", "NewToolTip", None)
        except Exception:
            pass

    def _menu_call(self, _c, _s, _p, _i, method, params, inv):
        if method == "GetLayout":
            children = [GLib.Variant("(ia{sv}av)", (i, self._item_props(i), []))
                        for i, _l, _e in self._items()]
            root = (0, {"children-display": GLib.Variant("s", "submenu")}, children)
            inv.return_value(GLib.Variant("(u(ia{sv}av))", (self._rev, root)))
        elif method == "GetGroupProperties":
            ids = params.unpack()[0]
            out = [(i, self._item_props(i)) for i in ids]
            inv.return_value(GLib.Variant("(a(ia{sv}))", (out,)))
        elif method == "Event":
            iid, event, _d, _t = params.unpack()
            if event == "clicked":
                if iid == 1:
                    self.app._tray_toggle()
                elif iid == 6:
                    self.app._tray_stop()
                elif iid == 3:
                    self.app._tray_quit()
            inv.return_value(None)
        elif method == "AboutToShow":
            inv.return_value(GLib.Variant("(b)", (False,)))
        else:
            inv.return_value(None)

    def _menu_get(self, _c, _s, _p, _i, prop):
        if prop == "Version":
            return GLib.Variant("u", 3)
        return GLib.Variant("s", "normal")


class Frequency(Adw.Application):
    def __init__(self):
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.proc = None
        self.ready = False
        self.gen_id = 0
        self.cur_steps = 8
        self.last_wav = None
        self.last_duration = 30.0
        self.last_prompt = ""
        self.pending = None  # a generate request queued while the model loads
        self.init_audio_path = None
        self.inpaint_audio_path = None
        self.coda_clip_path = None
        # waveform inpaint region selector state
        self.wave_peaks = []          # list of 0..1 peak magnitudes
        self.wave_dur = 0.0           # seconds represented by the waveform
        self.inpaint_regions = []     # list of [start_sec, end_sec]
        self._drag = None             # (x0, x1) during a drag, in widget px
        self._action_buttons = []     # buttons disabled while the worker is busy
        # batch / automated mode
        self.batch_queue = []         # list of {"index","prompt","duration"}
        self.batch_total = 0
        self.batch_done = 0
        self.batch_dir = None
        self.batch_running = False
        self.batch_current = None
        self.longform_out = None
        self.longform_running = False
        self.lf_ref_path = None
        self.loop_out = None
        self._partial_loaded = False
        self._seg_t0 = None
        self._ntokens_pending = False
        self.video_image = None
        self.video_audio = None
        self.video_out = None
        self._ffmpeg = None
        self._video_encoding = False
        self.settings = config.load()
        self.css_provider = None

    # ---------- lifecycle ----------
    def do_activate(self):
        if self.props.active_window:
            self.props.active_window.present()
            return
        self._install_css_provider()
        Gtk.Window.set_default_icon_name(APP_ID)
        self.win = Adw.ApplicationWindow(application=self, title=APP_NAME)
        self.win.set_default_size(900, 760)
        self._build_ui()
        self._apply_theme()
        self._tray = _Tray(self)
        self.win.connect("close-request", self._on_close_request)
        self.win.present()
        self._start_worker()

    def _on_close_request(self, _win):
        # cascade-style: present the choices rather than silently doing one thing
        dlg = Adw.AlertDialog(heading="Close Frequency?",
                              body="A render in progress keeps going if you minimize "
                                   "to the tray.")
        dlg.add_response("cancel", "Cancel")
        if getattr(self._tray, "ok", False):
            dlg.add_response("tray", "Minimize to tray")
        dlg.add_response("quit", "Quit")
        dlg.set_response_appearance("quit", Adw.ResponseAppearance.DESTRUCTIVE)
        dlg.set_default_response("tray" if getattr(self._tray, "ok", False) else "quit")
        dlg.set_close_response("cancel")

        def resp(_d, r):
            if r == "tray":
                self.win.set_visible(False)
            elif r == "quit":
                self._tray_quit()
        dlg.connect("response", resp)
        dlg.present(self.win)
        return True   # we handle it via the dialog

    def _tray_toggle(self):
        if self.win.get_visible():
            self.win.set_visible(False)
        else:
            self.win.present()

    def _tray_stop(self):
        if self._busy or self.batch_running:
            self.on_stop(None)

    def _tray_quit(self):
        self.win.destroy()
        self.quit()

    def _tray_status(self, text):
        if getattr(self, "_tray", None):
            self._tray.update_status(text)

    def _install_css_provider(self):
        self.css_provider = Gtk.CssProvider()
        display = Gdk.Display.get_default()
        if display is not None:
            Gtk.StyleContext.add_provider_for_display(
                display, self.css_provider, Gtk.STYLE_PROVIDER_PRIORITY_USER)

    def _apply_theme(self):
        pal = theme.find_theme(self.settings["theme"], config.themes_dir())
        sm = Adw.StyleManager.get_default()
        if pal.follow_system:
            sm.set_color_scheme(Adw.ColorScheme.DEFAULT)
        elif pal.dark:
            sm.set_color_scheme(Adw.ColorScheme.FORCE_DARK)
        else:
            sm.set_color_scheme(Adw.ColorScheme.FORCE_LIGHT)
        css = theme.compile_css(pal, self.settings["accent"],
                                self.settings["glow_intensity"],
                                self.settings["density"], self.settings["font"])
        self.css_provider.load_from_string(css)

    # ---------- UI ----------
    def _build_ui(self):
        self.stack = Adw.ViewStack()
        # don't stretch every tab to the widest tab's width (that clipped the
        # simple tabs to the Advanced tab's fixed-column width when windowed)
        self.stack.set_hhomogeneous(False)
        self.stack.set_vhomogeneous(False)
        self.stack.add_titled_with_icon(self._simple_page(), "simple", "Simple",
                                        "audio-x-generic-symbolic")
        self.stack.add_titled_with_icon(self._advanced_page(), "advanced", "Advanced",
                                        "applications-engineering-symbolic")
        # Continue: needs latent continuation / crossfade-stitching that
        # ACE-Step does not expose. Kept in the source, not shown.
        # self.stack.add_titled_with_icon(self._coda_page(), "continue", "Continue",
                                        # "media-seek-forward-symbolic")
        self.stack.add_titled_with_icon(self._batch_page(), "batch", "Batch",
                                        "view-list-symbolic")
        # Long-form: needs latent continuation / crossfade-stitching that
        # ACE-Step does not expose. Kept in the source, not shown.
        # self.stack.add_titled_with_icon(self._longform_page(), "longform", "Long-form",
                                        # "media-playlist-repeat-symbolic")
        # Loop: needs latent continuation / crossfade-stitching that
        # ACE-Step does not expose. Kept in the source, not shown.
        # self.stack.add_titled_with_icon(self._loop_page(), "loop", "Loop",
                                        # "media-playlist-repeat-song-symbolic")
        self.stack.add_titled_with_icon(self._video_page(), "video", "Video",
                                        "video-x-generic-symbolic")
        self.stack.add_titled_with_icon(self._seeds_page(), "seeds", "Seeds",
                                        "media-playlist-shuffle-symbolic")
        self.stack.add_titled_with_icon(self._history_page(), "history", "History",
                                        "document-open-recent-symbolic")

        header = Adw.HeaderBar()
        switcher = Adw.ViewSwitcher(stack=self.stack, policy=Adw.ViewSwitcherPolicy.WIDE)
        header.set_title_widget(switcher)
        # Responsive tabs: with 9 pages the WIDE switcher squishes labels at
        # smaller window sizes. Breakpoints: mid width -> NARROW (icon-over-
        # label); small width -> plain title + a ViewSwitcherBar strip.
        self._switcher = switcher
        self._switch_bar = Adw.ViewSwitcherBar(stack=self.stack, reveal=False)
        bp_mid = Adw.Breakpoint.new(Adw.BreakpointCondition.parse("max-width: 1280sp"))
        bp_mid.add_setter(switcher, "policy", Adw.ViewSwitcherPolicy.NARROW)
        self.win.add_breakpoint(bp_mid)
        bp_small = Adw.Breakpoint.new(Adw.BreakpointCondition.parse("max-width: 860sp"))
        bp_small.add_setter(header, "title-widget", Adw.WindowTitle(title=APP_NAME))
        bp_small.add_setter(self._switch_bar, "reveal", True)
        self.win.add_breakpoint(bp_small)
        gear = Gtk.Button(icon_name="emblem-system-symbolic")
        gear.set_tooltip_text("Appearance & settings")
        gear.connect("clicked", self.on_open_settings)
        header.pack_end(gear)

        # primary menu — an always-visible Quit that never depends on the tray
        menu = Gio.Menu()
        menu.append("Show / hide window", "app.toggle_window")
        menu.append("Quit Frequency", "app.quit_app")
        menu_btn = Gtk.MenuButton(icon_name="open-menu-symbolic",
                                  menu_model=menu, tooltip_text="Menu")
        header.pack_end(menu_btn)
        act = Gio.SimpleAction.new("quit_app", None)
        act.connect("activate", lambda *_a: self._tray_quit())
        self.add_action(act)
        act2 = Gio.SimpleAction.new("toggle_window", None)
        act2.connect("activate", lambda *_a: self._tray_toggle())
        self.add_action(act2)

        toolbar = Adw.ToolbarView()
        toolbar.add_top_bar(header)
        toolbar.set_content(self.stack)

        # shared status + progress live in a bottom bar
        self.progress = Gtk.ProgressBar(show_text=True, hexpand=True, valign=Gtk.Align.CENTER)
        self.status = Gtk.Label(label="Loading model…", xalign=0)
        self.status.add_css_class("as-dim")
        self.stop_btn = Gtk.Button(icon_name="process-stop-symbolic")
        self.stop_btn.set_tooltip_text("Stop — abort the current run and reload the model")
        self.stop_btn.add_css_class("destructive-action")
        self.stop_btn.set_sensitive(False)
        self.stop_btn.connect("clicked", self.on_stop)
        prog_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        prog_row.append(self.progress)
        prog_row.append(self.stop_btn)
        bottom = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        bottom.set_margin_top(6); bottom.set_margin_bottom(8)
        bottom.set_margin_start(16); bottom.set_margin_end(16)
        bottom.append(prog_row)
        status_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        status_row.append(self.status)
        self.elapsed = Gtk.Label(label="", xalign=1, hexpand=True)
        self.elapsed.add_css_class("as-dim")
        status_row.append(self.elapsed)
        bottom.append(status_row)
        bottom.add_css_class("as-statusbar")
        toolbar.add_bottom_bar(self._switch_bar)
        toolbar.add_bottom_bar(bottom)

        self._job_started = None
        self._busy = False
        GLib.timeout_add_seconds(1, self._tick_elapsed)

        # lock button (visible only when the lock feature is enabled)
        self.lock_btn = Gtk.Button(icon_name="changes-prevent-symbolic")
        self.lock_btn.set_tooltip_text("Lock " + APP_NAME)
        self.lock_btn.set_visible(self._lock_ready())
        self.lock_btn.connect("clicked", lambda _b: self._lock_now())
        header.pack_end(self.lock_btn)

        # outer stack: the app, and a lock page covering it
        self._outer = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE)
        self._outer.add_named(toolbar, "app")
        self._outer.add_named(self._build_lock_page(), "lock")
        self.win.set_content(self._outer)
        self._watch_session_lock()

    # ---------- lock screen ----------
    def _lock_ready(self):
        return bool(self.settings.get("lock_enabled")) and bool(self.settings.get("lock_hash"))

    @staticmethod
    def _pin_hash(pin, salt=None):
        salt = salt or os.urandom(8).hex()
        return f"{salt}${hashlib.sha256((salt + pin).encode()).hexdigest()}"

    def _pin_matches(self, pin):
        stored = self.settings.get("lock_hash", "")
        if "$" not in stored:
            return False
        salt, _h = stored.split("$", 1)
        return self._pin_hash(pin, salt) == stored

    def _build_lock_page(self):
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16,
                       valign=Gtk.Align.CENTER, halign=Gtk.Align.CENTER)
        title = Gtk.Label(label=f"{APP_NAME} is locked")
        title.add_css_class("title-1")
        page.append(title)
        self._lock_entry = Gtk.PasswordEntry(show_peek_icon=True,
                                             placeholder_text="PIN")
        self._lock_entry.set_size_request(240, -1)
        self._lock_entry.connect("activate", lambda _e: self._try_unlock())
        page.append(self._lock_entry)
        self._lock_msg = Gtk.Label(label="")
        self._lock_msg.add_css_class("as-dim")
        page.append(self._lock_msg)
        ub = Gtk.Button(label="Unlock")
        ub.add_css_class("suggested-action"); ub.add_css_class("pill")
        ub.connect("clicked", lambda _b: self._try_unlock())
        page.append(ub)
        return page

    def _lock_now(self):
        if not self._lock_ready():
            return
        self._lock_entry.set_text("")
        self._lock_msg.set_text("")
        self._outer.set_visible_child_name("lock")
        self._lock_entry.grab_focus()

    def _try_unlock(self):
        if self._pin_matches(self._lock_entry.get_text()):
            self._lock_entry.set_text("")
            self._lock_msg.set_text("")
            self._outer.set_visible_child_name("app")
        else:
            self._lock_msg.set_text("Wrong PIN")
            self._lock_entry.set_text("")

    def _watch_session_lock(self):
        """Auto-lock the app when the SYSTEM session locks (tesseract-style)."""
        def on_saver(_conn, _sender, _path, _iface, _sig, params):
            try:
                active = params.unpack()[0]
            except Exception:
                return
            if active and self._lock_ready():
                self._lock_now()
        try:
            bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            for iface in ("org.gnome.ScreenSaver", "org.freedesktop.ScreenSaver"):
                bus.signal_subscribe(None, iface, "ActiveChanged", None, None,
                                     Gio.DBusSignalFlags.NONE, on_saver)
        except Exception:
            pass

    def _tick_elapsed(self):
        if self._busy and self._job_started:
            s = int(time.time() - self._job_started)
            el = (f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
                  if s >= 3600 else f"{s // 60}:{s % 60:02d}")
            self.elapsed.set_text("⏱ " + el)
            prog = self.progress.get_text() or "Working…"
            self._tray_status(f"{prog}  ·  {el}")
        else:
            self.elapsed.set_text("")
            if not getattr(self, "ready", False):
                self._tray_status("Loading model…")
            else:
                self._tray_status("Idle")
        return True

    def _scroller(self, child):
        # AUTOMATIC (not NEVER) so an over-wide tab scrolls instead of clipping
        sw = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.AUTOMATIC, vexpand=True)
        sw.set_child(child)
        return sw

    def _info_btn(self, text):
        """A small (i) button that opens a one-line explanation popover."""
        btn = Gtk.MenuButton(icon_name="help-about-symbolic", valign=Gtk.Align.CENTER)
        btn.add_css_class("flat"); btn.add_css_class("circular")
        lbl = Gtk.Label(label=text, wrap=True, xalign=0, max_width_chars=38)
        lbl.set_margin_top(10); lbl.set_margin_bottom(10)
        lbl.set_margin_start(12); lbl.set_margin_end(12)
        pop = Gtk.Popover(); pop.set_child(lbl)
        btn.set_popover(pop)
        return btn

    def _info(self, row, text):
        """Attach an (i) popover to an Adw row and return the row (chainable)."""
        row.add_suffix(self._info_btn(text))
        return row

    # ----- Simple tab -----
    def _simple_page(self):
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        outer.set_margin_top(18); outer.set_margin_bottom(18)
        outer.set_margin_start(18); outer.set_margin_end(18)

        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        card.add_css_class("as-card")
        outer.append(card)

        card.append(Gtk.Label(label="Prompt", xalign=0))
        sw, self.s_prompt = _textview()
        card.append(sw)

        card.append(self._duration_box("s_dur", 60))

        # ---- lyrics (ACE-Step engine) ----
        lyr = Adw.PreferencesGroup()
        self.s_instrumental = Adw.SwitchRow(
            title="Instrumental",
            subtitle="No vocals at all")
        self.s_instrumental.connect("notify::active", self._on_lyrics_mode)
        lyr.add(self.s_instrumental)
        self.s_auto_lyrics = Adw.SwitchRow(
            title="Write lyrics for me",
            subtitle="The 5Hz LM writes lyrics from your prompt (adds ~4 min).\n"
                     "Detailed prompts give far better lyrics — several "
                     "comma-separated descriptors, including a vocal one.")
        self.s_auto_lyrics.connect("notify::active", self._on_lyrics_mode)
        lyr.add(self.s_auto_lyrics)
        card.append(lyr)

        self.s_lyrics_label = Gtk.Label(label="Lyrics", xalign=0)
        card.append(self.s_lyrics_label)
        sw_l, self.s_lyrics = _textview(min_height=140)
        self.s_lyrics_scroller = sw_l
        card.append(sw_l)
        self.s_lyrics_hint = Gtk.Label(
            label="Use section tags: [Intro] [Verse 1] [Pre-Chorus] [Chorus] "
                  "[Bridge] [Outro].  Max 4096 characters.",
            xalign=0, wrap=True)
        self.s_lyrics_hint.add_css_class("dim-label")
        self.s_lyrics_hint.add_css_class("caption")
        card.append(self.s_lyrics_hint)

        adv = Adw.PreferencesGroup()
        exp = Adw.ExpanderRow(title="Advanced settings")
        adv.add(exp)
        self.s_steps = _spin(1, 50, 1, 8); self.s_steps.set_title("Steps")
        exp.add_row(self.s_steps)
        self.s_cfg = _spin(0.5, 8.0, 0.1, 1.0, 1); self.s_cfg.set_title("CFG scale")
        exp.add_row(self.s_cfg)
        self.s_sampler = Adw.ComboRow(title="Sampler", model=Gtk.StringList.new(SAMPLERS))
        exp.add_row(self.s_sampler)
        self.s_seed = Adw.EntryRow(title="Seed (blank = random)")
        exp.add_row(self.s_seed)
        self.s_bpm = _spin(0, 300, 1, 0); self.s_bpm.set_title("BPM (0 = model decides)")
        exp.add_row(self.s_bpm)
        self.s_keyscale = Adw.EntryRow(title="Key / scale (e.g. E-flat minor)")
        exp.add_row(self.s_keyscale)
        self.s_timesig = Adw.EntryRow(title="Time signature (e.g. 4)")
        exp.add_row(self.s_timesig)
        self.s_language = Adw.EntryRow(title="Vocal language (e.g. en)")
        self.s_language.set_text("en")
        exp.add_row(self.s_language)
        card.append(adv)

        self.s_gen = Gtk.Button(label="Generate")
        self.s_gen.add_css_class("suggested-action"); self.s_gen.add_css_class("pill")
        self.s_gen.connect("clicked", self.on_generate_simple)
        card.append(self.s_gen)
        self._action_buttons.append(self.s_gen)

        # output player
        out = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        out.add_css_class("as-card")
        out.append(Gtk.Label(label="Output", xalign=0))
        self.s_media = Gtk.MediaControls()
        out.append(self.s_media)
        self.s_save = Gtk.Button(label="Save audio…")
        self.s_save.set_sensitive(False)
        self.s_save.connect("clicked", self.on_save)
        out.append(self.s_save)
        s_trim = Gtk.Button(label="Trim & export…")
        s_trim.add_css_class("flat")
        s_trim.connect("clicked", self.on_trim_export)
        out.append(s_trim)
        outer.append(out)

        return self._scroller(outer)

    # ----- Advanced tab -----
    def _advanced_page(self):
        outer = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=14)
        outer.set_margin_top(18); outer.set_margin_bottom(18)
        outer.set_margin_start(18); outer.set_margin_end(18)

        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12, hexpand=True)
        outer.append(left)

        prompts = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        prompts.add_css_class("as-card")
        prompts.append(Gtk.Label(label="Prompt", xalign=0))
        sw, self.a_prompt = _textview()
        prompts.append(sw)
        self.a_negative = Gtk.Entry(
            placeholder_text="Negative prompt (only takes effect when CFG scale > 1)")
        prompts.append(self.a_negative)
        left.append(prompts)

        left.append(self._duration_box("a_dur", 60, label="Seconds total"))

        sc = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12, homogeneous=True)
        steps_g = Adw.PreferencesGroup()
        self.a_steps = _spin(1, 500, 1, 8); self.a_steps.set_title("Steps")
        self._info(self.a_steps, "Number of diffusion steps. More steps = higher quality "
                   "but slower (~linear). 8 is a good default; 4 for quick drafts.")
        steps_g.add(self.a_steps)
        cfg_g = Adw.PreferencesGroup()
        self.a_cfg = _spin(0.0, 25.0, 0.1, 1.0, 1); self.a_cfg.set_title("CFG scale")
        self._info(self.a_cfg, "Classifier-free guidance: how strongly the audio follows the "
                   "prompt. Higher = more literal but can sound harsh. ~1–7 typical.")
        cfg_g.add(self.a_cfg)
        sc.append(steps_g); sc.append(cfg_g)
        left.append(sc)

        # Sampler params
        sp = Adw.PreferencesGroup()
        sp_exp = Adw.ExpanderRow(title="Sampler params")
        sp.add(sp_exp)
        self.a_seed = Adw.EntryRow(title="Seed (blank = random)")
        self._info(self.a_seed, "Random seed. The same seed + same settings reproduces the "
                   "same audio. Leave blank for a new random result each time.")
        sp_exp.add_row(self.a_seed)
        self.a_sampler = Adw.ComboRow(title="Sampler type", model=Gtk.StringList.new(SAMPLERS))
        self._info(self.a_sampler, "Diffusion sampler algorithm. 'pingpong' is the default; "
                   "euler/rk4/dpmpp are alternatives that can change texture at low step counts.")
        sp_exp.add_row(self.a_sampler)
        self.a_sigma = _spin(0.0, 1.0, 0.01, 1.0, 2); self.a_sigma.set_title("Sigma max")
        self._info(self.a_sigma, "Maximum noise level the sampler starts from (0–1). 1.0 = full "
                   "fresh generation. Lower keeps more of any init audio / structure.")
        sp_exp.add_row(self.a_sigma)
        self.a_apg = _spin(0.0, 1.0, 0.1, 1.0, 1)
        self.a_apg.set_title("APG scale"); self.a_apg.set_subtitle("1.0=full APG, 0.0=vanilla CFG")
        self._info(self.a_apg, "Adaptive Projected Guidance. 1.0 = full APG (cleaner guidance), "
                   "0.0 = plain CFG. Reduces over-saturation artifacts at high CFG.")
        sp_exp.add_row(self.a_apg)
        self.a_padding = _spin(0.0, 30.0, 0.5, 6.0, 1); self.a_padding.set_title("Duration padding (sec)")
        self._info(self.a_padding, "Extra headroom seconds added internally when adapting the "
                   "length, then trimmed. Helps the ending resolve cleanly. 6s is typical.")
        sp_exp.add_row(self.a_padding)
        left.append(sp)

        # Output params
        op = Adw.PreferencesGroup()
        op_exp = Adw.ExpanderRow(title="Output params")
        op.add(op_exp)
        self.a_preview = _spin(0, 100, 1, 0)
        self.a_preview.set_title("Spec preview every N steps"); self.a_preview.set_subtitle("0 = off")
        self._info(self.a_preview, "Render a spectrogram preview every N diffusion steps so you "
                   "can watch it form. 0 disables previews (slightly faster).")
        op_exp.add_row(self.a_preview)
        self.a_cut = Adw.SwitchRow(title="Cut to seconds total", active=True)
        self._info(self.a_cut, "Trim the output to exactly the requested length. Off keeps the "
                   "full internally-padded tail.")
        op_exp.add_row(self.a_cut)
        self.a_lufs = Adw.SwitchRow(title="Normalize to −14 LUFS", active=False)
        self._info(self.a_lufs, "Loudness-normalize the output to −14 LUFS (YouTube's playback "
                   "target) instead of peak normalization. Uploads keep consistent volume.")
        op_exp.add_row(self.a_lufs)
        left.append(op)

        # Init audio
        ia = Adw.PreferencesGroup()
        ia_exp = Adw.ExpanderRow(title="Init audio")
        ia.add(ia_exp)
        self.a_init_row = Adw.ActionRow(title="Init audio", subtitle="none")
        init_btn = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        init_btn.connect("clicked", self._choose_audio, "init")
        init_clr = Gtk.Button(icon_name="edit-clear-symbolic", valign=Gtk.Align.CENTER)
        init_clr.connect("clicked", lambda _b: self._set_audio("init", None))
        self.a_init_row.add_suffix(init_btn); self.a_init_row.add_suffix(init_clr)
        self.a_init_row.add_suffix(self._info_btn(
            "Audio to start generation from (style/structure seed). The model reshapes it "
            "toward your prompt. Use with Init noise level to set how much is kept."))
        ia_exp.add_row(self.a_init_row)
        self.a_init_noise = _spin(0.01, 1.0, 0.01, 0.9, 2); self.a_init_noise.set_title("Init noise level")
        self._info(self.a_init_noise, "How much noise to add to the init audio (0.01–1.0). "
                   "Lower = stay closer to the original; higher = freer reinterpretation.")
        ia_exp.add_row(self.a_init_noise)
        left.append(ia)

        # Inpainting
        ip = Adw.PreferencesGroup()
        ip_exp = Adw.ExpanderRow(title="Inpainting")
        ip.add(ip_exp)
        self.a_inpaint_row = Adw.ActionRow(title="Inpaint audio", subtitle="none")
        ip_btn = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        ip_btn.connect("clicked", self._choose_audio, "inpaint")
        ip_clr = Gtk.Button(icon_name="edit-clear-symbolic", valign=Gtk.Align.CENTER)
        ip_clr.connect("clicked", lambda _b: self._set_audio("inpaint", None))
        self.a_inpaint_row.add_suffix(ip_btn); self.a_inpaint_row.add_suffix(ip_clr)
        self.a_inpaint_row.add_suffix(self._info_btn(
            "Audio to regenerate parts of. Load a WAV, then drag on the waveform below to "
            "highlight the region(s) to repaint. The rest is kept. Multiple regions allowed."))
        ip_exp.add_row(self.a_inpaint_row)

        # Waveform region selector (WAV only). Drag to mark regions; falls back to
        # the numeric Mask start/end below when no regions are drawn / non-WAV.
        wf_row = Adw.ActionRow()
        wf_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, hexpand=True)
        self.wave_area = Gtk.DrawingArea()
        self.wave_area.set_size_request(-1, 96)
        self.wave_area.set_draw_func(self._draw_wave)
        self.wave_area.add_css_class("card")
        drag = Gtk.GestureDrag()
        drag.connect("drag-begin", self._wave_drag_begin)
        drag.connect("drag-update", self._wave_drag_update)
        drag.connect("drag-end", self._wave_drag_end)
        self.wave_area.add_controller(drag)
        wf_box.append(self.wave_area)
        wf_ctl = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.wave_hint = Gtk.Label(label="Load a WAV to mark regions by dragging.", xalign=0,
                                   hexpand=True)
        self.wave_hint.add_css_class("as-dim")
        clr_regions = Gtk.Button(label="Clear regions", valign=Gtk.Align.CENTER)
        clr_regions.connect("clicked", lambda _b: self._clear_regions())
        wf_ctl.append(self.wave_hint); wf_ctl.append(clr_regions)
        wf_box.append(wf_ctl)
        wf_row.set_child(wf_box)
        ip_exp.add_row(wf_row)

        self.a_mask_start = _spin(0.0, MAX_SECONDS, 0.1, 0.0, 1); self.a_mask_start.set_title("Mask start (sec)")
        self._info(self.a_mask_start, "Manual fallback for a single inpaint region's start, in "
                   "seconds. Ignored when you've drawn regions on the waveform above.")
        ip_exp.add_row(self.a_mask_start)
        self.a_mask_end = _spin(0.0, MAX_SECONDS, 0.1, 0.0, 1); self.a_mask_end.set_title("Mask end (sec)")
        self._info(self.a_mask_end, "Manual fallback for a single inpaint region's end, in "
                   "seconds. Must be greater than Mask start. Ignored when regions are drawn.")
        ip_exp.add_row(self.a_mask_end)
        left.append(ip)

        # right column: generate + output + spectrogram + send-to
        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        right.set_size_request(320, -1)
        outer.append(right)

        self.a_gen = Gtk.Button(label="Generate")
        self.a_gen.add_css_class("suggested-action"); self.a_gen.add_css_class("pill")
        self.a_gen.connect("clicked", self.on_generate_advanced)
        right.append(self.a_gen)
        self._action_buttons.append(self.a_gen)

        outc = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        outc.add_css_class("as-card")
        outc.append(Gtk.Label(label="Output audio", xalign=0))
        self.a_media = Gtk.MediaControls()
        outc.append(self.a_media)
        self.a_save = Gtk.Button(label="Save audio…")
        self.a_save.set_sensitive(False)
        self.a_save.connect("clicked", self.on_save)
        outc.append(self.a_save)
        a_trim = Gtk.Button(label="Trim & export…")
        a_trim.add_css_class("flat")
        a_trim.connect("clicked", self.on_trim_export)
        outc.append(a_trim)
        right.append(outc)

        specc = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        specc.add_css_class("as-card")
        specc.append(Gtk.Label(label="Output spectrogram", xalign=0))
        self.a_spec = Gtk.Picture()
        self.a_spec.set_size_request(-1, 240)
        self.a_spec.set_content_fit(Gtk.ContentFit.CONTAIN)
        specc.append(self.a_spec)
        right.append(specc)

        send_init = Gtk.Button(label="Send to init audio")
        send_init.connect("clicked", lambda _b: self._send_output_to("init"))
        right.append(send_init)
        send_inpaint = Gtk.Button(label="Send to inpaint audio")
        send_inpaint.connect("clicked", lambda _b: self._send_output_to("inpaint"))
        right.append(send_inpaint)

        # ----- Remix panel: re-roll the just-generated track -----
        remix = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        remix.add_css_class("as-card")
        rhead = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        rhead.append(Gtk.Label(label="Remix output", xalign=0, hexpand=True))
        rhead.append(self._info_btn(
            "Feed the track you just generated back in as init audio and re-roll it. "
            "Leave the prompt blank to keep the vibe, or type a new direction. Lower "
            "strength stays closer to the original; higher reinterprets more freely."))
        remix.append(rhead)
        self.remix_prompt = Gtk.Entry(placeholder_text="New direction (blank = keep prompt)")
        remix.append(self.remix_prompt)
        rg = Adw.PreferencesGroup()
        self.remix_strength = _spin(0.05, 1.0, 0.05, 0.7, 2)
        self.remix_strength.set_title("Remix strength")
        self.remix_strength.set_subtitle("lower = closer to original")
        rg.add(self.remix_strength)
        self.remix_seed = Adw.EntryRow(title="Seed (blank = random)")
        rg.add(self.remix_seed)
        remix.append(rg)
        self.remix_btn = Gtk.Button(label="Remix")
        self.remix_btn.add_css_class("pill")
        self.remix_btn.set_sensitive(False)
        self.remix_btn.connect("clicked", self.on_remix)
        remix.append(self.remix_btn)
        right.append(remix)

        return self._scroller(outer)

    # ----- Continue (CODA-style) tab -----
    def _coda_page(self):
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        outer.set_margin_top(18); outer.set_margin_bottom(18)
        outer.set_margin_start(18); outer.set_margin_end(18)

        intro = Gtk.Label(xalign=0, wrap=True)
        intro.set_markup("<b>Continue a song.</b>  Load a clip of ANY length; it's analyzed "
                         "for key &amp; tempo, then extended in the same style and spliced on "
                         "with a seamless crossfade. Your original is preserved exactly — for "
                         "long tracks the model listens to the final stretch (context) and the "
                         "rest is copied through untouched.")
        intro.add_css_class("as-dim")
        outer.append(intro)

        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        card.add_css_class("as-card")
        outer.append(card)

        clip_g = Adw.PreferencesGroup()
        self.coda_clip_row = Adw.ActionRow(title="Source clip", subtitle="none")
        c_btn = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        c_btn.connect("clicked", self._choose_audio, "coda")
        c_clr = Gtk.Button(icon_name="edit-clear-symbolic", valign=Gtk.Align.CENTER)
        c_clr.connect("clicked", lambda _b: self._set_audio("coda", None))
        self.coda_clip_row.add_suffix(c_btn); self.coda_clip_row.add_suffix(c_clr)
        clip_g.add(self.coda_clip_row)
        card.append(clip_g)

        an_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.coda_analysis = Gtk.Label(label="Key / tempo: —", xalign=0, hexpand=True)
        self.coda_analyze_btn = Gtk.Button(label="Analyze", valign=Gtk.Align.CENTER)
        self.coda_analyze_btn.connect("clicked", self.on_coda_analyze)
        an_row.append(self.coda_analysis); an_row.append(self.coda_analyze_btn)
        card.append(an_row)
        self._action_buttons.append(self.coda_analyze_btn)

        card.append(self._duration_box("coda_cont", 30, label="Continue by"))

        card.append(Gtk.Label(label="Steering prompt (optional)", xalign=0))
        self.coda_prompt = Gtk.Entry(placeholder_text="e.g. add strings and a slow build")
        card.append(self.coda_prompt)

        opt = Adw.PreferencesGroup()
        self.coda_use_analysis = Adw.SwitchRow(title="Steer with detected key/tempo", active=True)
        self._info(self.coda_use_analysis, "Append the detected key and BPM to the prompt so "
                   "the continuation stays in key and in time.")
        opt.add(self.coda_use_analysis)
        self.coda_steps = _spin(1, 500, 1, 8); self.coda_steps.set_title("Steps")
        self._info(self.coda_steps, "Diffusion steps for the continuation. 8 is a good default.")
        opt.add(self.coda_steps)
        self.coda_cfg = _spin(0.0, 25.0, 0.1, 1.0, 1); self.coda_cfg.set_title("CFG scale")
        opt.add(self.coda_cfg)
        self.coda_best = _spin(1, 5, 1, 1); self.coda_best.set_title("Best of N candidates")
        self._info(self.coda_best, "Generate N continuations and auto-keep the cleanest "
                   "(fewest artifacts). N× slower. 1 = single take.")
        opt.add(self.coda_best)
        self.coda_context = _spin(30, 300, 10, 120)
        self.coda_context.set_title("Context (sec)")
        self._info(self.coda_context, "For long clips: how much of the END the model hears "
                   "while composing the continuation. More context = better musical flow "
                   "but slower. The rest of the track is copied through unchanged.")
        opt.add(self.coda_context)
        self.coda_seed = Adw.EntryRow(title="Seed (blank = random)")
        opt.add(self.coda_seed)
        card.append(opt)

        self.coda_btn = Gtk.Button(label="▶ Continue")
        self.coda_btn.add_css_class("suggested-action"); self.coda_btn.add_css_class("pill")
        self.coda_btn.connect("clicked", self.on_coda_continue)
        card.append(self.coda_btn)
        self._action_buttons.append(self.coda_btn)

        out = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        out.add_css_class("as-card")
        out.append(Gtk.Label(label="Completed track", xalign=0))
        self.coda_media = Gtk.MediaControls()
        out.append(self.coda_media)
        self.coda_spec = Gtk.Picture()
        self.coda_spec.set_size_request(-1, 200)
        self.coda_spec.set_content_fit(Gtk.ContentFit.CONTAIN)
        out.append(self.coda_spec)
        self.coda_save = Gtk.Button(label="Save audio…")
        self.coda_save.set_sensitive(False)
        self.coda_save.connect("clicked", self.on_save)
        out.append(self.coda_save)
        outer.append(out)

        return self._scroller(outer)

    # ----- Batch (automated) tab -----
    def _batch_page(self):
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        outer.set_margin_top(18); outer.set_margin_bottom(18)
        outer.set_margin_start(18); outer.set_margin_end(18)

        intro = Gtk.Label(xalign=0, wrap=True)
        intro.set_markup("<b>Batch mode.</b>  One prompt per line — it generates them one after "
                         "another and saves each into a folder you pick. Optional per-line length: "
                         "<tt>dreamy synthwave :: 45</tt>. Leave it running in the background.")
        intro.add_css_class("as-dim")
        outer.append(intro)

        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        card.add_css_class("as-card")
        outer.append(card)

        bpg = Adw.PreferencesGroup()
        bpg.add(self._make_preset_row(self._apply_batch_preset))
        card.append(bpg)

        card.append(Gtk.Label(label="Prompts (one per line)", xalign=0))
        card.append(self._prompt_editor("batch_prompts", min_height=170))

        dir_g = Adw.PreferencesGroup()
        self.batch_dir_row = Adw.ActionRow(title="Output folder", subtitle="none")
        d_btn = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        d_btn.connect("clicked", self.on_batch_choose_dir)
        self.batch_dir_row.add_suffix(d_btn)
        dir_g.add(self.batch_dir_row)
        card.append(dir_g)

        settings = Adw.PreferencesGroup(title="Shared settings")
        self.batch_dur = _spin(1, MAX_SECONDS, 1, 30)
        self.batch_dur.set_title("Default length (sec)")
        self.batch_dur.set_subtitle("used when a line has no :: length")
        settings.add(self.batch_dur)
        self.batch_steps = _spin(1, 500, 1, 8); self.batch_steps.set_title("Steps")
        self._info(self.batch_steps, "Diffusion steps per track. 8 is a good default; lower is faster.")
        settings.add(self.batch_steps)
        self.batch_cfg = _spin(0.0, 25.0, 0.1, 1.0, 1); self.batch_cfg.set_title("CFG scale")
        settings.add(self.batch_cfg)
        self.batch_sampler = Adw.ComboRow(title="Sampler", model=Gtk.StringList.new(SAMPLERS))
        settings.add(self.batch_sampler)
        self.batch_instrumental = Adw.SwitchRow(
            title="Instrumental",
            subtitle="No vocals on any track in the batch")
        settings.add(self.batch_instrumental)
        self.batch_auto_lyrics = Adw.SwitchRow(
            title="Write lyrics for every prompt",
            subtitle="The 5Hz LM writes fresh lyrics per track (adds ~4 min each).\n"
                     "Detailed prompts give far better lyrics.")
        settings.add(self.batch_auto_lyrics)
        self.batch_spec = Adw.SwitchRow(title="Also save spectrogram PNG", active=False)
        settings.add(self.batch_spec)
        card.append(settings)

        self.batch_btn = Gtk.Button(label="▶ Start batch")
        self.batch_btn.add_css_class("suggested-action"); self.batch_btn.add_css_class("pill")
        self.batch_btn.connect("clicked", self.on_batch_start)
        card.append(self.batch_btn)
        self._action_buttons.append(self.batch_btn)

        log_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        log_card.add_css_class("as-card")
        log_card.append(Gtk.Label(label="Progress", xalign=0))
        self.batch_log = Gtk.TextView(editable=False, cursor_visible=False, monospace=True,
                                      wrap_mode=Gtk.WrapMode.WORD_CHAR, top_margin=8,
                                      bottom_margin=8, left_margin=8, right_margin=8)
        logsw = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER)
        logsw.set_min_content_height(140); logsw.set_child(self.batch_log)
        logsw.add_css_class("card")
        log_card.append(logsw)
        self.batch_open_btn = Gtk.Button(label="Open output folder")
        self.batch_open_btn.set_sensitive(False)
        self.batch_open_btn.connect("clicked", self._open_batch_dir)
        log_card.append(self.batch_open_btn)
        outer.append(log_card)

        return self._scroller(outer)

    # ----- prompt editor + library (shared by Long-form and Batch) -----
    def _prompts_dir(self):
        d = os.path.join(config.config_dir(), "prompts")
        os.makedirs(d, exist_ok=True)
        return d

    def _prompt_editor(self, attr, min_height=150):
        """Multi-prompt editor: pasted text is NEVER altered — one line = one
        prompt. Adds a live counter (+ exact token check via the worker), an
        Expand dialog for big collections, and a save/load prompt library."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        sw, tv = _textview(min_height)
        setattr(self, attr, tv)
        box.append(sw)
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        counter = Gtk.Label(label="0 prompts", xalign=0, hexpand=True, wrap=True)
        counter.add_css_class("as-dim")
        row.append(counter)
        maxi = Gtk.Button(label="Expand")
        maxi.add_css_class("flat")
        maxi.connect("clicked", lambda _b: self._open_prompt_dialog(tv))
        row.append(maxi)
        forge = Gtk.Button(label="Forge")
        forge.add_css_class("flat")
        forge.set_tooltip_text("Assemble fresh prompt lines from your phrase pools")
        forge.connect("clicked", lambda _b: self._open_forge(tv))
        row.append(forge)
        lib = Gtk.MenuButton(label="Library")
        lib.add_css_class("flat")
        lib.set_popover(self._prompt_lib_popover(tv))
        row.append(lib)
        box.append(row)
        tv.get_buffer().connect(
            "changed", lambda _b: self._update_prompt_counter(tv, counter))
        self._update_prompt_counter(tv, counter)
        return box

    def _update_prompt_counter(self, tv, counter):
        lines = [l for l in _tv_text(tv).splitlines() if l.strip()]
        if not lines:
            counter.set_text("0 prompts")
            return
        words = max(len(l.split()) for l in lines)
        counter.set_text(f"{len(lines)} prompts · longest {words} words")
        if self.ready and not self._busy:
            if getattr(self, "_ntok_src", None):
                GLib.source_remove(self._ntok_src)

            def fire():
                self._ntok_src = None
                if self.ready and not self._busy:
                    self._ntok_ctx = counter
                    self.gen_id += 1
                    self._send({"cmd": "ntokens", "id": self.gen_id, "lines": lines})
                return False
            self._ntok_src = GLib.timeout_add(800, fire)

    def _on_ntokens(self, ev):
        counter = getattr(self, "_ntok_ctx", None)
        counts = ev.get("counts") or []
        if not counter or not counts:
            return
        limit = ev.get("limit", 256)
        mx = max(counts)
        over = sum(1 for c in counts if c > limit)
        base = counter.get_text().split(" · tokens")[0]
        txt = f"{base} · tokens: max {mx}/{limit}"
        if over:
            txt += f" — ⚠ {over} line(s) over the limit get truncated"
        counter.set_text(txt)

    def _open_prompt_dialog(self, tv):
        dlg = Adw.Dialog(title="Edit prompts", content_width=920, content_height=720)
        tbv = Adw.ToolbarView()
        hb = Adw.HeaderBar()
        wrap = Gtk.ToggleButton(label="Wrap lines")
        wrap.set_active(True)
        hb.pack_start(wrap)
        hint = Gtk.Label(label="one line = one prompt")
        hint.add_css_class("as-dim")
        hb.pack_start(hint)
        tbv.add_top_bar(hb)
        big = Gtk.TextView(buffer=tv.get_buffer(), wrap_mode=Gtk.WrapMode.WORD_CHAR,
                           monospace=True, top_margin=10, bottom_margin=10,
                           left_margin=10, right_margin=10)
        wrap.connect("toggled", lambda b: big.set_wrap_mode(
            Gtk.WrapMode.WORD_CHAR if b.get_active() else Gtk.WrapMode.NONE))
        sw = Gtk.ScrolledWindow(vexpand=True, hexpand=True)
        sw.set_child(big)
        tbv.set_content(sw)
        dlg.set_child(tbv)
        dlg.present(self.win)

    def _prompt_lib_popover(self, tv):
        pop = Gtk.Popover()
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        for m in ("top", "bottom", "start", "end"):
            getattr(box, f"set_margin_{m}")(10)
        box.append(Gtk.Label(label="Prompt library", xalign=0))
        lst = Gtk.ListBox()
        lst.add_css_class("boxed-list")
        sc = Gtk.ScrolledWindow(min_content_height=220, min_content_width=380,
                                max_content_height=380, propagate_natural_height=True)
        sc.set_child(lst)
        box.append(sc)
        save_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        name_e = Gtk.Entry(placeholder_text="Save current as…", hexpand=True)
        save_b = Gtk.Button(label="Save")
        save_row.append(name_e); save_row.append(save_b)
        box.append(save_row)

        def refresh(*_a):
            while (c := lst.get_first_child()) is not None:
                lst.remove(c)
            for fn in sorted(os.listdir(self._prompts_dir())):
                if not fn.endswith(".txt"):
                    continue
                path = os.path.join(self._prompts_dir(), fn)
                row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
                for m in ("top", "bottom", "start", "end"):
                    getattr(row, f"set_margin_{m}")(6)
                row.append(Gtk.Label(label=fn[:-4], xalign=0, hexpand=True,
                                     ellipsize=3))  # 3 = END

                def mk_apply(pth, mode):
                    def cb(_b):
                        try:
                            txt = open(pth).read().strip()
                        except OSError:
                            return
                        b = tv.get_buffer()
                        if mode == "load":
                            b.set_text(txt, -1)
                        else:
                            cur = _tv_text(tv)
                            b.set_text((cur + "\n" + txt).strip(), -1)
                        pop.popdown()
                    return cb

                def mk_del(pth):
                    def cb(_b):
                        try:
                            os.remove(pth)
                        except OSError:
                            pass
                        refresh()
                    return cb
                for label, mode in (("Load", "load"), ("Append", "append")):
                    b = Gtk.Button(label=label)
                    b.connect("clicked", mk_apply(path, mode))
                    row.append(b)
                d = Gtk.Button(icon_name="user-trash-symbolic")
                d.add_css_class("flat")
                d.connect("clicked", mk_del(path))
                row.append(d)
                lr = Gtk.ListBoxRow()
                lr.set_child(row)
                lst.append(lr)

        def do_save(_b):
            nm = re.sub(r"[^\w\s-]", "", name_e.get_text()).strip()
            txt = _tv_text(tv)
            if not nm or not txt:
                return
            with open(os.path.join(self._prompts_dir(), nm + ".txt"), "w") as fh:
                fh.write(txt + "\n")
            name_e.set_text("")
            refresh()
        save_b.connect("clicked", do_save)
        name_e.connect("activate", do_save)
        pop.connect("show", refresh)
        pop.set_child(box)
        return pop

    # ----- prompt forge (combinatorial prompt assembly from user pools) -----
    FORGE_DEFAULTS = {
        "genre / mood base": [
            "dreamy chillwave focus", "deep work synthwave", "ambient dreamwave for deep concentration",
            "focused outrun trance chillwave", "soft uplifting synthwave study soundscape",
            "ethereal space-age retrowave", "productive chillwave synthwave hybrid",
            "warm lofi hip hop beat", "minimal ambient focus music"],
        "vocals": [
            "hypnotic vocal melodies", "ethereal non-verbal vocal hooks", "soaring lyricless vocalizing",
            "melodic vocal chops", "ethereal vocal humming hooks", "echoing non-verbal vocal stabs",
            "wordless choral harmonies", "no vocals"],
        "pads / synths": [
            "spacious retro pads", "driving analog bassline", "glowing synth leads",
            "hypnotic repeating synth arpeggios", "warm analog pads", "steady driving bassline",
            "spacious focus pads", "hypnotic analog arps"],
        "build / drop event": [
            "steady flow-state pulse leading to a massive sub-bass drop",
            "explosive build-up into a heavy drop", "cinematic riser to a colossal bass drop",
            "massive sidechained bass drop", "dramatic transition with a deep sub-bass drop",
            "slow evolving build with no drop", "massive build-up to an earth-shaking bass drop"],
        "drums": [
            "punchy 80s drums", "banging gated snares", "driving retro-futuristic beat",
            "crisp banging drums", "driving gated drums", "heavy retro percussion",
            "relaxed boom-bap drums", "no percussion"],
        "effects": [
            "lush echo effects", "sweeping phaser effects", "spacious reverb",
            "sweeping filter effects", "rich delay", "lush stereo panning effects",
            "cosmic delay", "sci-fi sweeps"],
        "purpose / vibe": [
            "immersive productive concentration", "hypnotic focus", "hypnotic flow vocal state",
            "stellar night mood", "deep work momentum", "high-focus euphoria",
            "retrowave", "calm and spacious"],
    }

    def _forge_pools_path(self):
        d = config.config_dir()
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "forge_pools.json")

    def _forge_load_pools(self):
        try:
            with open(self._forge_pools_path()) as fh:
                pools = json.load(fh)
                if isinstance(pools, dict) and pools:
                    return pools
        except Exception:
            pass
        return {k: list(v) for k, v in self.FORGE_DEFAULTS.items()}

    def _open_forge(self, tv):
        pools = self._forge_load_pools()
        dlg = Adw.Dialog(title="Prompt forge", content_width=980, content_height=740)
        tbv = Adw.ToolbarView()
        hb = Adw.HeaderBar()
        tbv.add_top_bar(hb)
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        for m in ("top", "bottom", "start", "end"):
            getattr(root, f"set_margin_{m}")(14)

        hint = Gtk.Label(xalign=0, wrap=True)
        hint.set_markup("Each pool: one phrase per line. Forge assembles lines as "
                        "<i>base, vocals, pads, event, drums, effects, vibe</i> — random picks, "
                        "no immediate repeats between consecutive lines. Your pools are saved.")
        hint.add_css_class("as-dim")
        root.append(hint)

        grid = Gtk.FlowBox(selection_mode=Gtk.SelectionMode.NONE,
                           max_children_per_line=4, column_spacing=8, row_spacing=8)
        editors = {}
        for name in self.FORGE_DEFAULTS:
            col = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
            col.append(Gtk.Label(label=name, xalign=0))
            psw, ptv = _textview(min_height=160)
            ptv.get_buffer().set_text("\n".join(pools.get(name, [])), -1)
            editors[name] = ptv
            col.append(psw)
            grid.append(col)
        gsc = Gtk.ScrolledWindow(vexpand=True)
        gsc.set_child(grid)
        root.append(gsc)

        ctl = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        count = Adw.SpinRow.new_with_range(5, 500, 5)
        count.set_value(50)
        count.set_title("Lines to forge")
        cg = Adw.PreferencesGroup(); cg.add(count); cg.set_hexpand(True)
        ctl.append(cg)
        mode = Gtk.ComboBoxText()
        for m in ("Replace prompts", "Append to prompts"):
            mode.append_text(m)
        mode.set_active(1)
        mode.set_valign(Gtk.Align.CENTER)
        ctl.append(mode)
        forge_b = Gtk.Button(label="Forge")
        forge_b.add_css_class("suggested-action")
        forge_b.set_valign(Gtk.Align.CENTER)
        ctl.append(forge_b)
        root.append(ctl)

        def do_forge(_b):
            import random
            pool_lists = {}
            for name, ptv in editors.items():
                items = [l.strip() for l in _tv_text(ptv).splitlines() if l.strip()]
                if items:
                    pool_lists[name] = items
            try:
                with open(self._forge_pools_path(), "w") as fh:
                    json.dump(pool_lists, fh, indent=1)
            except OSError:
                pass
            order = [n for n in self.FORGE_DEFAULTS if n in pool_lists]
            if not order:
                return
            lines, prev = [], {}
            for _ in range(int(count.get_value())):
                parts = []
                for name in order:
                    opts = pool_lists[name]
                    cand = [o for o in opts if o != prev.get(name)] or opts
                    pick = random.choice(cand)
                    prev[name] = pick
                    parts.append(pick)
                lines.append(", ".join(parts))
            buf = tv.get_buffer()
            if mode.get_active() == 0:
                buf.set_text("\n".join(lines), -1)
            else:
                cur = _tv_text(tv)
                buf.set_text((cur + "\n" + "\n".join(lines)).strip(), -1)
            dlg.close()
            self.status.set_text(f"Forged {len(lines)} prompt lines.")
        forge_b.connect("clicked", do_forge)

        tbv.set_content(root)
        dlg.set_child(tbv)
        dlg.present(self.win)

    # ----- preset helpers -----
    def _make_preset_row(self, on_apply):
        names = Gtk.StringList.new(["— choose a preset —"] + [p["name"] for p in PRESETS])
        row = Adw.ComboRow(title="Preset", model=names)
        row.set_subtitle("Fills in tuned prompts + settings; edit freely after")

        def changed(r, _p):
            i = r.get_selected()
            if i > 0:
                on_apply(PRESETS[i - 1])
        row.connect("notify::selected", changed)
        return row

    @staticmethod
    def _select_sampler(combo, name):
        if name in SAMPLERS:
            combo.set_selected(SAMPLERS.index(name))

    def _apply_batch_preset(self, p):
        self.batch_prompts.get_buffer().set_text("\n".join(p["prompts"]), -1)
        self.batch_dur.set_value(p["segment"])
        self.batch_steps.set_value(p["steps"])
        self.batch_cfg.set_value(p["cfg"])
        self._select_sampler(self.batch_sampler, p["sampler"])

    def _apply_longform_preset(self, p):
        self.lf_prompts.get_buffer().set_text("\n".join(p["prompts"]), -1)
        self.lf_segment.set_value(p["segment"])
        self.lf_steps.set_value(p["steps"])
        self.lf_cfg.set_value(p["cfg"])
        self._select_sampler(self.lf_sampler, p["sampler"])
        self._update_lf_estimate()

    # ----- Long-form tab -----
    def _longform_page(self):
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        outer.set_margin_top(18); outer.set_margin_bottom(18)
        outer.set_margin_start(18); outer.set_margin_end(18)

        intro = Gtk.Label(xalign=0, wrap=True)
        intro.set_markup("<b>Long-form render.</b>  Pick a preset (or write your own theme), set a "
                         "target length, and it generates back-to-back segments and crossfades them "
                         "into one continuous track — ideal for 1–3 hour focus / sleep videos. "
                         "Streamed to disk, so length is limited by disk, not RAM.")
        intro.add_css_class("as-dim")
        outer.append(intro)

        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        card.add_css_class("as-card")
        outer.append(card)

        pg = Adw.PreferencesGroup()
        pg.add(self._make_preset_row(self._apply_longform_preset))
        card.append(pg)

        card.append(Gtk.Label(label="Theme prompts (one per line — rotated for variety)", xalign=0))
        card.append(self._prompt_editor("lf_prompts", min_height=170))

        # optional reference track: segments inherit its character (init audio)
        ref_g = Adw.PreferencesGroup()
        self.lf_ref_row = Adw.ActionRow(title="Reference track (optional)", subtitle="none")
        rb = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        rb.connect("clicked", self._choose_audio, "lf_ref")
        rc = Gtk.Button(icon_name="edit-clear-symbolic", valign=Gtk.Align.CENTER)
        rc.connect("clicked", lambda _b: self._set_audio("lf_ref", None))
        self.lf_ref_row.add_suffix(rb); self.lf_ref_row.add_suffix(rc)
        self.lf_ref_row.add_suffix(self._info_btn(
            "Optional: each segment starts from a random slice of this track, so the whole "
            "render builds off its vibe (prompts still rotate on top). Leave empty for the "
            "classic behavior — output is completely unchanged when unset."))
        ref_g.add(self.lf_ref_row)
        self.lf_ref_strength = _spin(0.5, 0.95, 0.05, 0.8, 2)
        self.lf_ref_strength.set_title("Reference freedom")
        self.lf_ref_strength.set_subtitle("lower = closer to the reference sound")
        ref_g.add(self.lf_ref_strength)
        card.append(ref_g)

        settings = Adw.PreferencesGroup(title="Render settings")
        self.lf_minutes = _spin(1, 600, 5, 60); self.lf_minutes.set_title("Target length (minutes)")
        self._info(self.lf_minutes, "Total length of the finished track. 60–180 min is typical for "
                   "focus/sleep videos. Render time grows with this.")
        self.lf_minutes.connect("notify::value", lambda *_: self._update_lf_estimate())
        settings.add(self.lf_minutes)
        self.lf_segment = _spin(30, MAX_SECONDS, 10, 300); self.lf_segment.set_title("Segment length (sec)")
        self._info(self.lf_segment, "Length of each generated chunk before crossfading. Longer = "
                   "fewer seams but slower per chunk. 240–340s is a good range.")
        self.lf_segment.connect("notify::value", lambda *_: self._update_lf_estimate())
        settings.add(self.lf_segment)
        self.lf_xfade = _spin(0, 30, 1, 6); self.lf_xfade.set_title("Crossfade (sec)")
        self._info(self.lf_xfade, "Equal-power crossfade between segments for a seamless blend. "
                   "4–8s hides the seam well.")
        self.lf_xfade.connect("notify::value", lambda *_: self._update_lf_estimate())
        settings.add(self.lf_xfade)
        self.lf_steps = _spin(1, 500, 1, 8); self.lf_steps.set_title("Steps")
        self._info(self.lf_steps, "Diffusion steps per segment. 6–8 is a good quality/speed balance "
                   "for ambient; drop to 4 for faster drafts.")
        settings.add(self.lf_steps)
        self.lf_cfg = _spin(0.0, 25.0, 0.1, 1.0, 1); self.lf_cfg.set_title("CFG scale")
        settings.add(self.lf_cfg)
        self.lf_sampler = Adw.ComboRow(title="Sampler", model=Gtk.StringList.new(SAMPLERS))
        settings.add(self.lf_sampler)
        self.lf_seed = Adw.EntryRow(title="Seed (blank = random)")
        settings.add(self.lf_seed)
        card.append(settings)

        out_g = Adw.PreferencesGroup()
        self.lf_out_row = Adw.ActionRow(title="Output file", subtitle="none")
        o_btn = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        o_btn.connect("clicked", self.on_longform_choose_out)
        o_open = Gtk.Button(icon_name="folder-open-symbolic", valign=Gtk.Align.CENTER)
        o_open.set_tooltip_text("Open the output file's folder")
        o_open.connect("clicked", self._open_longform_dir)
        self.lf_out_row.add_suffix(o_btn)
        self.lf_out_row.add_suffix(o_open)
        out_g.add(self.lf_out_row)
        card.append(out_g)

        self.lf_estimate = Gtk.Label(xalign=0)
        self.lf_estimate.add_css_class("as-dim")
        card.append(self.lf_estimate)

        self.lf_btn = Gtk.Button(label="▶ Render long-form")
        self.lf_btn.add_css_class("suggested-action"); self.lf_btn.add_css_class("pill")
        self.lf_btn.connect("clicked", self.on_longform_start)
        card.append(self.lf_btn)
        self._action_buttons.append(self.lf_btn)

        norm_btn = Gtk.Button(label="Normalize a file to −14 LUFS (YouTube)…")
        norm_btn.add_css_class("flat")
        norm_btn.set_tooltip_text("Loudness-normalize any WAV/FLAC — including multi-hour "
                                  "long-form renders (streamed, low memory). Writes a new file.")
        norm_btn.connect("clicked", self._choose_audio, "lufs_file")
        card.append(norm_btn)
        self._action_buttons.append(norm_btn)

        play = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        play.add_css_class("as-card")
        play.append(Gtk.Label(label="Preview", xalign=0))
        self.lf_media = Gtk.MediaControls()
        play.append(self.lf_media)
        outer.append(play)

        self._update_lf_estimate()
        return self._scroller(outer)

    def _longform_segments(self):
        seg = int(self.lf_segment.get_value())
        xf = min(int(self.lf_xfade.get_value()), seg // 2)
        tgt = int(self.lf_minutes.get_value()) * 60
        n, produced = 1, seg
        while produced < tgt:
            produced += (seg - xf); n += 1
        return n

    def _update_lf_estimate(self):
        n = self._longform_segments()
        mins = int(self.lf_minutes.get_value())
        self.lf_estimate.set_text(
            f"≈ {n} segments to fill {mins} min. CPU render is slow — lower steps or run it "
            f"in the background. Stop is available any time (the partial file is kept).")

    # ----- Loop tab (seamless loop maker) -----
    def _loop_page(self):
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        outer.set_margin_top(18); outer.set_margin_bottom(18)
        outer.set_margin_start(18); outer.set_margin_end(18)

        intro = Gtk.Label(xalign=0, wrap=True)
        intro.set_markup("<b>Seamless loop maker.</b>  Generates a clip and repairs its "
                         "end→start seam so it loops perfectly, then (optionally) tiles it "
                         "into an hours-long track — a 3-hour video from minutes of compute. "
                         "Great for focus/sleep uploads where a consistent vibe is fine.")
        intro.add_css_class("as-dim")
        outer.append(intro)

        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        card.add_css_class("as-card")
        outer.append(card)

        card.append(Gtk.Label(label="Prompt", xalign=0))
        sw, self.loop_prompt = _textview()
        card.append(sw)

        sset = Adw.PreferencesGroup()
        self.loop_dur = _spin(30, MAX_SECONDS, 10, 120)
        self.loop_dur.set_title("Loop length (sec)")
        self._info(self.loop_dur, "Length of the repeating unit. 2–5 min loops feel least "
                   "repetitive on long renders.")
        sset.add(self.loop_dur)
        self.loop_seam = _spin(2, 10, 1, 4)
        self.loop_seam.set_title("Seam repair window (sec)")
        self._info(self.loop_seam, "How much audio around the loop point gets regenerated to "
                   "hide the seam. 4s works well.")
        sset.add(self.loop_seam)
        self.loop_total = _spin(0, 600, 15, 0)
        self.loop_total.set_title("Tile to total length (minutes)")
        self.loop_total.set_subtitle("0 = just the loop itself")
        sset.add(self.loop_total)
        self.loop_steps = _spin(1, 500, 1, 8); self.loop_steps.set_title("Steps")
        sset.add(self.loop_steps)
        self.loop_cfg = _spin(0.0, 25.0, 0.1, 1.0, 1); self.loop_cfg.set_title("CFG scale")
        sset.add(self.loop_cfg)
        self.loop_sampler = Adw.ComboRow(title="Sampler", model=Gtk.StringList.new(SAMPLERS))
        sset.add(self.loop_sampler)
        self.loop_seed = Adw.EntryRow(title="Seed (blank = random)")
        sset.add(self.loop_seed)
        self.loop_lufs = Adw.SwitchRow(title="Normalize to −14 LUFS", active=True)
        sset.add(self.loop_lufs)
        card.append(sset)

        out_g = Adw.PreferencesGroup()
        self.loop_out_row = Adw.ActionRow(title="Output file", subtitle="auto (/tmp)")
        lob = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        lob.connect("clicked", self.on_loop_choose_out)
        self.loop_out_row.add_suffix(lob)
        out_g.add(self.loop_out_row)
        card.append(out_g)

        self.loop_btn = Gtk.Button(label="Render loop")
        self.loop_btn.add_css_class("suggested-action"); self.loop_btn.add_css_class("pill")
        self.loop_btn.connect("clicked", self.on_loop_start)
        card.append(self.loop_btn)
        self._action_buttons.append(self.loop_btn)

        play = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        play.add_css_class("as-card")
        play.append(Gtk.Label(label="Output", xalign=0))
        self.loop_media = Gtk.MediaControls()
        play.append(self.loop_media)
        self.loop_save = Gtk.Button(label="Save audio…")
        self.loop_save.set_sensitive(False)
        self.loop_save.connect("clicked", self.on_save)
        play.append(self.loop_save)
        outer.append(play)

        return self._scroller(outer)

    def on_loop_choose_out(self, _btn):
        dlg = Gtk.FileDialog(title="Save loop as")
        dlg.set_initial_name("loop.wav")
        dlg.save(self.win, None, self._loop_out_done)

    def _loop_out_done(self, dlg, result):
        try:
            gf = dlg.save_finish(result)
            if gf:
                path = gf.get_path()
                if not path.lower().endswith(".wav"):
                    path += ".wav"
                self.loop_out = path
                self.loop_out_row.set_subtitle(os.path.basename(path))
        except GLib.Error:
            pass

    def on_loop_start(self, _btn):
        prompt = _tv_text(self.loop_prompt)
        if not prompt:
            self.status.set_text("Enter a prompt first.")
            return
        sm = self.loop_sampler.get_selected_item()
        self.last_prompt = prompt
        params = {
            "cmd": "loop",
            "prompt": prompt,
            "duration": int(self.loop_dur.get_value()),
            "seam_seconds": float(self.loop_seam.get_value()),
            "total_minutes": int(self.loop_total.get_value()),
            "steps": int(self.loop_steps.get_value()),
            "cfg": float(self.loop_cfg.get_value()),
            "seed": self._seed_of(self.loop_seed.get_text()),
            "sampler": sm.get_string() if sm else "pingpong",
            "spectrogram": True,
        }
        if self.loop_lufs.get_active():
            params["lufs_target"] = -14.0
        if self.loop_out:
            params["out"] = self.loop_out
        self._dispatch(params, steps=int(self.loop_steps.get_value()))

    # ----- Video tab (still image + audio -> MP4) -----
    def _video_page(self):
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        outer.set_margin_top(18); outer.set_margin_bottom(18)
        outer.set_margin_start(18); outer.set_margin_end(18)

        intro = Gtk.Label(xalign=0, wrap=True)
        intro.set_markup("<b>Make a YouTube video.</b>  YouTube needs a video file, not bare audio. "
                         "Drop in one still image + a track and it renders a 1080p MP4 (H.264 + AAC) "
                         "with the image held for the whole song. Uses a bundled encoder — nothing to "
                         "install.")
        intro.add_css_class("as-dim")
        outer.append(intro)

        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        card.add_css_class("as-card")
        outer.append(card)

        g = Adw.PreferencesGroup()
        self.video_img_row = Adw.ActionRow(title="Cover image", subtitle="none")
        ib = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        ib.connect("clicked", self.on_choose_image)
        self.video_img_row.add_suffix(ib)
        self.video_img_row.add_suffix(self._info_btn(
            "Any JPG/PNG/WebP. It's scaled and letterboxed to fit 16:9 — a 1920×1080 image "
            "fills the frame exactly with no bars."))
        g.add(self.video_img_row)

        self.video_audio_row = Adw.ActionRow(title="Audio track", subtitle="none")
        ab = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        ab.connect("clicked", self._choose_audio, "video_audio")
        last_btn = Gtk.Button(label="Use last output", valign=Gtk.Align.CENTER)
        last_btn.connect("clicked", lambda _b: self._video_use_last())
        self.video_audio_row.add_suffix(last_btn)
        self.video_audio_row.add_suffix(ab)
        g.add(self.video_audio_row)

        self.video_style = Adw.ComboRow(
            title="Style",
            model=Gtk.StringList.new(["Still image", "Waveform overlay", "Spectrum overlay"]))
        self._info(self.video_style, "Still = the image held for the whole song (fastest "
                   "encode). Waveform/Spectrum add an animated audio visual over the lower "
                   "part of the image — more engaging for viewers, slower to encode.")
        g.add(self.video_style)

        self.video_text = Adw.EntryRow(title="Overlay text (blank = none)")
        self._info(self.video_text, "Title text drawn over the video — e.g. the track or "
                   "channel name. Rendered with the font below.")
        g.add(self.video_text)
        self._fonts = _bundled_fonts()
        font_names = [n for n, _f in self._fonts] + ["Custom TTF…"]
        self.video_font = Adw.ComboRow(title="Font", model=Gtk.StringList.new(font_names))
        g.add(self.video_font)
        self.video_font_custom = None
        self.video_font_row = Adw.ActionRow(title="Custom font file", subtitle="none")
        fb = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        fb.connect("clicked", self.on_choose_font)
        self.video_font_row.add_suffix(fb)
        g.add(self.video_font_row)
        self.video_fontsize = _spin(24, 240, 4, 96)
        self.video_fontsize.set_title("Text size (px)")
        g.add(self.video_fontsize)
        self.video_fontcolor = Adw.EntryRow(title="Text colour (#rrggbb)")
        self.video_fontcolor.set_text("#ffffff")
        g.add(self.video_fontcolor)
        self.video_textpos = Adw.ComboRow(title="Text position",
                                          model=Gtk.StringList.new(["Top", "Center", "Bottom"]))
        g.add(self.video_textpos)

        self.video_res = Adw.ComboRow(title="Resolution",
                                      model=Gtk.StringList.new(["1920×1080 (1080p)", "1280×720 (720p)"]))
        g.add(self.video_res)

        self.video_out_row = Adw.ActionRow(title="Output video", subtitle="auto (next to audio)")
        ob = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        ob.connect("clicked", self.on_choose_video_out)
        self.video_out_row.add_suffix(ob)
        g.add(self.video_out_row)
        card.append(g)

        self.video_btn = Gtk.Button(label="Make video")
        self.video_btn.add_css_class("suggested-action"); self.video_btn.add_css_class("pill")
        self.video_btn.connect("clicked", self.on_make_video)
        card.append(self.video_btn)

        self.video_open_btn = Gtk.Button(label="Open video")
        self.video_open_btn.set_sensitive(False)
        self.video_open_btn.connect("clicked", self._open_video)
        card.append(self.video_open_btn)

        # ----- batch: whole folder of WAVs -> MP4s with the same cover -----
        bcard = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        bcard.add_css_class("as-card")
        bintro = Gtk.Label(xalign=0, wrap=True)
        bintro.set_markup("<b>Batch export.</b>  Renders EVERY .wav in a folder to an MP4 "
                          "using the cover image and resolution above — pairs with the Batch "
                          "tab for one-click upload batches.")
        bintro.add_css_class("as-dim")
        bcard.append(bintro)
        bg = Adw.PreferencesGroup()
        self.vbatch_dir_row = Adw.ActionRow(title="WAV folder", subtitle="none")
        vb = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        vb.connect("clicked", self.on_vbatch_choose_dir)
        self.vbatch_dir_row.add_suffix(vb)
        bg.add(self.vbatch_dir_row)
        bcard.append(bg)
        self.vbatch_btn = Gtk.Button(label="Export folder to MP4s")
        self.vbatch_btn.add_css_class("pill")
        self.vbatch_btn.connect("clicked", self.on_vbatch_start)
        bcard.append(self.vbatch_btn)
        self.vbatch_status = Gtk.Label(label="", xalign=0, wrap=True)
        self.vbatch_status.add_css_class("as-dim")
        bcard.append(self.vbatch_status)
        outer.append(bcard)

        return self._scroller(outer)

    def on_vbatch_choose_dir(self, _btn):
        dlg = Gtk.FileDialog(title="Choose folder of WAVs")
        dlg.select_folder(self.win, None, self._vbatch_dir_done)

    def _vbatch_dir_done(self, dlg, result):
        try:
            gf = dlg.select_folder_finish(result)
            if gf:
                self.vbatch_dir = gf.get_path()
                self.vbatch_dir_row.set_subtitle(self.vbatch_dir)
        except GLib.Error:
            pass

    def on_vbatch_start(self, _btn):
        if self._video_encoding:
            return
        if not self.video_image:
            self.status.set_text("Choose a cover image first (above).")
            return
        d = getattr(self, "vbatch_dir", None)
        if not d or not os.path.isdir(d):
            self.status.set_text("Choose a WAV folder first.")
            return
        wavs = sorted(f for f in os.listdir(d) if f.lower().endswith(".wav"))
        if not wavs:
            self.status.set_text("No .wav files in that folder.")
            return
        ff = self._ffmpeg_path()
        if not ff:
            self.status.set_text("No encoder found (ffmpeg missing).")
            return
        res = "1280:720" if self.video_res.get_selected() == 1 else "1920:1080"
        self._video_encoding = True
        self.video_btn.set_sensitive(False)
        self.vbatch_btn.set_sensitive(False)
        GLib.timeout_add(120, self._pulse_video)
        threading.Thread(target=self._vbatch_run,
                         args=(ff, self.video_image, d, wavs, res), daemon=True).start()

    def _vbatch_run(self, ff, img, d, wavs, res):
        ok_n = 0
        for i, fn in enumerate(wavs, 1):
            audio = os.path.join(d, fn)
            out = os.path.splitext(audio)[0] + ".mp4"
            GLib.idle_add(self.vbatch_status.set_text,
                          f"[{i}/{len(wavs)}] {fn} → {os.path.basename(out)}")
            w, h = res.split(":")
            vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                  f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1")
            cmd = [ff, "-y", "-loglevel", "error", "-loop", "1", "-framerate", "2",
                   "-i", img, "-i", audio, "-c:v", "libx264", "-tune", "stillimage",
                   "-preset", "veryfast", "-pix_fmt", "yuv420p", "-vf", vf,
                   "-c:a", "aac", "-b:a", "192k", "-r", "2", "-shortest", out]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode == 0:
                    ok_n += 1
            except Exception:
                pass
        GLib.idle_add(self._vbatch_done, ok_n, len(wavs))

    def _vbatch_done(self, ok_n, total):
        self._video_encoding = False
        self.video_btn.set_sensitive(True)
        self.vbatch_btn.set_sensitive(True)
        self.progress.set_fraction(1.0 if ok_n == total else 0.0)
        self.progress.set_text("")
        self.vbatch_status.set_text(f"Done — {ok_n}/{total} videos exported.")
        self.status.set_text(f"Batch video export: {ok_n}/{total} MP4s written.")
        self._notify("Video export finished", f"{ok_n}/{total} MP4s written")
        return False

    # ----- Seed explorer tab (cheap drafts -> pick -> full render) -----
    def _seeds_page(self):
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        outer.set_margin_top(18); outer.set_margin_bottom(18)
        outer.set_margin_start(18); outer.set_margin_end(18)

        intro = Gtk.Label(xalign=0, wrap=True)
        intro.set_markup("<b>Seed explorer.</b>  Audition a prompt cheaply: N short low-step "
                         "drafts with different random seeds. Pick the vibe you like, then "
                         "render that exact seed properly at full steps/length.")
        intro.add_css_class("as-dim")
        outer.append(intro)

        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        card.add_css_class("as-card")
        outer.append(card)
        card.append(Gtk.Label(label="Prompt", xalign=0))
        sw, self.seed_prompt = _textview()
        card.append(sw)
        sg = Adw.PreferencesGroup()
        self.seed_count = _spin(2, 8, 1, 4); self.seed_count.set_title("Drafts")
        sg.add(self.seed_count)
        self.seed_dur = _spin(10, 60, 5, 20); self.seed_dur.set_title("Draft length (sec)")
        sg.add(self.seed_dur)
        self.seed_steps = _spin(2, 8, 1, 4); self.seed_steps.set_title("Draft steps")
        self._info(self.seed_steps, "Low steps = fast, rough sketches. The final render "
                   "below uses the full settings.")
        sg.add(self.seed_steps)
        self.seed_full_dur = _spin(10, MAX_SECONDS, 10, 120)
        self.seed_full_dur.set_title("Final render length (sec)")
        sg.add(self.seed_full_dur)
        self.seed_full_steps = _spin(1, 500, 1, 8)
        self.seed_full_steps.set_title("Final render steps")
        sg.add(self.seed_full_steps)
        card.append(sg)
        self.seed_btn = Gtk.Button(label="Generate drafts")
        self.seed_btn.add_css_class("suggested-action"); self.seed_btn.add_css_class("pill")
        self.seed_btn.connect("clicked", self.on_seed_explore)
        card.append(self.seed_btn)
        self._action_buttons.append(self.seed_btn)

        res = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        res.add_css_class("as-card")
        res.append(Gtk.Label(label="Drafts", xalign=0))
        self.seed_grid = Gtk.FlowBox(selection_mode=Gtk.SelectionMode.NONE,
                                     max_children_per_line=2, column_spacing=10,
                                     row_spacing=10)
        res.append(self.seed_grid)
        outer.append(res)

        self._seed_queue = []
        self._seed_running = False
        return self._scroller(outer)

    def on_seed_explore(self, _btn):
        prompt = _tv_text(self.seed_prompt)
        if not prompt:
            self.status.set_text("Enter a prompt first.")
            return
        while (c := self.seed_grid.get_first_child()) is not None:
            self.seed_grid.remove(c)
        n = int(self.seed_count.get_value())
        seeds = [int.from_bytes(os.urandom(4), "little") for _ in range(n)]
        self._seed_queue = [(prompt, s) for s in seeds]
        self._seed_running = True
        self._seed_next()

    def _seed_next(self):
        if not self._seed_queue:
            self._seed_running = False
            self.status.set_text("Drafts done — pick one and render it properly.")
            return
        prompt, seed = self._seed_queue.pop(0)
        self.last_prompt = prompt
        self._dispatch({
            "cmd": "generate", "prompt": prompt, "negative": "",
            "duration": int(self.seed_dur.get_value()),
            "steps": int(self.seed_steps.get_value()),
            "cfg": 1.0, "seed": seed, "sampler": "pingpong",
            "sigma_max": 1.0, "apg_scale": 1.0, "duration_padding": 6.0,
            "cut_to_duration": True, "spectrogram": False,
            "out": tmp_path(f"steppy_draft_{seed}.wav"),
        }, steps=int(self.seed_steps.get_value()))

    def _seed_draft_done(self, ev, sent):
        seed = sent.get("seed")
        path = ev.get("path")
        col = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        col.add_css_class("card")
        for m in ("top", "bottom", "start", "end"):
            getattr(col, f"set_margin_{m}")(6)
        col.append(Gtk.Label(label=f"seed {seed}", xalign=0))
        mc = Gtk.MediaControls()
        if path and os.path.isfile(path):
            mc.set_media_stream(Gtk.MediaFile.new_for_filename(path))
        col.append(mc)
        full = Gtk.Button(label="Render this seed properly")
        full.connect("clicked", lambda _b, s=seed, p=sent.get("prompt", ""):
                     self._seed_render_full(p, s))
        col.append(full)
        self.seed_grid.append(col)
        if self._seed_queue:
            GLib.idle_add(self._seed_next)
        else:
            self._seed_running = False

    def _seed_render_full(self, prompt, seed):
        self.last_prompt = prompt
        self._dispatch({
            "cmd": "generate", "prompt": prompt, "negative": "",
            "duration": int(self.seed_full_dur.get_value()),
            "steps": int(self.seed_full_steps.get_value()),
            "cfg": 1.0, "seed": int(seed), "sampler": "pingpong",
            "sigma_max": 1.0, "apg_scale": 1.0, "duration_padding": 6.0,
            "cut_to_duration": True, "spectrogram": True,
        }, steps=int(self.seed_full_steps.get_value()))

    # ----- History tab (auto-log of every generation) -----
    def _history_path(self):
        d = config.config_dir()
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "history.jsonl")

    def _history_append(self, rec):
        try:
            with open(self._history_path(), "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError:
            pass
        if hasattr(self, "hist_list"):
            self._history_refresh()

    def _history_load(self, limit=300):
        try:
            with open(self._history_path()) as fh:
                lines = fh.readlines()[-2000:]
        except OSError:
            return []
        out = []
        for ln in reversed(lines):
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
            if len(out) >= limit:
                break
        return out

    def _history_page(self):
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        outer.set_margin_top(18); outer.set_margin_bottom(18)
        outer.set_margin_start(18); outer.set_margin_end(18)

        # A/B compare card
        ab = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        ab.add_css_class("as-card")
        ab.append(Gtk.Label(label="A / B compare — send two tracks here from the list below",
                            xalign=0))
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12, homogeneous=True)
        for slot in ("a", "b"):
            col = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
            lbl = Gtk.Label(label=f"{slot.upper()}: —", xalign=0)
            lbl.add_css_class("as-dim")
            media = Gtk.MediaControls()
            setattr(self, f"ab_{slot}_label", lbl)
            setattr(self, f"ab_{slot}_media", media)
            col.append(lbl); col.append(media)
            row.append(col)
        ab.append(row)
        outer.append(ab)

        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        card.add_css_class("as-card")
        head = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        head.append(Gtk.Label(label="Every generation, newest first", xalign=0, hexpand=True))
        self.hist_search = Gtk.SearchEntry(placeholder_text="Search prompts…")
        self.hist_search.connect("search-changed", lambda _e: self._history_refresh())
        head.append(self.hist_search)
        clear_b = Gtk.Button(label="Clear…")
        clear_b.add_css_class("flat")
        clear_b.connect("clicked", self._history_clear)
        head.append(clear_b)
        card.append(head)
        self.hist_list = Gtk.ListBox()
        self.hist_list.add_css_class("boxed-list")
        sc = Gtk.ScrolledWindow(vexpand=True, min_content_height=420)
        sc.set_child(self.hist_list)
        card.append(sc)
        outer.append(card)
        self._history_refresh()
        return self._scroller(outer)

    def _history_refresh(self):
        q = (self.hist_search.get_text() or "").lower() if hasattr(self, "hist_search") else ""
        while (c := self.hist_list.get_first_child()) is not None:
            self.hist_list.remove(c)
        for rec in self._history_load():
            if q and q not in (rec.get("prompt") or "").lower():
                continue
            self.hist_list.append(self._history_row(rec))

    def _history_row(self, rec):
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        for m in ("top", "bottom", "start", "end"):
            getattr(row, f"set_margin_{m}")(8)
        info = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
        p = (rec.get("prompt") or "(no prompt)")[:110]
        title = Gtk.Label(label=p, xalign=0, wrap=True)
        meta = Gtk.Label(xalign=0)
        meta.add_css_class("as-dim")
        exists = rec.get("path") and os.path.isfile(rec["path"])
        meta.set_text(f"{rec.get('when', '')} · {rec.get('kind', 'generate')} · "
                      f"{rec.get('seconds', '?')}s · seed {rec.get('seed', '?')} · "
                      f"steps {rec.get('steps', '?')}"
                      + ("" if exists else " · (file gone)"))
        info.append(title); info.append(meta)
        row.append(info)

        def btn(label, cb, sensitive=True):
            b = Gtk.Button(label=label, valign=Gtk.Align.CENTER)
            b.add_css_class("flat")
            b.set_sensitive(sensitive)
            b.connect("clicked", cb)
            row.append(b)

        btn("▶", lambda _b: self._history_play(rec), exists)
        btn("→A", lambda _b: self._ab_set("a", rec), exists)
        btn("→B", lambda _b: self._ab_set("b", rec), exists)
        btn("Re-run", lambda _b: self._history_rerun(rec),
            rec.get("kind") == "generate" and self.ready)
        d = Gtk.Button(icon_name="user-trash-symbolic", valign=Gtk.Align.CENTER)
        d.add_css_class("flat")
        d.set_tooltip_text("Remove this entry (audio file is not deleted)")
        d.connect("clicked", lambda _b: self._history_delete(rec))
        row.append(d)
        lr = Gtk.ListBoxRow()
        lr.set_child(row)
        return lr

    def _history_delete(self, rec):
        """Remove one entry from history.jsonl (leaves the audio file alone)."""
        target = json.dumps(rec, sort_keys=True)
        try:
            with open(self._history_path()) as fh:
                lines = fh.readlines()
            kept, removed = [], False
            for ln in lines:
                try:
                    same = (not removed and
                            json.dumps(json.loads(ln), sort_keys=True) == target)
                except Exception:
                    same = False
                if same:
                    removed = True
                    continue
                kept.append(ln)
            with open(self._history_path(), "w") as fh:
                fh.writelines(kept)
        except OSError:
            pass
        self._history_refresh()

    def _history_clear(self, _btn):
        dlg = Adw.AlertDialog(heading="Clear all history?",
                              body="Removes every entry from the list. Your audio "
                                   "files are NOT deleted.")
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("clear", "Clear history")
        dlg.set_response_appearance("clear", Adw.ResponseAppearance.DESTRUCTIVE)

        def on_resp(_d, resp):
            if resp == "clear":
                try:
                    open(self._history_path(), "w").close()
                except OSError:
                    pass
                self._history_refresh()
        dlg.connect("response", on_resp)
        dlg.present(self.win)

    def _history_play(self, rec):
        if rec.get("path") and os.path.isfile(rec["path"]):
            self.last_wav = rec["path"]
            for mc in (self.s_media, self.a_media):
                mc.set_media_stream(Gtk.MediaFile.new_for_filename(rec["path"]))
            self.status.set_text(f"Loaded into players: {os.path.basename(rec['path'])}")

    def _ab_set(self, slot, rec):
        getattr(self, f"ab_{slot}_label").set_text(
            f"{slot.upper()}: {(rec.get('prompt') or '')[:60]} · seed {rec.get('seed')}")
        getattr(self, f"ab_{slot}_media").set_media_stream(
            Gtk.MediaFile.new_for_filename(rec["path"]))

    def _history_rerun(self, rec):
        params = dict(rec.get("params") or {})
        if not params:
            self.status.set_text("This entry has no stored settings to re-run.")
            return
        params["cmd"] = "generate"
        params.pop("out", None)
        if rec.get("seed") is not None:
            params["seed"] = int(rec["seed"])  # reproduce THIS track, not a new random
        self.last_prompt = params.get("prompt", "")
        self._dispatch(params, params.get("steps", 8))

    def _video_use_last(self):
        src = self.last_wav or self.longform_out
        if src and os.path.isfile(src):
            self._set_audio("video_audio", src)
        else:
            self.status.set_text("No generated track yet to use.")

    def on_choose_image(self, _btn):
        dlg = Gtk.FileDialog(title="Choose cover image")
        flt = Gtk.FileFilter(); flt.set_name("Images")
        for p in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
            flt.add_pattern(p)
        store = Gio.ListStore.new(Gtk.FileFilter); store.append(flt)
        dlg.set_filters(store)
        dlg.open(self.win, None, self._image_done)

    def _image_done(self, dlg, result):
        try:
            gf = dlg.open_finish(result)
            if gf:
                self.video_image = gf.get_path()
                self.video_img_row.set_subtitle(os.path.basename(self.video_image))
        except GLib.Error:
            pass

    def on_choose_video_out(self, _btn):
        dlg = Gtk.FileDialog(title="Save video as")
        dlg.set_initial_name("video.mp4")
        dlg.save(self.win, None, self._video_out_done)

    def _video_out_done(self, dlg, result):
        try:
            gf = dlg.save_finish(result)
            if gf:
                path = gf.get_path()
                if not path.lower().endswith(".mp4"):
                    path += ".mp4"
                self.video_out = path
                self.video_out_row.set_subtitle(os.path.basename(path))
        except GLib.Error:
            pass

    def _ffmpeg_path(self):
        if self._ffmpeg:
            return self._ffmpeg
        try:
            r = subprocess.run([VENV_PY, "-c",
                                "import imageio_ffmpeg as f; print(f.get_ffmpeg_exe())"],
                               capture_output=True, text=True, timeout=20)
            p = r.stdout.strip()
            if p and os.path.exists(p):
                self._ffmpeg = p
                return p
        except Exception:
            pass
        sysff = shutil.which("ffmpeg")
        if sysff:
            self._ffmpeg = sysff
        return self._ffmpeg

    def on_choose_font(self, _btn):
        dlg = Gtk.FileDialog(title="Choose a font file")
        flt = Gtk.FileFilter(); flt.set_name("Fonts")
        for p in ("*.ttf", "*.otf"):
            flt.add_pattern(p)
        store = Gio.ListStore.new(Gtk.FileFilter); store.append(flt)
        dlg.set_filters(store)
        dlg.open(self.win, None, self._font_done)

    def _font_done(self, dlg, result):
        try:
            gf = dlg.open_finish(result)
            if gf:
                self.video_font_custom = gf.get_path()
                self.video_font_row.set_subtitle(os.path.basename(self.video_font_custom))
                self.video_font.set_selected(len(self._fonts))  # select "Custom TTF…"
        except GLib.Error:
            pass

    def _video_opts(self):
        """Collect style/text options for the encode thread."""
        sel = self.video_font.get_selected()
        if sel < len(self._fonts):
            font_path = self._fonts[sel][1]
        else:
            font_path = self.video_font_custom
        return {
            "style": ["still", "waves", "spectrum"][self.video_style.get_selected()],
            "text": self.video_text.get_text().strip(),
            "font": font_path,
            "size": int(self.video_fontsize.get_value()),
            "color": self.video_fontcolor.get_text().strip() or "#ffffff",
            "pos": ["top", "center", "bottom"][self.video_textpos.get_selected()],
        }

    def _render_text_png(self, opts, w, h):
        """Render the overlay text to a transparent PNG via the bundled runtime's
        PIL (the GUI's system python has no imaging library). Returns path or None."""
        if not opts.get("text") or not opts.get("font"):
            return None
        out = tmp_path("steppy_text_overlay.png")
        script = (
            "import sys, json\n"
            "from PIL import Image, ImageDraw, ImageFont\n"
            "o = json.loads(sys.argv[1])\n"
            "img = Image.new('RGBA', (o['w'], o['h']), (0, 0, 0, 0))\n"
            "d = ImageDraw.Draw(img)\n"
            "f = ImageFont.truetype(o['font'], o['size'])\n"
            "c = o['color'].lstrip('#')\n"
            "rgb = tuple(int(c[i:i+2], 16) for i in (0, 2, 4)) if len(c) == 6 else (255, 255, 255)\n"
            "bb = d.multiline_textbbox((0, 0), o['text'], font=f, align='center')\n"
            "tw, th = bb[2] - bb[0], bb[3] - bb[1]\n"
            "x = (o['w'] - tw) // 2 - bb[0]\n"
            "y = {'top': int(o['h'] * 0.08), 'center': (o['h'] - th) // 2,\n"
            "     'bottom': o['h'] - th - int(o['h'] * 0.12)}[o['pos']] - bb[1]\n"
            "sh = max(2, o['size'] // 28)\n"
            "d.multiline_text((x + sh, y + sh), o['text'], font=f, fill=(0, 0, 0, 160), align='center')\n"
            "d.multiline_text((x, y), o['text'], font=f, fill=rgb + (255,), align='center')\n"
            "img.save(o['out'])\n")
        payload = json.dumps({"w": w, "h": h, "text": opts["text"], "font": opts["font"],
                              "size": opts["size"], "color": opts["color"],
                              "pos": opts["pos"], "out": out})
        try:
            r = subprocess.run([VENV_PY, "-c", script, payload],
                               capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and os.path.isfile(out):
                return out
            self.status.set_text(f"Text overlay failed: {r.stderr.strip()[-120:]}")
        except Exception as e:
            self.status.set_text(f"Text overlay failed: {e}")
        return None

    def on_make_video(self, _btn):
        if self._video_encoding:
            return
        if not self.video_image:
            self.status.set_text("Choose a cover image first.")
            return
        audio = self.video_audio or self.last_wav
        if not audio or not os.path.isfile(audio):
            self.status.set_text("Choose an audio track (or use last output).")
            return
        ff = self._ffmpeg_path()
        if not ff:
            self.status.set_text("No encoder found (ffmpeg missing).")
            return
        out = self.video_out or (os.path.splitext(audio)[0] + ".mp4")
        res = "1280:720" if self.video_res.get_selected() == 1 else "1920:1080"
        opts = self._video_opts()
        if opts["text"] and not opts["font"]:
            self.status.set_text("Pick a font (or choose a custom TTF) for the overlay text.")
            return
        self.video_btn.set_sensitive(False)
        self.video_open_btn.set_sensitive(False)
        self._video_encoding = True
        note = " — animated styles encode slower" if opts["style"] != "still" else ""
        self.status.set_text(f"Encoding video…{note}")
        self.progress.set_text("Encoding video…")
        GLib.timeout_add(120, self._pulse_video)
        threading.Thread(target=self._encode_video,
                         args=(ff, self.video_image, audio, out, res, opts), daemon=True).start()

    def _pulse_video(self):
        if self._video_encoding:
            self.progress.pulse()
            return True
        return False

    def _encode_video(self, ff, img, audio, out, res, opts=None):
        opts = opts or {"style": "still", "text": ""}
        w, h = (int(x) for x in res.split(":"))
        style = opts.get("style", "still")
        fps = 2 if style == "still" else 24
        text_png = self._render_text_png(opts, w, h) if opts.get("text") else None

        inputs = ["-loop", "1", "-framerate", str(fps), "-i", img, "-i", audio]
        fg = (f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
              f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1[bg]")
        last = "[bg]"
        if style == "waves":
            vh = h // 4
            fg += (f";[1:a]showwaves=s={w}x{vh}:mode=cline:rate={fps},"
                   f"format=rgba,colorchannelmixer=aa=0.85[viz]"
                   f";{last}[viz]overlay=0:{h - vh - h // 18}[v1]")
            last = "[v1]"
        elif style == "spectrum":
            vh = h // 3
            fg += (f";[1:a]showspectrum=s={w}x{vh}:mode=combined:color=intensity:"
                   f"scale=log:slide=scroll,fps={fps},format=rgba,"
                   f"colorchannelmixer=aa=0.8[viz]"
                   f";{last}[viz]overlay=0:{h - vh}[v1]")
            last = "[v1]"
        n_in = 2
        if text_png:
            inputs += ["-loop", "1", "-framerate", str(fps), "-i", text_png]
            fg += f";{last}[{n_in}:v]overlay=0:0[vout]"
            last = "[vout]"

        cmd = [ff, "-y", "-loglevel", "error"] + inputs + [
            "-filter_complex", fg, "-map", last, "-map", "1:a",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
        if style == "still":
            cmd += ["-tune", "stillimage"]
        cmd += ["-c:a", "aac", "-b:a", "192k", "-r", str(fps), "-shortest", out]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            ok = r.returncode == 0
            msg = out if ok else (r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "ffmpeg failed")
        except Exception as e:
            ok, msg = False, str(e)
        GLib.idle_add(self._video_done, ok, msg, out)

    def _video_done(self, ok, msg, out):
        self._video_encoding = False
        self.video_btn.set_sensitive(True)
        self.progress.set_fraction(1.0 if ok else 0.0)
        if ok:
            self.video_out = out
            self.progress.set_text("Video done")
            self.status.set_text(f"Video saved: {out}")
            self.video_open_btn.set_sensitive(True)
        else:
            self.progress.set_text("")
            self.status.set_text(f"Video error: {msg[:160]}")
        return False

    def _open_video(self, _btn):
        if self.video_out and os.path.isfile(self.video_out):
            try:
                Gio.AppInfo.launch_default_for_uri(f"file://{self.video_out}", None)
            except Exception as e:
                self.status.set_text(f"Could not open video: {e}")

    # ----- waveform inpaint region selector -----
    def _wav_peaks(self, path, buckets=1000):
        """Downsampled 0..1 peak magnitudes + duration from a WAV (stdlib only).

        STREAMS the file bucket-by-bucket — a multi-hour WAV costs a few MB,
        not gigabytes (it used to slurp the whole file into the GUI process)."""
        import wave as _wave
        import array as _array
        peaks = []
        with _wave.open(path, "rb") as wf:
            nch, sw, sr, n = (wf.getnchannels(), wf.getsampwidth(),
                              wf.getframerate(), wf.getnframes())
            if sw not in (1, 2, 4) or n == 0:
                return [], 0.0
            step = max(1, n // buckets)
            typ, norm = {1: ("B", 128.0), 2: ("h", 32768.0), 4: ("i", 2147483648.0)}[sw]
            read = 0
            while read < n:
                raw = wf.readframes(min(step, n - read))
                if not raw:
                    break
                read += len(raw) // (sw * nch)
                a = _array.array(typ); a.frombytes(raw)
                ch0 = a[0::nch] if nch > 1 else a
                if sw == 1:   # 8-bit WAV is unsigned, centered at 128
                    pk = max((abs(v - 128) for v in ch0), default=0)
                else:
                    pk = max((abs(v) for v in ch0), default=0)
                peaks.append(min(1.0, pk / norm))
        return peaks, (n / sr if sr else 0.0)

    def _draw_wave(self, area, cr, w, h, *_):
        cr.set_source_rgba(0, 0, 0, 0)
        cr.paint()
        mid = h / 2.0
        # waveform
        if self.wave_peaks:
            cr.set_source_rgba(0.55, 0.55, 0.6, 0.9)
            cr.set_line_width(1.0)
            npk = len(self.wave_peaks)
            for x in range(int(w)):
                pk = self.wave_peaks[min(npk - 1, int(x / max(1, w) * npk))]
                cr.move_to(x + 0.5, mid - pk * mid)
                cr.line_to(x + 0.5, mid + pk * mid)
            cr.stroke()
        else:
            cr.set_source_rgba(0.5, 0.5, 0.5, 0.5)
            cr.move_to(0, mid); cr.line_to(w, mid); cr.stroke()
        # committed regions
        if self.wave_dur > 0:
            cr.set_source_rgba(0.20, 0.60, 1.0, 0.30)
            for s, e in self.inpaint_regions:
                x0 = s / self.wave_dur * w
                x1 = e / self.wave_dur * w
                cr.rectangle(x0, 0, max(1, x1 - x0), h); cr.fill()
        # in-progress drag
        if self._drag is not None:
            x0, x1 = self._drag
            cr.set_source_rgba(0.20, 0.60, 1.0, 0.45)
            cr.rectangle(min(x0, x1), 0, abs(x1 - x0), h); cr.fill()

    def _wave_drag_begin(self, gesture, x, y):
        if not self.wave_peaks:
            return
        self._drag = (x, x)
        self.wave_area.queue_draw()

    def _wave_drag_update(self, gesture, dx, dy):
        if self._drag is None:
            return
        ok, sx, sy = gesture.get_start_point()
        x0 = sx if ok else self._drag[0]
        self._drag = (x0, x0 + dx)
        self.wave_area.queue_draw()

    def _wave_drag_end(self, gesture, dx, dy):
        if self._drag is None:
            return
        x0, x1 = self._drag
        self._drag = None
        w = max(1, self.wave_area.get_width())
        s = max(0.0, min(x0, x1) / w * self.wave_dur)
        e = min(self.wave_dur, max(x0, x1) / w * self.wave_dur)
        if e - s >= 0.05:
            self.inpaint_regions.append([round(s, 2), round(e, 2)])
            self._update_wave_hint()
        self.wave_area.queue_draw()

    def _clear_regions(self):
        self.inpaint_regions = []
        self._update_wave_hint()
        self.wave_area.queue_draw()

    def _update_wave_hint(self):
        if not self.wave_peaks:
            self.wave_hint.set_text("Load a WAV to mark regions by dragging.")
        elif not self.inpaint_regions:
            self.wave_hint.set_text("Drag on the waveform to mark region(s) to repaint.")
        else:
            txt = ", ".join(f"{s:.1f}–{e:.1f}s" for s, e in self.inpaint_regions)
            self.wave_hint.set_text(f"Regions: {txt}")

    def _duration_box(self, attr, value, label="Length"):
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        box.append(Gtk.Label(label=label))
        scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 1, MAX_SECONDS, 1)
        scale.set_value(value); scale.set_hexpand(True); scale.set_draw_value(True)
        scale.set_value_pos(Gtk.PositionType.RIGHT)
        box.append(scale)
        box.append(Gtk.Label(label="sec"))
        setattr(self, attr, scale)
        return box

    # ----- audio choosers -----
    def _choose_audio(self, _btn, which):
        dialog = Gtk.FileDialog(title="Choose audio")
        flt = Gtk.FileFilter(); flt.set_name("Audio (WAV/FLAC/MP3/OGG)")
        # only formats the bundled decoder actually handles — m4a always failed
        for p in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
            flt.add_pattern(p)
        filters = Gio.ListStore.new(Gtk.FileFilter); filters.append(flt)
        dialog.set_filters(filters)
        dialog.open(self.win, None, self._choose_audio_done, which)

    def _choose_audio_done(self, dialog, result, which):
        try:
            gfile = dialog.open_finish(result)
            if gfile:
                self._set_audio(which, gfile.get_path())
        except GLib.Error:
            pass

    def _set_audio(self, which, path):
        if which == "init":
            self.init_audio_path = path
            self.a_init_row.set_subtitle(os.path.basename(path) if path else "none")
        elif which == "coda":
            self.coda_clip_path = path
            for w in _widgets(self, "coda_clip_row"):
                w.set_subtitle(os.path.basename(path) if path else "none")
            for w in _widgets(self, "coda_analysis"): w.set_text("Key / tempo: —")
        elif which == "video_audio":
            self.video_audio = path
            self.video_audio_row.set_subtitle(os.path.basename(path) if path else "none")
        elif which == "lf_ref":
            self.lf_ref_path = path
            for w in _widgets(self, "lf_ref_row"):
                w.set_subtitle(os.path.basename(path) if path else "none")
        elif which == "lufs_file":
            if path:
                self._dispatch({"cmd": "lufs_file", "path": path, "target": -14.0,
                                "out": os.path.splitext(path)[0] + "_-14LUFS.wav"}, steps=1)
        else:
            self.inpaint_audio_path = path
            self.a_inpaint_row.set_subtitle(os.path.basename(path) if path else "none")
            self.inpaint_regions = []
            self.wave_peaks, self.wave_dur = [], 0.0
            if path and path.lower().endswith(".wav"):
                try:
                    self.wave_peaks, self.wave_dur = self._wav_peaks(path)
                except Exception:
                    self.wave_peaks, self.wave_dur = [], 0.0
            self._update_wave_hint()
            if hasattr(self, "wave_area"):
                self.wave_area.queue_draw()

    def _send_output_to(self, which):
        if self.last_wav:
            self._set_audio(which, self.last_wav)
            self.status.set_text(f"Sent output → {which} audio")

    # ---------- settings dialog ----------
    def on_open_settings(self, _btn):
        dlg = Adw.PreferencesDialog(title="Appearance")
        page = Adw.PreferencesPage(title="Appearance", icon_name="applications-graphics-symbolic")
        dlg.add(page)
        group = Adw.PreferencesGroup(title="Theme")
        page.add(group)

        self._themes = theme.all_themes(config.themes_dir())
        names = Gtk.StringList()
        cur_idx = 0
        for i, t in enumerate(self._themes):
            names.append(t.name)
            if t.id == self.settings["theme"]:
                cur_idx = i
        theme_row = Adw.ComboRow(title="Theme", model=names)
        theme_row.set_selected(cur_idx)
        theme_row.connect("notify::selected", self._on_theme_changed)
        group.add(theme_row)

        accent_row = Adw.EntryRow(title="Accent override (#rrggbb, blank = theme)")
        accent_row.set_text(self.settings["accent"])
        accent_row.connect("changed", self._on_accent_changed)
        group.add(accent_row)

        density_names = Gtk.StringList.new(["Comfortable", "Compact"])
        density_row = Adw.ComboRow(title="Density", model=density_names)
        density_row.set_selected(1 if self.settings["density"] == "compact" else 0)
        density_row.connect("notify::selected", self._on_density_changed)
        group.add(density_row)

        glow_row = _spin(0, 100, 5, self.settings["glow_intensity"])
        glow_row.set_title("Glow intensity")
        glow_row.set_subtitle("Only affects glow themes (Neon, Homebrew, Mono Cyan)")
        glow_row.connect("notify::value", self._on_glow_changed)
        group.add(glow_row)

        font_row = Adw.EntryRow(title="Font family (blank = system)")
        font_row.set_text(self.settings["font"])
        font_row.connect("changed", self._on_font_changed)
        group.add(font_row)

        out_group = Adw.PreferencesGroup(title="Output")
        page.add(out_group)
        fmts = [e for _n, e, _a in self.EXPORT_FORMATS]
        fmt_row = Adw.ComboRow(title="Default audio format",
                               subtitle="Used by Save audio… and preselected in Trim and export "
                                        "(WAV = classic behavior)",
                               model=Gtk.StringList.new([n for n, _e, _a in self.EXPORT_FORMATS]))
        cur = self.settings.get("export_format", "wav")
        fmt_row.set_selected(fmts.index(cur) if cur in fmts else 0)
        fmt_row.connect("notify::selected", self._on_export_format_changed)
        out_group.add(fmt_row)

        beh_group = Adw.PreferencesGroup(title="Behaviour and privacy")
        page.add(beh_group)
        tray_row = Adw.SwitchRow(title="Close to tray",
                                 subtitle="Closing the window keeps the app running "
                                          "as a tray icon; click it to bring it back")
        tray_row.set_active(bool(self.settings.get("close_to_tray", True)))
        tray_row.connect("notify::active", self._on_tray_changed)
        beh_group.add(tray_row)
        lock_row = Adw.SwitchRow(title="Lock screen",
                                 subtitle="PIN-protect the app; auto-locks when your "
                                          "session lock screen engages")
        lock_row.set_active(bool(self.settings.get("lock_enabled", False)))
        lock_row.connect("notify::active", self._on_lock_changed)
        beh_group.add(lock_row)
        pin_row = Adw.ActionRow(title="PIN",
                                subtitle="set" if self.settings.get("lock_hash") else "not set")
        pin_btn = Gtk.Button(label="Set PIN…", valign=Gtk.Align.CENTER)
        pin_btn.connect("clicked", lambda _b: self._set_pin_dialog(pin_row))
        pin_row.add_suffix(pin_btn)
        beh_group.add(pin_row)

        dlg.present(self.win)

    def _on_tray_changed(self, row, _p):
        self.settings["close_to_tray"] = bool(row.get_active())
        config.save(self.settings)

    def _on_lock_changed(self, row, _p):
        self.settings["lock_enabled"] = bool(row.get_active())
        config.save(self.settings)
        if self.settings["lock_enabled"] and not self.settings.get("lock_hash"):
            self.status.set_text("Lock enabled — set a PIN for it to take effect.")
        self.lock_btn.set_visible(self._lock_ready())

    def _set_pin_dialog(self, pin_row):
        dlg = Adw.AlertDialog(heading="Set PIN",
                              body="Used to unlock the app. Don't forget it — "
                                   "there is no recovery.")
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        p1 = Gtk.PasswordEntry(show_peek_icon=True, placeholder_text="New PIN")
        p2 = Gtk.PasswordEntry(show_peek_icon=True, placeholder_text="Confirm PIN")
        box.append(p1); box.append(p2)
        dlg.set_extra_child(box)
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("save", "Save PIN")
        dlg.set_response_appearance("save", Adw.ResponseAppearance.SUGGESTED)

        def on_resp(_d, resp):
            if resp != "save":
                return
            a, b = p1.get_text(), p2.get_text()
            if len(a) < 4:
                self.status.set_text("PIN must be at least 4 characters.")
                return
            if a != b:
                self.status.set_text("PINs didn't match — not saved.")
                return
            self.settings["lock_hash"] = self._pin_hash(a)
            config.save(self.settings)
            pin_row.set_subtitle("set")
            self.lock_btn.set_visible(self._lock_ready())
            self.status.set_text("PIN saved.")
        dlg.connect("response", on_resp)
        dlg.present(self.win)

    def _on_export_format_changed(self, row, _p):
        self.settings["export_format"] = self.EXPORT_FORMATS[row.get_selected()][1]
        config.save(self.settings)

    def _on_theme_changed(self, row, _p):
        self.settings["theme"] = self._themes[row.get_selected()].id
        self._apply_and_save()

    def _on_accent_changed(self, row):
        txt = row.get_text().strip()
        self.settings["accent"] = txt if (len(txt) == 7 and txt.startswith("#")) else ""
        self._apply_and_save()

    def _on_density_changed(self, row, _p):
        self.settings["density"] = "compact" if row.get_selected() == 1 else "comfortable"
        self._apply_and_save()

    def _on_glow_changed(self, row, _p):
        self.settings["glow_intensity"] = int(row.get_value())
        self._apply_and_save()

    def _on_font_changed(self, row):
        self.settings["font"] = row.get_text().strip()
        self._apply_and_save()

    def _apply_and_save(self):
        self._apply_theme()
        config.save(self.settings)

    # ---------- worker ----------
    def _start_worker(self):
        env = dict(os.environ, SF_MODEL_DIR=MODEL_DIR, SF_SA3_REPO=SA3_REPO,
                   PYTHONUNBUFFERED="1")
        # Keep the worker's error stream — silently discarding it hid a
        # "No working C++ compiler" message for weeks while decodes ran 8x slow.
        try:
            errlog = open(tmp_path("steppy_worker.log"), "w")
        except OSError:
            errlog = subprocess.DEVNULL
        self.proc = subprocess.Popen(
            [VENV_PY, WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=errlog, text=True, bufsize=1, env=env)
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            GLib.idle_add(self._on_event, ev)

    def _send(self, obj):
        try:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()
        except Exception as e:
            self.status.set_text(f"worker error: {e}")

    def _set_busy(self, busy):
        self._busy = busy
        if not busy:
            self._job_started = None
        for b in self._action_buttons:
            b.set_sensitive(not busy)
        # Remix needs both: not busy AND a track already generated.
        self.remix_btn.set_sensitive(not busy and self.last_wav is not None)
        # Keep Stop live for the whole batch; keep Start disabled until the batch ends.
        self.stop_btn.set_sensitive(busy or self.batch_running)
        if self.batch_running:
            self.batch_btn.set_sensitive(False)

    @staticmethod
    def _repair_wav_header(path):
        """Fix RIFF/data sizes of a WAV whose writer was killed (header says 0
        frames until a clean close). Makes 'partial file is kept' actually true."""
        try:
            size = os.path.getsize(path)
            if size <= 44:
                return False
            with open(path, "r+b") as f:
                if f.read(4) != b"RIFF":
                    return False
                f.seek(4); f.write(struct.pack("<I", size - 8))
                f.seek(40); f.write(struct.pack("<I", size - 44))
            return True
        except OSError:
            return False

    def on_stop(self, _btn):
        """Abort the current run by killing + respawning the worker (reloads model)."""
        was_batch = self.batch_running
        was_longform = self.longform_running
        self.batch_running = False
        self.longform_running = False
        self.batch_queue = []
        self.batch_current = None
        if self.proc:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None
        self.ready = False
        self.pending = None
        self._set_busy(False)
        self.stop_btn.set_sensitive(False)
        self.progress.set_fraction(0); self.progress.set_text("")
        if was_batch:
            self._batch_log(f"Batch stopped — {self.batch_done}/{self.batch_total} done.")
        if was_longform and self.longform_out and self._repair_wav_header(self.longform_out):
            self.status.set_text(f"Stopped — partial long-form saved: {self.longform_out} "
                                 "· reloading model…")
        else:
            self.status.set_text("Stopped — reloading model…")
        self._start_worker()

    def _on_event(self, ev):
        e = ev.get("event")
        if e == "loading":
            self.status.set_text("Loading model… (first launch can take a minute)")
        elif e == "ready":
            self.ready = True
            if self.pending is not None:
                p, self.pending = self.pending, None
                self.progress.set_text("Starting…")
                self.status.set_text("Working…")
                self._send(p)
            else:
                self._set_busy(False)
                self.status.set_text("Ready.")
                self.progress.set_fraction(0); self.progress.set_text("")
        elif e == "progress":
            tot = max(1, ev.get("total", self.cur_steps))
            self.progress.set_fraction(ev.get("step", 0) / tot)
            label = f"Generating… step {ev.get('step')}/{tot}"
            if ev.get("segments"):
                label = f"Segment {ev.get('segment')}/{ev.get('segments')} — step {ev.get('step')}/{tot}"
            elif ev.get("candidates", 1) > 1:
                label = f"Candidate {ev.get('candidate')}/{ev.get('candidates')} — step {ev.get('step')}/{tot}"
            self.progress.set_text(label)
        elif e == "analysis":
            key = ev.get("key") or "?"
            tempo = ev.get("tempo")
            tt = f"{tempo} BPM" if tempo else "tempo ?"
            clip = ev.get("clip_seconds")
            extra = f" · clip {clip}s" if clip else ""
            for w in _widgets(self, "coda_analysis"): w.set_text(f"Key / tempo: {key} · {tt}{extra}")
            if ev.get("standalone"):
                self.status.set_text("Analysis complete.")
                self._set_busy(False)
                self.progress.set_fraction(0); self.progress.set_text("")
        elif e == "preview":
            p = ev.get("path")
            if p and os.path.isfile(p):
                self.a_spec.set_filename(p)
        elif e == "partial":
            # decoded-so-far WAV, playable while the rest still decodes
            p = ev.get("path")
            if p and os.path.isfile(p):
                self.status.set_text(f"Decoding… {ev.get('seconds', 0):.0f}s already "
                                     "playable in the output player.")
                if not self._partial_loaded:
                    self._partial_loaded = True
                    for mc in (self.s_media, self.a_media):
                        mc.set_media_stream(Gtk.MediaFile.new_for_filename(p))
        elif e == "ntokens":
            self._on_ntokens(ev)
        elif e == "stage":
            name = ev.get("name")
            if name == "loading model":
                # one-time, and slow on CPU — say so, or it reads as a hang
                self.progress.set_fraction(0)
                self.progress.set_text("Loading model…")
                self.status.set_text("Loading model — first generate only.")
            elif name == "writing lyrics":
                self.progress.set_fraction(0)
                self.progress.set_text("Writing lyrics…")
                self.status.set_text("The lyric model is writing. Nothing renders yet.")
            elif name == "lyric writing failed":
                self.status.set_text("Lyric writing failed — continuing without lyrics.")
            elif name == "generating":
                self.progress.set_fraction(0)
                self.progress.set_text("Generating…")
                self.status.set_text("Working…")
            elif name == "decoding" and ev.get("total"):
                ch, tot = ev.get("chunk", 0), ev.get("total")
                self.progress.set_fraction((ch - 1) / tot)
                self.progress.set_text(f"Decoding audio… {ch}/{tot}")
            elif name == "candidate" and ev.get("total"):
                self.progress.set_text(f"Candidate {ev.get('chunk')}/{ev.get('total')}…")
            elif name == "segment" and ev.get("total"):
                self.progress.set_text(f"Segment {ev.get('chunk')}/{ev.get('total')}…")
                if self._seg_t0 is None:
                    self._seg_t0 = (time.time(), ev.get("chunk", 1))
            elif name == "stitched":
                eta = ""
                if self._seg_t0 and ev.get("chunk"):
                    t0, first = self._seg_t0
                    done_segs = ev["chunk"] - first + 1
                    if done_segs >= 1 and ev.get("total"):
                        per = (time.time() - t0) / done_segs
                        rem = per * (ev["total"] - ev["chunk"])
                        eta = f" · ≈{int(rem // 3600)}h{int(rem % 3600 // 60):02d}m left" \
                            if rem >= 3600 else f" · ≈{int(rem // 60)}m left"
                self.status.set_text(f"Long-form: {ev.get('seconds', 0):.0f}s rendered "
                                     f"({ev.get('chunk')}/{ev.get('total')} segments){eta}")
            elif name in ("measuring", "tiling") and ev.get("total"):
                self.progress.set_text(f"{name.capitalize()}… {ev.get('chunk')}/{ev.get('total')}")
            elif name == "writing":
                self.progress.set_text("Writing audio file…")
            elif name:
                # unknown stage: show what the worker actually said rather
                # than inventing a label for it
                self.progress.set_text(f"{name[:1].upper()}{name[1:]}…")
        elif e == "done":
            took = time.time() - self._job_started if self._job_started else 0
            self.last_wav = ev.get("path")
            self.last_duration = float(ev.get("seconds", self.last_duration))
            self.longform_running = False
            self.progress.set_fraction(1.0); self.progress.set_text("Done")
            extra = ""
            if ev.get("key"):
                extra = f" · {ev.get('key')}" + (f" {ev.get('tempo')}BPM" if ev.get("tempo") else "")
            if ev.get("lufs") is not None:
                extra += f" · {ev.get('lufs')} LUFS"
            if ev.get("loop") and ev.get("loop") != ev.get("path"):
                extra += f" · loop clip: {os.path.basename(ev['loop'])}"
            self.status.set_text(f"Done — seed {ev.get('seed')}{extra}")
            self._set_busy(False)
            saves = _widgets(self, "s_save", "a_save", "coda_save")
            if hasattr(self, "loop_save"):
                saves.extend(_widgets(self, "loop_save"))
            for sv in saves:
                sv.set_sensitive(True)
            players = _widgets(self, "s_media", "a_media", "coda_media", "lf_media")
            if hasattr(self, "loop_media"):
                players.extend(_widgets(self, "loop_media"))
            for mc in players:
                mc.set_media_stream(Gtk.MediaFile.new_for_filename(self.last_wav))
            spec = ev.get("spectrogram")
            if spec and os.path.isfile(spec):
                self.a_spec.set_filename(spec)
                for w in _widgets(self, "coda_spec"): w.set_filename(spec)
            if not self.batch_running and took > 120:
                self._notify("Track finished",
                             f"{ev.get('seconds', 0):.0f}s rendered — seed {ev.get('seed')}")
            sent = getattr(self, "_inflight", {}).pop(ev.get("id"), {})
            if os.path.basename(sent.get("out") or "").startswith("steppy_draft_"):
                self._seed_draft_done(ev, sent)
            if sent.get("cmd") not in (None, "analyze", "ntokens", "lufs_file"):
                self._history_append({
                    "when": time.strftime("%Y-%m-%d %H:%M"),
                    "kind": sent.get("cmd", "generate"),
                    "prompt": sent.get("prompt") or (sent.get("prompts") or [""])[0],
                    "seed": ev.get("seed"),
                    "steps": sent.get("steps"),
                    "seconds": ev.get("seconds"),
                    "path": ev.get("path"),
                    "params": {k: v for k, v in sent.items()
                               if k in ("cmd", "prompt", "negative", "duration", "steps",
                                        "cfg", "seed", "sampler", "sigma_max", "apg_scale",
                                        "duration_padding", "cut_to_duration", "lufs_target",
                                        "spectrogram")} if sent.get("cmd") == "generate" else None,
                })
            if self.batch_running:
                self._batch_advance(ok=True)
        elif e == "error":
            self.pending = None
            took = time.time() - self._job_started if self._job_started else 0
            self.longform_running = False
            getattr(self, "_inflight", {}).pop(ev.get("id"), None)
            self.status.set_text("Error: " + str(ev.get("msg")))
            self._set_busy(False)
            if getattr(self, "_seed_running", False) and self._seed_queue:
                GLib.idle_add(self._seed_next)
            if not self.batch_running and took > 120:
                self._notify("Generation failed", str(ev.get("msg"))[:120])
            if self.batch_running:
                self._batch_advance(ok=False, msg=str(ev.get("msg")))
        return False

    def _notify(self, title, body):
        try:
            n = Gio.Notification.new(title)
            n.set_body(body)
            self.send_notification(None, n)
        except Exception:
            pass

    # ---------- generate ----------
    @staticmethod
    def _seed_of(text):
        text = text.strip()
        return int(text) if text.lstrip("-").isdigit() else -1

    def _dispatch(self, params, steps):
        """Shared queue/busy/send path for any worker command (generate/continue/analyze)."""
        if not self.ready and self.pending is not None:
            # a request is already queued behind the model load — don't
            # silently overwrite it (that dropped the first request before)
            self.status.set_text("A request is already queued — wait for the model to load.")
            return
        self.gen_id += 1
        self.cur_steps = max(1, int(steps))
        params["id"] = self.gen_id
        params.setdefault("out", tmp_path(f"steppy_{self.gen_id}.wav"))
        if not hasattr(self, "_inflight"):
            self._inflight = {}
        self._inflight[self.gen_id] = dict(params)
        self._job_started = time.time()
        self._partial_loaded = False
        self._seg_t0 = None
        self._set_busy(True)
        for sv in _widgets(self, "s_save", "a_save", "coda_save"):
            sv.set_sensitive(False)
        self.progress.set_fraction(0)
        if not self.ready:
            # Queue it: fires automatically when the worker emits "ready".
            self.pending = params
            self.progress.set_text("Queued…")
            self.status.set_text("Queued — starts when the model finishes loading.")
            return
        self.progress.set_text("Starting…")
        self.status.set_text("Working…")
        self._send(params)

    def _launch(self, params):
        if not params.get("prompt"):
            self.status.set_text("Enter a prompt first.")
            return
        self.last_prompt = params["prompt"]
        params["cmd"] = "generate"
        self._dispatch(params, params.get("steps", 8))

    def _on_lyrics_mode(self, *_a):
        """Instrumental wins over everything; auto-lyrics hides the manual box."""
        inst = self.s_instrumental.get_active()
        auto = self.s_auto_lyrics.get_active()
        if inst and auto:
            self.s_auto_lyrics.set_active(False)
            auto = False
        self.s_auto_lyrics.set_sensitive(not inst)
        show_box = not inst and not auto
        for w in (self.s_lyrics_label, self.s_lyrics_scroller, self.s_lyrics_hint):
            w.set_visible(show_box)

    def on_generate_simple(self, _btn):
        sm = self.s_sampler.get_selected_item()
        self._launch({
            "prompt": _tv_text(self.s_prompt),
            "negative": "",
            "duration": int(self.s_dur.get_value()),
            "steps": int(self.s_steps.get_value()),
            "cfg": float(self.s_cfg.get_value()),
            "seed": self._seed_of(self.s_seed.get_text()),
            "sampler": sm.get_string() if sm else "pingpong",
            "sigma_max": 1.0, "apg_scale": 1.0, "duration_padding": 6.0,
            "cut_to_duration": True, "spectrogram": True,
            # ---- lyrics engine ----
            "lyrics": _tv_text(self.s_lyrics),
            "auto_lyrics": self.s_auto_lyrics.get_active(),
            "instrumental": self.s_instrumental.get_active(),
            "bpm": (int(self.s_bpm.get_value()) or None),
            "keyscale": self.s_keyscale.get_text().strip(),
            "timesignature": self.s_timesig.get_text().strip(),
            "language": self.s_language.get_text().strip() or "en",
        })

    def on_generate_advanced(self, _btn):
        sm = self.a_sampler.get_selected_item()
        params = {
            "prompt": _tv_text(self.a_prompt),
            "negative": self.a_negative.get_text().strip(),
            "duration": int(self.a_dur.get_value()),
            "steps": int(self.a_steps.get_value()),
            "cfg": float(self.a_cfg.get_value()),
            "seed": self._seed_of(self.a_seed.get_text()),
            "sampler": sm.get_string() if sm else "pingpong",
            "sigma_max": float(self.a_sigma.get_value()),
            "apg_scale": float(self.a_apg.get_value()),
            "duration_padding": float(self.a_padding.get_value()),
            "cut_to_duration": self.a_cut.get_active(),
            "init_audio": self.init_audio_path,
            "init_noise": float(self.a_init_noise.get_value()),
            "inpaint_audio": self.inpaint_audio_path,
            "mask_start": float(self.a_mask_start.get_value()),
            "mask_end": float(self.a_mask_end.get_value()),
            "preview_every": int(self.a_preview.get_value()),
            "spectrogram": True,
        }
        if self.a_lufs.get_active():
            params["lufs_target"] = -14.0
        # Waveform-drawn regions take precedence over the manual single mask.
        if self.inpaint_audio_path and self.inpaint_regions:
            params["mask_starts"] = [r[0] for r in self.inpaint_regions]
            params["mask_ends"] = [r[1] for r in self.inpaint_regions]
        elif self.inpaint_audio_path and \
                float(self.a_mask_end.get_value()) <= float(self.a_mask_start.get_value()):
            # inpaint audio chosen but NO region: the worker would silently
            # ignore it and do a plain generation — refuse instead of surprising.
            self.status.set_text("Inpaint audio is set but no region is marked — drag on the "
                                 "waveform or set Mask start/end (or clear the inpaint audio).")
            return
        self._launch(params)

    # ---------- Continue (CODA) ----------
    def on_coda_analyze(self, _btn):
        if not self.coda_clip_path:
            self.status.set_text("Choose a source clip first.")
            return
        self.coda_analysis.set_text("Analyzing…")
        self._dispatch({"cmd": "analyze", "clip": self.coda_clip_path}, steps=1)

    def on_coda_continue(self, _btn):
        if not self.coda_clip_path:
            self.status.set_text("Choose a source clip first.")
            return
        self._dispatch({
            "cmd": "continue",
            "clip": self.coda_clip_path,
            "continue_seconds": int(self.coda_cont.get_value()),
            "prompt": self.coda_prompt.get_text().strip(),
            "use_analysis": self.coda_use_analysis.get_active(),
            "analyze": True,
            "steps": int(self.coda_steps.get_value()),
            "cfg": float(self.coda_cfg.get_value()),
            "best_of": int(self.coda_best.get_value()),
            "context_seconds": float(self.coda_context.get_value()),
            "seed": self._seed_of(self.coda_seed.get_text()),
            "spectrogram": True,
        }, steps=int(self.coda_steps.get_value()))

    # ---------- Remix ----------
    def on_remix(self, _btn):
        if not self.last_wav:
            self.status.set_text("Generate a track first, then remix it.")
            return
        sm = self.a_sampler.get_selected_item()
        rp = self.remix_prompt.get_text().strip() or self.last_prompt or "variation"
        self._launch({
            "prompt": rp,
            "negative": self.a_negative.get_text().strip(),
            "duration": int(round(self.last_duration)),
            "steps": int(self.a_steps.get_value()),
            "cfg": float(self.a_cfg.get_value()),
            "seed": self._seed_of(self.remix_seed.get_text()),
            "sampler": sm.get_string() if sm else "pingpong",
            "sigma_max": float(self.a_sigma.get_value()),
            "apg_scale": float(self.a_apg.get_value()),
            "duration_padding": float(self.a_padding.get_value()),
            "cut_to_duration": True,
            "init_audio": self.last_wav,
            "init_noise": float(self.remix_strength.get_value()),
            "spectrogram": True,
        })

    # ---------- Batch / automated mode ----------
    @staticmethod
    def _slug(s):
        s = re.sub(r"[^\w\s-]", "", s).strip().lower()
        s = re.sub(r"[\s_-]+", "-", s)
        return s[:40] or "track"

    def on_batch_choose_dir(self, _btn):
        dlg = Gtk.FileDialog(title="Choose output folder")
        dlg.select_folder(self.win, None, self._batch_dir_done)

    def _batch_dir_done(self, dlg, result):
        try:
            gf = dlg.select_folder_finish(result)
            if gf:
                self.batch_dir = gf.get_path()
                self.batch_dir_row.set_subtitle(self.batch_dir)
        except GLib.Error:
            pass

    def _batch_log(self, text):
        b = self.batch_log.get_buffer()
        b.insert(b.get_end_iter(), text + "\n")
        self.batch_log.scroll_to_mark(b.create_mark(None, b.get_end_iter(), False),
                                      0.0, False, 0, 0)

    def on_batch_start(self, _btn):
        if self.batch_running:
            return
        if not self.batch_dir:
            self.status.set_text("Choose an output folder first.")
            return
        default_dur = int(self.batch_dur.get_value())
        queue = []
        for line in _tv_text(self.batch_prompts).splitlines():
            line = line.strip()
            if not line:
                continue
            dur = default_dur
            if "::" in line:
                p, _, tail = line.rpartition("::")
                tail = tail.strip()
                if tail.isdigit():
                    dur = max(1, min(MAX_SECONDS, int(tail)))
                    line = p.strip()
            if line:
                queue.append({"index": len(queue) + 1, "prompt": line, "duration": dur})
        if not queue:
            self.status.set_text("Add at least one prompt (one per line).")
            return
        self.batch_queue = queue
        self.batch_total = len(queue)
        self.batch_done = 0
        self.batch_current = None
        self.batch_running = True
        self.batch_open_btn.set_sensitive(True)
        self.batch_log.get_buffer().set_text("")
        self._batch_log(f"Batch of {self.batch_total} → {self.batch_dir}")
        self._batch_next()

    def _batch_next(self):
        item = self.batch_queue.pop(0)
        self.batch_current = item
        idx = item["index"]
        out = os.path.join(self.batch_dir, f"{idx:02d}_{self._slug(item['prompt'])}.wav")
        item["out"] = out
        sm = self.batch_sampler.get_selected_item()
        self.status.set_text(f"Batch {idx}/{self.batch_total}: {item['prompt'][:50]}")
        self._batch_log(f"▷ {idx:02d}  {item['prompt'][:60]}")
        self.last_prompt = item["prompt"]
        self._dispatch({
            "cmd": "generate",
            "prompt": item["prompt"],
            "negative": "",
            "duration": item["duration"],
            "steps": int(self.batch_steps.get_value()),
            "cfg": float(self.batch_cfg.get_value()),
            "seed": -1,
            "sampler": sm.get_string() if sm else "pingpong",
            "sigma_max": 1.0, "apg_scale": 1.0, "duration_padding": 6.0,
            "cut_to_duration": True,
            "spectrogram": self.batch_spec.get_active(),
            "out": out,
            # ---- lyrics engine: fresh lyrics per prompt ----
            "auto_lyrics": self.batch_auto_lyrics.get_active(),
            "instrumental": self.batch_instrumental.get_active(),
            "lyrics": "",
            "language": "en",
        }, steps=int(self.batch_steps.get_value()))

    def _batch_advance(self, ok, msg=""):
        cur = self.batch_current
        self.batch_current = None
        if cur:
            if ok:
                self.batch_done += 1
                self._batch_log(f"  ✓ saved {os.path.basename(cur['out'])}")
            else:
                self._batch_log(f"  ✗ failed: {msg[:60]}")
        if self.batch_queue:
            self._batch_next()
        else:
            self.batch_running = False
            self._set_busy(False)
            self.progress.set_fraction(1.0); self.progress.set_text("Batch done")
            self.status.set_text(
                f"Batch complete — {self.batch_done}/{self.batch_total} saved to {self.batch_dir}")
            self._batch_log(f"Done. {self.batch_done}/{self.batch_total} saved.")

    def _open_batch_dir(self, _btn):
        if self.batch_dir:
            try:
                Gio.AppInfo.launch_default_for_uri(f"file://{self.batch_dir}", None)
            except Exception as e:
                self.status.set_text(f"Could not open folder: {e}")

    # ---------- Long-form ----------
    def on_longform_choose_out(self, _btn):
        dlg = Gtk.FileDialog(title="Save long-form track as")
        dlg.set_initial_name("longform.wav")
        # default somewhere findable — a 60-min render once "vanished" into ~/
        music = os.path.expanduser("~/Music")
        if os.path.isdir(music):
            dlg.set_initial_folder(Gio.File.new_for_path(music))
        dlg.save(self.win, None, self._longform_out_done)

    def _longform_out_done(self, dlg, result):
        try:
            gf = dlg.save_finish(result)
            if gf:
                path = gf.get_path()
                if not path.lower().endswith(".wav"):
                    path += ".wav"
                self.longform_out = path
                # FULL path, not basename — basename-only once convinced the
                # user a finished 60-min render hadn't saved at all
                self.lf_out_row.set_subtitle(path)
        except GLib.Error:
            pass

    def _open_longform_dir(self, _btn):
        if self.longform_out:
            d = os.path.dirname(self.longform_out) or os.path.expanduser("~")
            try:
                Gio.AppInfo.launch_default_for_uri(f"file://{d}", None)
            except Exception as e:
                self.status.set_text(f"Could not open folder: {e}")

    def on_longform_start(self, _btn):
        prompts = [l.strip() for l in _tv_text(self.lf_prompts).splitlines() if l.strip()]
        if not prompts:
            self.status.set_text("Add at least one theme prompt (or pick a preset).")
            return
        if not self.longform_out:
            self.status.set_text("Choose an output file first.")
            return
        sm = self.lf_sampler.get_selected_item()
        self.last_prompt = prompts[0]
        params = {
            "cmd": "longform",
            "prompts": prompts,
            "total_minutes": int(self.lf_minutes.get_value()),
            "segment_seconds": int(self.lf_segment.get_value()),
            "crossfade_seconds": int(self.lf_xfade.get_value()),
            "steps": int(self.lf_steps.get_value()),
            "cfg": float(self.lf_cfg.get_value()),
            "seed": self._seed_of(self.lf_seed.get_text()),
            "sampler": sm.get_string() if sm else "pingpong",
            "out": self.longform_out,
        }
        if self.lf_ref_path:
            params["reference"] = self.lf_ref_path
            params["reference_noise"] = float(self.lf_ref_strength.get_value())
        self.longform_running = True
        self._dispatch(params, steps=int(self.lf_steps.get_value()))

    # ---------- trim & format export ----------
    EXPORT_FORMATS = [("WAV (lossless)", "wav", ["-c:a", "pcm_s16le"]),
                      ("FLAC (lossless, ~half size)", "flac", ["-c:a", "flac"]),
                      ("MP3 320k", "mp3", ["-c:a", "libmp3lame", "-b:a", "320k"]),
                      ("OGG Vorbis q6", "ogg", ["-c:a", "libvorbis", "-q:a", "6"])]

    def on_trim_export(self, _btn):
        src = self.last_wav
        if not src or not os.path.isfile(src):
            self.status.set_text("Generate (or load from History) a track first.")
            return
        try:
            import wave as _wave
            with _wave.open(src, "rb") as wf:
                dur = wf.getnframes() / wf.getframerate()
        except Exception:
            dur = self.last_duration or 60.0
        dlg = Adw.Dialog(title="Trim and export", content_width=520, content_height=380)
        tbv = Adw.ToolbarView(); tbv.add_top_bar(Adw.HeaderBar())
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        for m in ("top", "bottom", "start", "end"):
            getattr(box, f"set_margin_{m}")(14)
        box.append(Gtk.Label(label=os.path.basename(src) + f" · {dur:.1f}s", xalign=0))
        g = Adw.PreferencesGroup()
        t0 = Adw.SpinRow.new_with_range(0.0, max(0.0, dur - 0.1), 0.5)
        t0.set_title("Start (sec)"); t0.set_digits(1); t0.set_value(0.0)
        g.add(t0)
        t1 = Adw.SpinRow.new_with_range(0.1, dur, 0.5)
        t1.set_title("End (sec)"); t1.set_digits(1); t1.set_value(dur)
        g.add(t1)
        fmt = Adw.ComboRow(title="Format", model=Gtk.StringList.new(
            [n for n, _e, _a in self.EXPORT_FORMATS]))
        _exts = [e for _n, e, _a in self.EXPORT_FORMATS]
        _cur = self.settings.get("export_format", "wav")
        fmt.set_selected(_exts.index(_cur) if _cur in _exts else 0)
        g.add(fmt)
        box.append(g)
        exp = Gtk.Button(label="Export…")
        exp.add_css_class("suggested-action"); exp.add_css_class("pill")
        box.append(exp)

        def do_export(_b):
            name, ext, args = self.EXPORT_FORMATS[fmt.get_selected()]
            s, e = float(t0.get_value()), float(t1.get_value())
            if e <= s:
                self.status.set_text("End must be after start.")
                return
            fd = Gtk.FileDialog(title="Export as")
            fd.set_initial_name(f"{APP_NAME.lower()}.{ext}")
            music = os.path.expanduser("~/Music")
            if os.path.isdir(music):
                fd.set_initial_folder(Gio.File.new_for_path(music))

            def done(d, result):
                try:
                    gf = d.save_finish(result)
                except GLib.Error:
                    return
                if not gf:
                    return
                out = gf.get_path()
                if not out.lower().endswith("." + ext):
                    out += "." + ext
                dlg.close()
                ff = self._ffmpeg_path()
                if not ff:
                    self.status.set_text("No encoder found (ffmpeg missing).")
                    return
                self.status.set_text("Exporting…")

                def run():
                    cmd = [ff, "-y", "-loglevel", "error", "-ss", str(s), "-to", str(e),
                           "-i", src] + args + [out]
                    try:
                        r = subprocess.run(cmd, capture_output=True, text=True)
                        ok = r.returncode == 0
                        msg = out if ok else (r.stderr.strip().splitlines() or ["export failed"])[-1]
                    except Exception as ex:
                        ok, msg = False, str(ex)
                    GLib.idle_add(self.status.set_text,
                                  f"Exported: {msg}" if ok else f"Export error: {msg[:140]}")
                threading.Thread(target=run, daemon=True).start()
            fd.save(self.win, None, done)
        exp.connect("clicked", do_export)
        tbv.set_content(box)
        dlg.set_child(tbv)
        dlg.present(self.win)

    # ---------- save ----------
    def on_save(self, _btn):
        if not self.last_wav:
            return
        ext = self.settings.get("export_format", "wav")
        dialog = Gtk.FileDialog(title="Save audio")
        dialog.set_initial_name(f"{APP_NAME.lower()}.{ext}")
        dialog.save(self.win, None, self._save_done)

    def _save_done(self, dialog, result):
        try:
            gfile = dialog.save_finish(result)
        except GLib.Error:
            return
        if not (gfile and self.last_wav):
            return
        out = gfile.get_path()
        ext = out.rsplit(".", 1)[-1].lower() if "." in os.path.basename(out) else "wav"
        if ext == "wav" or ext not in [e for _n, e, _a in self.EXPORT_FORMATS]:
            # classic behavior: straight copy of the WAV
            shutil.copyfile(self.last_wav, out)
            self.status.set_text("Saved: " + out)
            return
        args = next(a for _n, e, a in self.EXPORT_FORMATS if e == ext)
        ff = self._ffmpeg_path()
        if not ff:
            self.status.set_text("No encoder for that format (ffmpeg missing) — saved as WAV instead.")
            shutil.copyfile(self.last_wav, out)
            return
        src = self.last_wav
        self.status.set_text(f"Converting to {ext.upper()}…")

        def run():
            cmd = [ff, "-y", "-loglevel", "error", "-i", src] + args + [out]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True)
                ok = r.returncode == 0
                msg = out if ok else (r.stderr.strip().splitlines() or ["convert failed"])[-1]
            except Exception as ex:
                ok, msg = False, str(ex)
            GLib.idle_add(self.status.set_text,
                          f"Saved: {msg}" if ok else f"Save error: {msg[:140]}")
        threading.Thread(target=run, daemon=True).start()

    def do_shutdown(self):
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass
        Adw.Application.do_shutdown(self)


if __name__ == "__main__":
    import sys
    Frequency().run(sys.argv)

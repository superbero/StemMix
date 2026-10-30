"""Qt actions for saving and opening StemsMix projects.

The DAW window calls `save_project_dialog(window)` and
`open_project_dialog(window)`. Both return the chosen path or None on
cancel, and raise `ProjectLoadError` (for open) if the file is broken.

`install_shortcuts(window)` wires Ctrl+S / Ctrl+Shift+S / Ctrl+O.
`recent_files()` / `remember_file(path)` manage a small MRU list used by
the Open dialog as its starting directory.
"""
import os

from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from core.project_io import FILE_EXT, ProjectLoadError, load_project, save_project
from ui.style import COLORS


FILE_FILTER = "StemsMix Project (*{})".format(FILE_EXT)


# --------------------------------------------------------------------------
# Recent files (small MRU list in QSettings)
# --------------------------------------------------------------------------

def _settings():
    from PyQt6.QtCore import QSettings
    return QSettings("StemsMix", "StemsMix")


def recent_files():
    s = _settings()
    raw = s.value("recent_projects", [], list) or []
    return [p for p in raw if isinstance(p, str) and os.path.exists(p)]


def remember_file(path):
    s = _settings()
    paths = [p for p in (s.value("recent_projects", [], list) or [])
             if isinstance(p, str)]
    if path in paths:
        paths.remove(path)
    paths.insert(0, path)
    s.setValue("recent_projects", paths[:8])


def _default_start_dir():
    recents = recent_files()
    if recents:
        return os.path.dirname(recents[0])
    return os.path.expanduser("~")


# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------

def save_project_dialog(window, save_as=False):
    """Save the current project.

    If `save_as` is False and the project already has a path, saves
    directly. Otherwise prompts for a path. Returns the path or None.
    """
    if not getattr(window, "project", None) or not window.project.tracks:
        QMessageBox.information(
            window, "Nothing to save",
            "Load some stems first — there is no project to save yet.")
        return None

    path = getattr(window, "project_path", None)
    if save_as or not path:
        start = path or os.path.join(
            _default_start_dir(),
            (window.windowTitle() or "Untitled") + FILE_EXT)
        path, _ = QFileDialog.getSaveFileName(
            window, "Save Project As" if save_as else "Save Project",
            start, FILE_FILTER)
        if not path:
            return None
        if not path.lower().endswith(FILE_EXT):
            path = path + FILE_EXT

    try:
        written = save_project(path, window.project, window.dsp_chains)
    except Exception as e:
        QMessageBox.critical(
            window, "Save failed",
            "Could not write the project:\n\n" + str(e))
        return None

    window.project_path = written
    remember_file(written)
    window.status.setText("Saved: " + os.path.basename(written))
    return written


# --------------------------------------------------------------------------
# Open
# --------------------------------------------------------------------------

def open_project_dialog(window):
    """Prompt for a project file and load it into the window.

    Returns the loaded ProjectLoadResult (with .project and .missing) on
    success, or None if the user cancelled or the file is unreadable.
    """
    start = _default_start_dir()
    path, _ = QFileDialog.getOpenFileName(
        window, "Open Project", start, FILE_FILTER)
    if not path:
        return None

    try:
        result = load_project(path)
    except ProjectLoadError as e:
        QMessageBox.critical(
            window, "Cannot open project", str(e))
        return None
    except Exception as e:
        QMessageBox.critical(
            window, "Cannot open project",
            "Unexpected error while reading the file:\n\n" + str(e))
        return None

    if not result.project.tracks:
        QMessageBox.warning(
            window, "Empty project",
            "The project file contains no tracks.")
        return None

    if result.missing:
        _warn_missing(window, result.missing)

    window.project_path = path
    remember_file(path)
    return result


def _warn_missing(window, missing):
    """Show a warning listing clips whose WAV is gone. They're still in
    the project with their parameters, just silent."""
    shown = missing[:10]
    lines = ["  {}  ->  {}".format(t, p) for t, p in shown]
    if len(missing) > len(shown):
        lines.append("  ... and {} more".format(len(missing) - len(shown)))
    QMessageBox.warning(
        window, "Missing audio files",
        "The project opened, but these clips reference audio files that "
        "are no longer on disk. The clips are kept with their settings "
        "but will play silence until the files are restored:\n\n"
        + "\n".join(lines))


# --------------------------------------------------------------------------
# Shortcuts
# --------------------------------------------------------------------------

def install_shortcuts(window):
    """Wire Ctrl+S, Ctrl+Shift+S, Ctrl+O. Each shortcut calls back into
    the DAW window so the project state is read at the moment of saving."""
    sc_save = QShortcut(QKeySequence.StandardKey.Save, window)
    sc_save.activated.connect(lambda: save_project_dialog(window, save_as=False))

    sc_save_as = QShortcut(
        QKeySequence("Ctrl+Shift+S"), window)
    sc_save_as.activated.connect(
        lambda: save_project_dialog(window, save_as=True))

    sc_open = QShortcut(QKeySequence.StandardKey.Open, window)
    sc_open.activated.connect(lambda: _open_and_install(window))

    return [sc_save, sc_save_as, sc_open]


def _open_and_install(window):
    """Load a project and hand it to the window's install method."""
    result = open_project_dialog(window)
    if result is None:
        return
    installer = getattr(window, "_install_loaded_project", None)
    if installer is None:
        QMessageBox.critical(
            window, "Not supported",
            "This window does not know how to install a loaded project.")
        return
    installer(result)

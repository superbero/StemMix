"""Top-level menu bar for the DAW window.

Attaches a QMenuBar to the given QMainWindow and returns it. Uses
explicit QAction objects so a missing method on the window is obvious
(an error at connect time) rather than silently hiding the menu item.
"""
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QMenuBar


def build_menubar(window):
    """Create, attach, and return the menubar for `window`.

    Requirements on `window`:
        save_project(save_as: bool)
        open_project()
        import_stems()
        export_mix()
        export_stems()
        open_batch()
        undo()
        redo()
        toggle_playback()
        stop_playback()
        rewind()
    """
    bar = QMenuBar(window)
    bar.setNativeMenuBar(True)   # macOS native bar at the top of the screen

    # --- File ---
    file_menu = bar.addMenu("&File")

    act_open = QAction("Open Project…", window)
    act_open.setShortcut(QKeySequence.StandardKey.Open)
    act_open.triggered.connect(window.open_project)
    file_menu.addAction(act_open)

    act_import = QAction("Import Stems…", window)
    act_import.setShortcut(QKeySequence("Ctrl+Shift+I"))
    act_import.triggered.connect(window.import_stems)
    file_menu.addAction(act_import)

    file_menu.addSeparator()

    act_save = QAction("Save", window)
    act_save.setShortcut(QKeySequence.StandardKey.Save)
    act_save.triggered.connect(lambda: window.save_project(save_as=False))
    file_menu.addAction(act_save)

    act_save_as = QAction("Save As…", window)
    act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
    act_save_as.triggered.connect(lambda: window.save_project(save_as=True))
    file_menu.addAction(act_save_as)

    file_menu.addSeparator()

    act_export = QAction("Export Mix…", window)
    act_export.setShortcut(QKeySequence("Ctrl+E"))
    act_export.triggered.connect(window.export_mix)
    file_menu.addAction(act_export)

    act_export_stems = QAction("Export Stems…", window)
    act_export_stems.setShortcut(QKeySequence("Ctrl+Shift+E"))
    act_export_stems.triggered.connect(window.export_stems)
    file_menu.addAction(act_export_stems)

    act_batch = QAction("Batch Separation…", window)
    act_batch.setShortcut(QKeySequence("Ctrl+B"))
    act_batch.triggered.connect(window.open_batch)
    file_menu.addAction(act_batch)

    file_menu.addSeparator()

    act_quit = QAction("Quit", window)
    act_quit.setShortcut(QKeySequence.StandardKey.Quit)
    act_quit.triggered.connect(window.close)
    file_menu.addAction(act_quit)

    # --- Edit ---
    edit_menu = bar.addMenu("&Edit")

    act_undo = QAction("Undo", window)
    act_undo.setShortcut(QKeySequence.StandardKey.Undo)
    act_undo.triggered.connect(window.undo)
    edit_menu.addAction(act_undo)

    act_redo = QAction("Redo", window)
    act_redo.setShortcut(QKeySequence("Ctrl+Shift+Z"))
    act_redo.triggered.connect(window.redo)
    edit_menu.addAction(act_redo)

    # --- Transport ---
    transport_menu = bar.addMenu("Trans&port")

    act_play = QAction("Play / Pause", window)
    act_play.setShortcut(QKeySequence("Space"))
    act_play.triggered.connect(window.toggle_playback)
    transport_menu.addAction(act_play)

    act_stop = QAction("Stop", window)
    act_stop.setShortcut(QKeySequence("."))
    act_stop.triggered.connect(window.stop_playback)
    transport_menu.addAction(act_stop)

    act_rewind = QAction("Rewind", window)
    act_rewind.setShortcut(QKeySequence("Home"))
    act_rewind.triggered.connect(window.rewind)
    transport_menu.addAction(act_rewind)

    # Attach to the window. QMainWindow keeps the menu alive, so no
    # explicit reference is required — but storing it on `window` makes
    # debugging easier.
    window.setMenuBar(bar)
    window._menubar = bar
    return bar

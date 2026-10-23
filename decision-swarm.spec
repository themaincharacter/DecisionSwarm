# PyInstaller spec for the decision-swarm launcher binary.
#
# The binary is a small launcher: model weights resolve from the shared cache
# (~/.decision-swarm/models) and workers run on per-package env interpreters.
# Registry + templates ship as data; the pure-Python decision_swarm sources are
# bundled as decision_swarm_src so worker PYTHONPATHs can import them.
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("decision_swarm")

a = Analysis(
    ["src/decision_swarm/cli.py"],
    pathex=["src"],
    binaries=[],
    datas=[
        ("registry", "registry"),
        ("templates", "templates"),
        ("src/decision_swarm", "decision_swarm_src/decision_swarm"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="decision-swarm",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,
)

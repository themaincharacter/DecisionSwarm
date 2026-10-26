# PyInstaller spec for the decision-swarm launcher binary.
#
# The binary is a small launcher: model weights resolve from the shared cache
# (~/.decision-swarm/models) and workers run on per-package env interpreters.
# Registry + templates ship as data; the pure-Python decision_swarm sources are
# bundled as decision_swarm_src so worker PYTHONPATHs can import them.
#
# `fetch` runs inside this binary, so the model downloader has to be inside it
# too. ``cache.hub_download`` imports ``huggingface_hub`` lazily (inside the
# function, so ``--version`` and ``models`` stay fast), which means a build
# environment that only has ``pyinstaller`` installed silently produces a
# launcher that dies with ``ModuleNotFoundError: No module named
# 'huggingface_hub'`` the first time a user clicks Download. The checks below
# turn that silent breakage into a failed build.
import importlib.util

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

REQUIRED_AT_RUNTIME = ("huggingface_hub",)

missing = [name for name in REQUIRED_AT_RUNTIME if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "decision-swarm.spec: "
        + ", ".join(missing)
        + " is not installed in the build environment; run 'python -m pip install -e .' "
        "before pyinstaller, or the launcher will not be able to download models."
    )

hiddenimports = collect_submodules("decision_swarm")
binaries: list = []
datas: list = []

# huggingface_hub exposes most of its surface lazily (PEP 562 ``__getattr__`` +
# ``importlib.import_module``), which static analysis cannot follow, so collect
# the whole package: submodules, its data files and its metadata (version
# reporting goes through ``importlib.metadata``). ``hf_xet`` is the optional
# Rust transfer backend that huggingface_hub imports dynamically; bundle it
# whenever the build environment has it, otherwise xet-backed repos fall back to
# plain HTTP downloads.
for package in ("huggingface_hub", "hf_xet"):
    if importlib.util.find_spec(package) is None:
        continue
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

datas += copy_metadata("huggingface-hub")

a = Analysis(
    ["src/decision_swarm/cli.py"],
    pathex=["src"],
    binaries=binaries,
    datas=[
        ("registry", "registry"),
        ("templates", "templates"),
        ("src/decision_swarm", "decision_swarm_src/decision_swarm"),
    ]
    + datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

packaged = {name for name, _path, _typecode in a.pure}
unpackaged = [name for name in REQUIRED_AT_RUNTIME if name not in packaged]
if unpackaged:
    raise SystemExit(
        "decision-swarm.spec: "
        + ", ".join(unpackaged)
        + " was not packaged into the launcher; refusing to build a binary whose "
        "'fetch' cannot download models."
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

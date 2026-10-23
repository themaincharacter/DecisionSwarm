# templates/runtime

Templates for the runtime package that `decision_swarm.pack.build_pack()`
generates into a swarm pack's `runtime/` directory (pip-installable package +
CLI entrypoint).

| template | output |
|---|---|
| `pyproject.toml.tmpl` | `runtime/pyproject.toml` |
| `__init__.py.tmpl` | `runtime/src/<pack_module>/__init__.py` |
| `cli.py.tmpl` | `runtime/src/<pack_module>/cli.py` |

Placeholders, substituted by the builder: `{{PACK_NAME}}` (manifest name),
`{{PACK_SLUG}}` (kebab-case project/script name), `{{PACK_MODULE}}`
(snake_case module name). The builder also copies `swarm.json` into the
package directory at build time so the runtime is self-contained.

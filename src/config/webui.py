"""Control panel (Config UI).

A small LOCAL web server (127.0.0.1 only, no internet, no new dependencies:
standard library) that shows the parameters of params.yaml in a form with
plain-language explanations. Flow designed for non-programmers:

  1. edit the fields you want;
  2. "Preview": the panel shows ONLY the lines that would change
     (from -> to), without touching the file;
  3. "Confirm and save": surgical edit of params.yaml (comments stay,
     automatic backup in params.yaml.bak).

Start it with `python cli.py config-ui` or from menu item 11; close it with
the "Close the panel" button (or Ctrl+C in the terminal).
"""
from __future__ import annotations

import html
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from .editor import ParamsEditor
from .schema import SECTIONS, Field, all_fields, validate

STYLE = """
body{font-family:-apple-system,'Segoe UI',Roboto,sans-serif;background:#f4f5f7;
     margin:0;color:#1d2733}
.wrap{max-width:860px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:22px} h2{font-size:16px;margin:28px 0 4px;border-bottom:2px solid #d8dee7;
   padding-bottom:6px}
.sec-desc{color:#5b6673;font-size:13px;margin:0 0 12px}
.field{background:#fff;border:1px solid #e2e7ee;border-radius:8px;
       padding:10px 14px;margin:8px 0;display:flex;gap:14px;align-items:center}
.field .texts{flex:1} .field label{font-weight:600;font-size:14px}
.field .help{color:#5b6673;font-size:12.5px;margin-top:2px}
input[type=text],input[type=number],select{padding:6px 8px;border:1px solid #b9c2ce;
   border-radius:6px;font-size:14px;width:170px;background:#fbfcfe}
input[type=checkbox]{transform:scale(1.35)}
.buttons{position:fixed;bottom:0;left:0;right:0;background:#fff;
   border-top:1px solid #d8dee7;padding:12px;text-align:center;box-shadow:0 -2px 8px #0001}
button{font-size:15px;padding:10px 22px;border-radius:8px;border:0;cursor:pointer;
   margin:0 6px}
.primary{background:#1a6ee0;color:#fff}.secondary{background:#e2e7ee}
.danger{background:#fff;color:#b3261e;border:1px solid #b3261e}
.error{background:#fdecea;border:1px solid #f5c6c0;color:#b3261e;
   border-radius:8px;padding:10px 14px;margin:8px 0}
.ok{background:#e8f5ec;border:1px solid #bfe3c8;color:#14652c;border-radius:8px;
   padding:10px 14px;margin:8px 0}
table{border-collapse:collapse;background:#fff;width:100%;margin:12px 0}
td,th{border:1px solid #e2e7ee;padding:8px 10px;font-size:14px;text-align:left}
th{background:#eef1f5}
code{background:#eef1f5;padding:1px 5px;border-radius:4px;font-size:13px}
.note{color:#5b6673;font-size:13px}
"""


def _e(s) -> str:
    return html.escape(str(s), quote=True)


# ---------------------------------------------------------------------- #
class ConfigPanel:
    """Panel logic, kept separate from HTTP so it can be tested."""

    def __init__(self, params_path: str = "params.yaml") -> None:
        self.editor = ParamsEditor(params_path)

    # ---------------- current values ----------------------------------- #
    @staticmethod
    def _navigate(tree: dict, key: str):
        """Value at the dotted key (handles the integer quadrant keys, e.g.
        vix.tdex_per_quad.1.sell_spx). (found, value)."""
        node = tree
        for pt in key.split("."):
            if isinstance(node, dict) and pt in node:
                node = node[pt]
            elif isinstance(node, dict) and pt.isdigit() and int(pt) in node:
                node = node[int(pt)]
            else:
                return False, None
        return True, node

    def current_value(self, key: str):
        return self._navigate(self.editor.read(), key)[1]

    def present_fields(self) -> list[Field]:
        """Only the fields whose key really exists in the file (optional
        blocks like tdex_per_quad may be missing)."""
        tree = self.editor.read()
        return [c for c in all_fields() if self._navigate(tree, c.key)[0]]

    # ---------------- form -> changes ----------------------------------- #
    def read_form(self, form: dict) -> tuple[dict, list[str]]:
        """Validates the form. Returns ({key: new_value}, [errors]),
        including ONLY the fields that differ from the file."""
        present = {c.key: c for c in self.present_fields()}
        changes: dict = {}
        errors: list[str] = []
        for key, fld in present.items():
            if key not in form and fld.kind != "bool":
                continue
            raw = (form.get(key) or [""])[-1]
            value, err = validate(fld, raw)
            if err:
                errors.append(err)
                continue
            current = self._current_normalized(key)
            if self._differs(value, current):
                changes[key] = value
        return changes, errors

    def _current_normalized(self, key: str):
        return self._navigate(self.editor.read(), key)[1]

    @staticmethod
    def _differs(new, current) -> bool:
        if isinstance(new, bool) or isinstance(current, bool):
            return bool(new) != bool(current)
        if isinstance(new, (int, float)) and isinstance(current, (int, float)):
            return float(new) != float(current)
        a = "" if current is None else str(current)
        b = "" if new is None else str(new)
        return a != b

    # ---------------- applying ------------------------------------------ #
    def apply(self, changes: dict) -> None:
        """Writes the changes to params.yaml: ONE backup of the original
        file, then surgical line-by-line edits (comments untouched)."""
        import shutil
        shutil.copyfile(self.editor.path, str(self.editor.path) + ".bak")
        for key, value in changes.items():
            parts = key.split(".")
            if len(parts) == 4 and parts[2].isdigit():
                # field inside a per-quadrant inline dict, e.g.
                # vix.tdex_per_quad.1.sell_spx: the whole quadrant line is
                # rewritten with the updated dict
                parent = ".".join(parts[:3])
                current = dict(self._current_normalized(parent) or {})
                current[parts[3]] = value
                self.editor.set(parent, current, backup=False)
            else:
                self.editor.set(key, value, backup=False)


# ---------------------------------------------------------------------- #
# HTML                                                                     #
# ---------------------------------------------------------------------- #
def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<title>{_e(title)}</title><style>{STYLE}</style></head>"
            f"<body><div class='wrap'><h1>{_e(title)}</h1>{body}</div></body></html>")


def _field_input(fld: Field, value) -> str:
    n = _e(fld.key)
    if fld.kind == "bool":
        check = "checked" if bool(value) else ""
        return (f"<input type='hidden' name='{n}' value='false'>"
                f"<input type='checkbox' name='{n}' value='true' {check}>")
    if fld.kind == "choice":
        opts = "".join(
            f"<option value='{_e(sc)}'{' selected' if str(value) == sc else ''}>"
            f"{_e(sc)}</option>" for sc in fld.choices)
        return f"<select name='{n}'>{opts}</select>"
    v = "" if value is None else value
    if isinstance(v, float) and v == int(v):
        v = int(v)
    return f"<input type='text' name='{n}' value='{_e(v)}'>"


def form_page(pan: ConfigPanel, errors: list[str] | None = None) -> str:
    present = {c.key for c in pan.present_fields()}
    blocks = []
    if errors:
        blocks.append("<div class='error'><b>Fix and try again:</b><br>"
                      + "<br>".join(_e(e) for e in errors) + "</div>")
    blocks.append("<p class='note'>Edit whatever you want and press "
                  "<b>Preview</b>: you will see only the lines that would "
                  "change, before saving. The comments in params.yaml stay "
                  "untouched and an automatic backup (.bak) is made.</p>")
    blocks.append("<form method='post' action='/preview'>")
    for sec in SECTIONS:
        fields = [c for c in sec.fields if c.key in present]
        if not fields:
            continue
        blocks.append(f"<h2>{_e(sec.name)}</h2>"
                      f"<p class='sec-desc'>{_e(sec.description)}</p>")
        for c in fields:
            val = pan.current_value(c.key)
            blocks.append(
                f"<div class='field'><div class='texts'>"
                f"<label>{_e(c.label)}</label>"
                f"<div class='help'>{_e(c.help)} — <code>{_e(c.key)}</code>"
                f"</div></div>{_field_input(c, val)}</div>")
    blocks.append(
        "<div class='buttons'>"
        "<button class='primary' type='submit'>Preview the changes</button>"
        "<button class='danger' formaction='/close' formmethod='post' "
        "formnovalidate>Close the panel</button></div></form>")
    return _page("Control panel — Tactical Bot", "".join(blocks))


def preview_page(pan: ConfigPanel, changes: dict, form: dict) -> str:
    labels = {c.key: c.label for c in all_fields()}
    rows = "".join(
        f"<tr><td>{_e(labels.get(k, k))}<br><code>{_e(k)}</code></td>"
        f"<td>{_e(pan._current_normalized(k))}</td>"
        f"<td><b>{_e(v if v != '' else '(empty)')}</b></td></tr>"
        for k, v in changes.items())
    hidden = "".join(
        f"<input type='hidden' name='{_e(k)}' value='{_e((vs or [''])[-1])}'>"
        for k, vs in form.items())
    body = (f"<p>These are the <b>{len(changes)}</b> changes that will be "
            f"written to <code>params.yaml</code> (automatic .bak backup):</p>"
            f"<table><tr><th>Parameter</th><th>Now</th><th>After</th></tr>"
            f"{rows}</table>"
            f"<form method='post' action='/save'>{hidden}"
            f"<div class='buttons'>"
            f"<button class='primary' type='submit'>Confirm and save</button>"
            f"<button class='secondary' formaction='/' formmethod='get' "
            f"formnovalidate>Go back</button></div></form>")
    return _page("Preview of the changes", body)


def saved_page(changes: dict) -> str:
    labels = {c.key: c.label for c in all_fields()}
    items = "".join(f"<li>{_e(labels.get(k, k))} → <b>{_e(v)}</b></li>"
                    for k, v in changes.items())
    body = (f"<div class='ok'>Saved. {len(changes)} parameters updated "
            f"in <code>params.yaml</code> (original in "
            f"<code>params.yaml.bak</code>).</div><ul>{items}</ul>"
            "<p class='note'>To see the effect: <b>Compute the signal</b> "
            "(menu item 3) or <b>Backtest</b> (item 4). If something does "
            "not look right, the previous file is in the .bak backup.</p>"
            "<div class='buttons'><form method='get' action='/'>"
            "<button class='primary'>Back to the panel</button></form></div>")
    return _page("Changes saved", body)


def nothing_page() -> str:
    body = ("<div class='ok'>No changes: the values in the form are the "
            "same as those already in params.yaml.</div>"
            "<div class='buttons'><form method='get' action='/'>"
            "<button class='primary'>Back to the panel</button></form></div>")
    return _page("No changes", body)


def closed_page() -> str:
    return _page("Panel closed",
                 "<p>You can close this browser window and go back to the "
                 "bot's menu.</p>")


# ---------------------------------------------------------------------- #
# HTTP server                                                             #
# ---------------------------------------------------------------------- #
def create_server(params_path: str = "params.yaml", port: int = 8766):
    pan = ConfigPanel(params_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):            # no console noise
            pass

        def _respond(self, body: str, code: int = 200) -> None:
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _form(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return parse_qs(self.rfile.read(n).decode("utf-8"),
                            keep_blank_values=True)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._respond(form_page(pan))
            else:
                self._respond(form_page(pan), 200)

        def do_POST(self):
            form = self._form()
            if self.path == "/close":
                self._respond(closed_page())
                threading.Thread(target=self.server.shutdown).start()
                return
            changes, errors = pan.read_form(form)
            if errors:
                self._respond(form_page(pan, errors))
            elif self.path == "/preview":
                self._respond(preview_page(pan, changes, form)
                              if changes else nothing_page())
            elif self.path == "/save":
                if changes:
                    pan.apply(changes)
                    self._respond(saved_page(changes))
                else:
                    self._respond(nothing_page())
            else:
                self._respond(form_page(pan), 404)

    for attempt in range(port, port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", attempt), Handler)
            return server, attempt
        except OSError:
            continue
    raise RuntimeError("no free port for the panel")


def run_config_ui(params_path: str = "params.yaml", port: int = 8766,
                  open_browser: bool = True) -> int:
    server, actual_port = create_server(params_path, port)
    url = f"http://127.0.0.1:{actual_port}/"
    print(f"\nControl panel running at {url}")
    print("Close it with the 'Close the panel' button (or Ctrl+C here).\n")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    print("Panel closed.")
    return 0

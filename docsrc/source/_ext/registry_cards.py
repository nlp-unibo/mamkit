"""A Sphinx directive rendering the library's registrations as cards.

The registry is the source of truth for what a key is called and what it
carries, so the page is built from it rather than written by hand and kept in
step. The directive builds the registry once per Sphinx run, groups the keys by
the kind of thing they register, and emits one expandable card per key with its
tags, its component and its parameters.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from docutils import nodes
from docutils.parsers.rst import Directive

_CACHE: list[dict[str, Any]] | None = None


def _default(field: Any) -> str:
    """The default of one field, as short text.

    A registration key is printed as its own tags rather than as its
    repr, which names the same object and is four times as long.
    """
    value = getattr(field, "default", None)
    if value is None or value is Ellipsis:
        return ""
    if hasattr(value, "tags") and hasattr(value, "name"):
        tags = ", ".join(sorted(value.tags or ()))
        return f"{value.name}[{tags}]" if tags else str(value.name)
    if isinstance(value, (list, tuple)):
        return ", ".join(_default(type("f", (), {"default": item})) for item in value)
    if isinstance(value, str):
        return repr(value)
    try:
        json.dumps(value)
    except TypeError:
        return type(value).__name__
    return str(value)


def _annotation(field: Any) -> str:
    annotation = getattr(field, "annotation", None)
    if annotation is None:
        return ""
    text = getattr(annotation, "__name__", None) or str(annotation)
    return text.replace("typing.", "").replace("mamkit.", "")


def _collect() -> list[dict[str, Any]]:
    """Every registration, with what a card needs to show it."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    import mamkit
    from cinnamon.registry import Registry

    Registry.build(directory=Path(mamkit.__file__).parent)

    cards = []
    for key in Registry.retrieve_keys():
        info = Registry.retrieve_configuration_info(key)
        config = Registry.retrieve_configuration(key)
        fields = []
        for name, field in getattr(config, "fields", {}).items():
            fields.append(
                {
                    "name": name,
                    "type": _annotation(field),
                    "default": _default(field),
                    "help": (getattr(field, "description", None) or "").strip(),
                }
            )
        cards.append(
            {
                "name": key.name,
                "tags": sorted(key.tags or ()),
                "namespace": key.namespace,
                "component": info.component or "",
                "runnable": bool(info.run_method),
                "fields": fields,
            }
        )
    cards.sort(key=lambda card: (card["name"], card["tags"]))
    _CACHE = cards
    return cards


def _card_html(card: dict[str, Any]) -> str:
    tags = "".join(
        f'<span class="rc-tag">{html.escape(tag)}</span>' for tag in card["tags"]
    )
    rows = "".join(
        "<tr>"
        f'<td><code>{html.escape(field["name"])}</code></td>'
        f'<td><code>{html.escape(field["type"])}</code></td>'
        f'<td><code>{html.escape(field["default"])}</code></td>'
        f'<td>{html.escape(field["help"])}</td>'
        "</tr>"
        for field in card["fields"]
    )
    table = (
        "<table class=\"rc-fields\"><thead><tr><th>Parameter</th><th>Type</th>"
        "<th>Default</th><th>What it does</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        if rows
        else '<p class="rc-empty">No parameters.</p>'
    )
    runnable = '<span class="rc-run">runnable</span>' if card["runnable"] else ""
    search = " ".join(
        [card["name"], *card["tags"], card["namespace"], card["component"]]
    ).lower()
    return (
        f'<details class="rc-card" data-search="{html.escape(search)}">'
        f'<summary><code class="rc-name">{html.escape(card["name"])}</code>{tags}{runnable}</summary>'
        f'<p class="rc-component"><code>{html.escape(card["component"])}</code></p>'
        f"{table}</details>"
    )


class RegistryCards(Directive):
    """Render every registration of one kind, or all of them."""

    has_content = False
    optional_arguments = 1

    def run(self):
        wanted = self.arguments[0] if self.arguments else None
        cards = [c for c in _collect() if wanted is None or c["name"] == wanted]
        if not cards:
            return [
                nodes.warning("", nodes.paragraph(text=f"no registration named {wanted}"))
            ]
        body = "".join(_card_html(card) for card in cards)
        markup = (
            '<div class="rc-block">'
            '<input class="rc-search" type="search" placeholder="Filter by key, tag or component" '
            'aria-label="Filter registrations">'
            f'<p class="rc-count"><span>{len(cards)}</span> registrations</p>'
            f"{body}</div>"
        )
        return [nodes.raw("", markup, format="html")]


def setup(app):
    app.add_directive("registry-cards", RegistryCards)
    app.add_css_file("registry.css")
    app.add_js_file("registry.js")
    return {"parallel_read_safe": False, "parallel_write_safe": True}

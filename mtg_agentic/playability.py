"""Mandatory approval/revision gates, separate from Commander format legality."""

from .mana import validate_mana_base
from .roles import validate_roles
from .spec import DeckSpec, ground_spec
from .mana import is_land


def spec_for_deck(deck, store, *, bracket=None):
    """Saved specs retain dependency policy; bare decks receive conservative defaults."""
    commander = deck["commander"]
    leader = store.get_card(commander)
    if leader is None:
        raise ValueError("Commander must resolve locally before playability review")
    identity = "".join(c for c in "WUBRG" if c in leader["color_identity"]) or "C"
    data = deck.get("spec")
    if data is not None:
        spec = DeckSpec.from_dict(data)
        if spec.commander != commander:
            raise ValueError("Deck and spec commander differ")
        data = spec.to_dict()
        if bracket is not None:
            data["bracket"] = bracket
    else:
        lands = sum(q for n, q in deck["cards"].items() if type(q) is int and q > 0 and (card := store.get_card(n)) and is_land(card))
        data = {
            "commander": commander, "color_identity": identity,
            "bracket": bracket if bracket is not None else deck.get("bracket", 3),
            "themes": [], "mechanics": [], "win_conditions": [], "playstyle": "unspecified",
            "must_include": [], "exclude": [], "commander_dependency": "medium",
            "role_targets": {"ramp": 10, "draw": 8, "removal": 8, "wipes": 0, "lands": lands, "creatures": 15, "protection": 4},
        }
    return ground_spec(DeckSpec.from_dict(data), store)


def validate_playability(cards, spec, store, *, tags=None):
    mana = validate_mana_base(cards, spec.commander, store)
    roles = validate_roles(cards, spec, store, tags)
    return {"valid": mana["valid"] and roles["valid"], "mana": mana, "roles": roles, "errors": mana["errors"] + roles["errors"]}

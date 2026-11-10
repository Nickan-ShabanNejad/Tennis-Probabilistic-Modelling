"""Tournament registry: name resolution, event type (level), surface, indoor, format.

Provider names vary wildly ("China Open", "Beijing", "China Open - Beijing",
"ATP Beijing"). Every name is reduced to a stable canonical key so that live
fixtures, odds, historical results and court-speed ratings all line up.
"""
from __future__ import annotations

import re
from functools import lru_cache
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    key: str
    name: str
    level: str          # "G", "F", "M", "500", "250"
    surface: str        # "Hard", "Clay", "Grass"
    indoor: bool
    aliases: tuple[str, ...] = ()

    @property
    def best_of(self) -> int:
        return 5 if self.level == "G" else 3


LEVEL_LABEL = {"G": "Grand Slam", "F": "ATP Finals", "M": "Masters 1000", "500": "ATP 500", "250": "ATP 250"}
LEVEL_RANK = {"250": 1.0, "500": 2.0, "M": 3.0, "F": 3.5, "G": 4.0}
BETTABLE_LEVELS = set(LEVEL_LABEL)

# key, display, level, surface, indoor, aliases (tokens that identify the event)
_EVENTS = [
    # Grand Slams
    ("australian open", "Australian Open", "G", "Hard", False, ("australian", "melbourne", "aus open")),
    ("roland garros", "Roland Garros", "G", "Clay", False, ("roland garros", "french open", "french")),
    ("wimbledon", "Wimbledon", "G", "Grass", False, ("wimbledon",)),
    ("us open", "US Open", "G", "Hard", False, ("us open", "u s open", "flushing")),
    # Finals
    ("atp finals", "ATP Finals", "F", "Hard", True, ("atp finals", "tour finals", "nitto", "masters cup", "turin")),
    # Masters 1000
    ("indian wells", "Indian Wells", "M", "Hard", False, ("indian wells", "bnp paribas open")),
    ("miami", "Miami", "M", "Hard", False, ("miami",)),
    ("monte carlo", "Monte Carlo", "M", "Clay", False, ("monte carlo", "monaco")),
    ("madrid", "Madrid", "M", "Clay", False, ("madrid", "mutua")),
    ("rome", "Rome", "M", "Clay", False, ("rome", "internazionali", "italian open", "roma")),
    ("canada", "Canadian Open", "M", "Hard", False, ("canada", "canadian", "national bank", "montreal", "toronto")),
    ("cincinnati", "Cincinnati", "M", "Hard", False, ("cincinnati", "western southern")),
    ("shanghai", "Shanghai", "M", "Hard", False, ("shanghai",)),
    ("paris masters", "Paris Masters", "M", "Hard", True, ("paris masters", "rolex paris", "bercy", "paris rolex")),
    # ATP 500
    ("dallas", "Dallas", "500", "Hard", True, ("dallas",)),
    ("rotterdam", "Rotterdam", "500", "Hard", True, ("rotterdam", "abn amro")),
    ("rio de janeiro", "Rio de Janeiro", "500", "Clay", False, ("rio de janeiro", "rio open")),
    ("doha", "Doha", "500", "Hard", False, ("doha", "qatar")),
    ("acapulco", "Acapulco", "500", "Hard", False, ("acapulco", "mexicano", "mexican open")),
    ("dubai", "Dubai", "500", "Hard", False, ("dubai",)),
    ("barcelona", "Barcelona", "500", "Clay", False, ("barcelona", "godo", "banc sabadell")),
    ("munich", "Munich", "500", "Clay", False, ("munich", "bmw open", "munchen")),
    ("hamburg", "Hamburg", "500", "Clay", False, ("hamburg",)),
    ("halle", "Halle", "500", "Grass", False, ("halle", "terra wortmann", "noventi")),
    ("queens club", "Queen's Club", "500", "Grass", False, ("queen", "queens", "hsbc championships", "cinch", "london queen")),
    ("washington", "Washington", "500", "Hard", False, ("washington", "citi open", "mubadala citi")),
    ("beijing", "Beijing", "500", "Hard", False, ("beijing", "china open")),
    ("tokyo", "Tokyo", "500", "Hard", False, ("tokyo", "japan open", "kinoshita", "rakuten")),
    ("basel", "Basel", "500", "Hard", True, ("basel", "swiss indoors")),
    ("vienna", "Vienna", "500", "Hard", True, ("vienna", "erste bank", "wien")),
    # ATP 250
    ("brisbane", "Brisbane", "250", "Hard", False, ("brisbane",)),
    ("hong kong", "Hong Kong", "250", "Hard", False, ("hong kong",)),
    ("adelaide", "Adelaide", "250", "Hard", False, ("adelaide",)),
    ("auckland", "Auckland", "250", "Hard", False, ("auckland", "asb classic")),
    ("montpellier", "Montpellier", "250", "Hard", True, ("montpellier", "occitanie", "sud de france")),
    ("buenos aires", "Buenos Aires", "250", "Clay", False, ("buenos aires", "argentina open")),
    ("delray beach", "Delray Beach", "250", "Hard", False, ("delray",)),
    ("marseille", "Marseille", "250", "Hard", True, ("marseille", "open 13", "provence")),
    ("santiago", "Santiago", "250", "Clay", False, ("santiago", "chile open")),
    ("houston", "Houston", "250", "Clay", False, ("houston", "clay court championship")),
    ("marrakech", "Marrakech", "250", "Clay", False, ("marrakech", "hassan")),
    ("bucharest", "Bucharest", "250", "Clay", False, ("bucharest", "tiriac")),
    ("geneva", "Geneva", "250", "Clay", False, ("geneva", "gonet")),
    ("lyon", "Lyon", "250", "Clay", False, ("lyon",)),
    ("stuttgart", "Stuttgart", "250", "Grass", False, ("stuttgart", "boss open")),
    ("s hertogenbosch", "'s-Hertogenbosch", "250", "Grass", False, ("hertogenbosch", "libema", "rosmalen")),
    ("mallorca", "Mallorca", "250", "Grass", False, ("mallorca",)),
    ("eastbourne", "Eastbourne", "250", "Grass", False, ("eastbourne",)),
    ("bastad", "Bastad", "250", "Clay", False, ("bastad", "nordea")),
    ("gstaad", "Gstaad", "250", "Clay", False, ("gstaad", "swiss open")),
    ("umag", "Umag", "250", "Clay", False, ("umag", "croatia open")),
    ("kitzbuhel", "Kitzbuhel", "250", "Clay", False, ("kitzbuhel", "generali")),
    ("estoril", "Estoril", "250", "Clay", False, ("estoril",)),
    ("atlanta", "Atlanta", "250", "Hard", False, ("atlanta",)),
    ("los cabos", "Los Cabos", "250", "Hard", False, ("los cabos", "cabo", "mifel")),
    ("winston salem", "Winston-Salem", "250", "Hard", False, ("winston",)),
    ("chengdu", "Chengdu", "250", "Hard", False, ("chengdu",)),
    ("hangzhou", "Hangzhou", "250", "Hard", False, ("hangzhou",)),
    ("almaty", "Almaty", "250", "Hard", True, ("almaty",)),
    ("astana", "Astana", "250", "Hard", True, ("astana",)),
    ("antwerp", "Brussels / Antwerp", "250", "Hard", True, ("antwerp", "brussels", "european open")),
    ("stockholm", "Stockholm", "250", "Hard", True, ("stockholm",)),
    ("athens", "Athens", "250", "Hard", True, ("athens", "hellenic")),
    ("metz", "Metz", "250", "Hard", True, ("metz", "moselle")),
    ("belgrade", "Belgrade", "250", "Hard", True, ("belgrade",)),
    ("sofia", "Sofia", "250", "Hard", True, ("sofia",)),
    ("florence", "Florence", "250", "Hard", True, ("florence",)),
    ("newport", "Newport", "250", "Grass", False, ("newport",)),
    ("cordoba", "Cordoba", "250", "Clay", False, ("cordoba",)),
    ("pune", "Pune", "250", "Hard", False, ("pune", "maharashtra")),
]

EVENTS: dict[str, Event] = {k: Event(k, n, lv, s, i, a) for k, n, lv, s, i, a in _EVENTS}

# Events that are not individual ATP-ranking tournaments and are never bet on.
EXCLUDED = re.compile(r"(davis cup|laver cup|united cup|atp cup|hopman|next gen|olympic|exhibition|six kings|ultimate tennis)", re.I)
LOWER_TIER = re.compile(
    r"(challenger|\bitf\b|futures?\b|^\s*[mw]\s?-?(?:15|25|35|50|75|100)\b|\butr\b|junior|\bch\s?\d{2,3}\b|\bwta\b|women|doubles)",
    re.I,
)
# Removed from the database entirely (no consistent history). Challengers are KEPT for
# ratings (TennisMyLife has every Challenger since 1978) but never bet on.
ITF_TIER = re.compile(r"(\bitf\b|futures?\b|^\s*[mw]\s?-?(?:15|25|35|50|75|100)\b|\butr\b|junior|\bwta\b|women|doubles)", re.I)
CHALLENGER = re.compile(r"(challenger|\bch\s?\d{2,3}\b)", re.I)
QUALIFYING = re.compile(r"(qualif|\bq[123]\b|^q\d?$)", re.I)


@lru_cache(maxsize=200000)
def _normalize_cached(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return " ".join(text.split())


def normalize(text) -> str:
    return _normalize_cached(str(text or ""))


def _event_text(name) -> str:
    text = normalize(name)
    text = re.sub(r"\b(atp|men s|mens|singles|tennis|20\d\d)\b", " ", text)
    return " " + " ".join(text.split()) + " "


def resolve(name) -> Event | None:
    return _resolve_cached(str(name or ""))


@lru_cache(maxsize=50000)
def _resolve_cached(name: str) -> Event | None:
    """Map any provider tournament name to a registry event (or None).

    Provider names often look like "<sponsor name> - <city>". Sponsors are shared
    between events (BNP Paribas sponsors Indian Wells AND Stockholm), so the city
    after " - " wins - except that a Grand Slam name always wins ("French Open - Paris"
    is Roland Garros, not the Paris Masters).
    """
    raw = str(name or "")
    if not raw.strip() or EXCLUDED.search(raw) or LOWER_TIER.search(raw):
        return None
    full = _best(_event_text(raw))
    if full is not None and full.level == "G":
        return full
    parts = [p for p in raw.split(" - ") if p.strip()]
    if len(parts) > 1:
        city = _best(_event_text(parts[-1]))
        if city is not None:
            return city
    if full is not None:
        return full
    # Plain "Paris" with no other context is the Paris Masters.
    if _event_text(raw).strip() == "paris":
        return EVENTS["paris masters"]
    # "ATP Finals" loses its "atp" token in _event_text; The Odds API calls it exactly that.
    if _event_text(raw).strip() == "finals":
        return EVENTS["atp finals"]
    return None


def _best(text: str) -> Event | None:
    # "French Open - Paris": the Slam must win over the Paris Masters key.
    best: tuple[int, Event] | None = None
    for event in EVENTS.values():
        for alias in event.aliases:
            if f" {alias} " in text or text.strip().startswith(alias + " ") or text.strip() == alias:
                score = len(alias)
                if event.level == "G":
                    score += 100  # a Slam token always dominates a city token
                if best is None or score > best[0]:
                    best = (score, event)
    return best[1] if best else None


def canonical(name) -> str:
    return _canonical_cached(str(name or ""))


@lru_cache(maxsize=50000)
def _canonical_cached(name: str) -> str:
    """Stable key for any tournament name; registry key when known."""
    ev = resolve(name)
    if ev is not None:
        return ev.key
    text = _event_text(name).strip()
    text = re.sub(r"\b(open|masters|championships?|international|cup|classic)\b", " ", text)
    return " ".join(text.split())


def encode_level(code) -> float:
    code = str(code or "").strip().upper()
    mapping = {"G": 4.0, "F": 3.5, "M": 3.0, "1000": 3.0, "500": 2.0, "250": 1.0, "A": 1.0, "D": 0.5, "C": 0.0, "O": 2.0}
    return mapping.get(code, 1.0)


def level_label(code) -> str:
    return LEVEL_LABEL.get(str(code), str(code))

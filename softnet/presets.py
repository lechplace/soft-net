"""
soft-net MLP presets — predefiniowane architektury sieci MLP.

Presety są wczytywane z ``softnet/presets.toml`` — możesz tam edytować
istniejące lub dodawać własne bez dotykania kodu Pythona.

Przykłady
---------
>>> from softnet.presets import list_presets, get_preset
>>> list_presets()
>>> clf = SoftClassifier.from_preset("medium", epochs=100)

>>> # własny plik z presetami
>>> from softnet.presets import load_presets_from_toml
>>> load_presets_from_toml("~/my_presets.toml")
>>> clf = SoftClassifier.from_preset("my_custom_net")
"""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Domyślny plik presetów dostarczany z pakietem
_BUILTIN_TOML = Path(__file__).parent / "presets.toml"


@dataclass
class MLPPreset:
    """
    Konfiguracja architektury sieci MLP.

    Parameters
    ----------
    name : str
        Unikalna nazwa presetu (klucz w TOML).

    layers : list of int
        Liczba neuronów w kolejnych warstwach ukrytych.
        Np. ``[256, 128, 64]`` → trzy warstwy ukryte.

    dropout : float, default 0.0
        Współczynnik dropout stosowany po każdej warstwie ukrytej.

    batch_norm : bool, default False
        Czy dodawać BatchNormalization po każdej warstwie.

    description : str
        Opis presetu wyświetlany przez ``list_presets()``.

    Examples
    --------
    >>> from softnet.presets import MLPPreset
    >>> custom = MLPPreset(
    ...     name="my_net",
    ...     layers=[512, 256, 128],
    ...     dropout=0.25,
    ...     batch_norm=True,
    ...     description="Moja sieć do fraud detection",
    ... )
    >>> from softnet.presets import register_preset
    >>> register_preset(custom)
    >>> clf = SoftClassifier.from_preset("my_net")
    """

    name: str
    layers: list[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.0
    batch_norm: bool = False
    description: str = ""

    def __repr__(self) -> str:
        return (
            f"MLPPreset(name={self.name!r}, layers={self.layers}, "
            f"dropout={self.dropout}, batch_norm={self.batch_norm})"
        )


# Globalny rejestr — wypełniany przy ładowaniu modułu
_REGISTRY: dict[str, MLPPreset] = {}


# ---------------------------------------------------------------------------
# API publiczne
# ---------------------------------------------------------------------------

def list_presets() -> None:
    """
    Wypisz wszystkie dostępne presety z opisami.

    Examples
    --------
    >>> from softnet.presets import list_presets
    >>> list_presets()
    ┌─────────────────┬──────────────────────────┬─────────┬────────────┐
    │ Name            │ Layers                   │ Dropout │ BatchNorm  │
    ├─────────────────┼──────────────────────────┼─────────┼────────────┤
    │ tiny            │ [64, 32]                 │  0.00   │     ✗      │
    ...
    """
    if not _REGISTRY:
        print("Brak zarejestrowanych presetów.")
        return

    name_w   = max(len(k) for k in _REGISTRY) + 2
    layer_w  = max(len(str(p.layers)) for p in _REGISTRY.values()) + 2

    header = (
        f"{'Name':<{name_w}}  {'Layers':<{layer_w}}  "
        f"{'Dropout':>8}  {'BatchNorm':>10}  Description"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for preset in _REGISTRY.values():
        bn = "✓" if preset.batch_norm else "✗"
        print(
            f"{preset.name:<{name_w}}  {str(preset.layers):<{layer_w}}  "
            f"{preset.dropout:>8.2f}  {bn:>10}  {preset.description}"
        )
    print(sep)


def get_preset(name: str) -> MLPPreset:
    """
    Zwróć preset o podanej nazwie.

    Parameters
    ----------
    name : str
        Nazwa presetu (case-sensitive). Dostępne nazwy → ``list_presets()``.

    Returns
    -------
    preset : MLPPreset

    Raises
    ------
    KeyError
        Jeśli preset o podanej nazwie nie istnieje.

    Examples
    --------
    >>> from softnet.presets import get_preset
    >>> p = get_preset("medium")
    >>> p.layers
    [256, 128, 64, 32]
    >>> p.dropout
    0.2
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(
            f"Preset {name!r} nie istnieje. "
            f"Dostępne: {available}\n"
            f"Uruchom list_presets() aby zobaczyć pełną listę."
        )
    return _REGISTRY[name]


def register_preset(preset: MLPPreset) -> None:
    """
    Zarejestruj własny preset w globalnym rejestrze.

    Jeśli preset o tej nazwie już istnieje — zostanie nadpisany.

    Parameters
    ----------
    preset : MLPPreset
        Instancja presetu do zarejestrowania.

    Examples
    --------
    >>> from softnet.presets import MLPPreset, register_preset
    >>> register_preset(MLPPreset(
    ...     name="fraud_net",
    ...     layers=[512, 256, 128, 64],
    ...     dropout=0.35,
    ...     batch_norm=True,
    ...     description="Zoptymalizowany dla niezbalansowanych danych",
    ... ))
    >>> from softnet import SoftClassifier
    >>> clf = SoftClassifier.from_preset("fraud_net")
    """
    _REGISTRY[preset.name] = preset


def load_presets_from_toml(path: str | Path) -> None:
    """
    Wczytaj presety z pliku TOML i dodaj je do globalnego rejestru.

    Istniejące presety o tych samych nazwach zostaną nadpisane.
    Pozwala to na tworzenie własnych kolekcji presetów projektowych.

    Parameters
    ----------
    path : str or Path
        Ścieżka do pliku ``.toml`` z presetami.
        Format identyczny jak ``softnet/presets.toml``.

    Examples
    --------
    >>> from softnet.presets import load_presets_from_toml, list_presets
    >>> load_presets_from_toml("~/projekty/moje_presety.toml")
    >>> list_presets()   # widoczne są teraz też twoje presety

    Przykładowy plik TOML
    ---------------------
    .. code-block:: toml

        [fraud_net]
        layers      = [512, 256, 128, 64]
        dropout     = 0.35
        batch_norm  = true
        description = "Zoptymalizowany dla niezbalansowanych danych"

        [quick_test]
        layers      = [32, 16]
        dropout     = 0.0
        batch_norm  = false
        description = "Minimalna sieć do szybkiego testowania pipeline'u"
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Plik presetów nie istnieje: {resolved}")

    with open(resolved, "rb") as f:
        data = tomllib.load(f)

    loaded = 0
    for name, cfg in data.items():
        if not isinstance(cfg, dict):
            continue
        preset = MLPPreset(
            name=name,
            layers=cfg.get("layers", [128, 64]),
            dropout=float(cfg.get("dropout", 0.0)),
            batch_norm=bool(cfg.get("batch_norm", False)),
            description=cfg.get("description", ""),
        )
        _REGISTRY[name] = preset
        loaded += 1

    print(f"[soft-net] Wczytano {loaded} presetów z: {resolved}")


# ---------------------------------------------------------------------------
# Ładowanie wbudowanych presetów przy imporcie modułu
# ---------------------------------------------------------------------------

def _load_builtin() -> None:
    if sys.version_info < (3, 11):
        try:
            import tomli as tomllib_compat  # type: ignore
        except ImportError:
            raise ImportError(
                "Python < 3.11 wymaga pakietu 'tomli': pip install tomli"
            )
        _load_from_module(tomllib_compat)
    else:
        _load_from_module(tomllib)


def _load_from_module(toml_module) -> None:
    with open(_BUILTIN_TOML, "rb") as f:
        data = toml_module.load(f)
    for name, cfg in data.items():
        if not isinstance(cfg, dict):
            continue
        _REGISTRY[name] = MLPPreset(
            name=name,
            layers=cfg.get("layers", [128, 64]),
            dropout=float(cfg.get("dropout", 0.0)),
            batch_norm=bool(cfg.get("batch_norm", False)),
            description=cfg.get("description", ""),
        )


_load_builtin()

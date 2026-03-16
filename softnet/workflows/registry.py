"""
Rejestr workflow presetów — ładowanie z TOML, rejestracja własnych.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_PRESETS_FILE = Path(__file__).parent / "presets.toml"


@dataclass
class WorkflowPreset:
    """
    Definicja presetu workflow.

    Parameters
    ----------
    name : str
        Unikalna nazwa presetu.
    steps : list of str
        Sekwencja kroków, np. ``["split", "scale", "fit", "validate"]``.
    step_params : dict
        Parametry domyślne dla każdego kroku, np.
        ``{"split": {"test_size": 0.2}, "scale": {"method": "standard"}}``.
    description : str
        Opis presetu wyświetlany przez ``list_workflows()``.
    """

    name: str
    steps: list[str]
    step_params: dict = field(default_factory=dict)
    description: str = ""


_REGISTRY: dict[str, WorkflowPreset] = {}


def _load_builtin() -> None:
    """Ładuje wbudowane presety z presets.toml."""
    with open(_PRESETS_FILE, "rb") as f:
        data = tomllib.load(f)

    step_defaults = data.pop("step_defaults", {})

    for name, cfg in data.items():
        _REGISTRY[name] = WorkflowPreset(
            name=name,
            steps=cfg["steps"],
            step_params=step_defaults,
            description=cfg.get("description", ""),
        )


_load_builtin()


# ── public API ────────────────────────────────────────────────────────────────

def list_workflows() -> None:
    """
    Wypisz dostępne presety workflow.

    Examples
    --------
    >>> from softnet.workflows import list_workflows
    >>> list_workflows()
    basic            split → fit → validate
    scaled           split → scale → fit → validate
    ...
    """
    max_name = max(len(n) for n in _REGISTRY)
    print(f"{'Preset':<{max_name + 2}}  Kroki")
    print("-" * 70)
    for name, preset in _REGISTRY.items():
        steps_str = " → ".join(preset.steps)
        print(f"{name:<{max_name + 2}}  {steps_str}")
        if preset.description:
            print(f"{'': <{max_name + 4}}{preset.description}")


def get_workflow(name: str) -> WorkflowPreset:
    """
    Pobierz preset po nazwie.

    Parameters
    ----------
    name : str
        Nazwa presetu.

    Returns
    -------
    WorkflowPreset

    Raises
    ------
    KeyError
        Jeśli preset nie istnieje. Użyj ``list_workflows()`` aby zobaczyć dostępne.

    Examples
    --------
    >>> preset = get_workflow("scaled")
    >>> preset.steps
    ['split', 'scale', 'fit', 'validate']
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(
            f"Workflow preset '{name}' nie istnieje. "
            f"Dostępne: {available}"
        )
    return _REGISTRY[name]


def register_workflow(preset: WorkflowPreset) -> None:
    """
    Zarejestruj własny preset workflow.

    Parameters
    ----------
    preset : WorkflowPreset
        Preset do zarejestrowania.

    Examples
    --------
    >>> from softnet.workflows.registry import WorkflowPreset, register_workflow
    >>> register_workflow(WorkflowPreset(
    ...     name="my_flow",
    ...     steps=["split", "scale", "fit", "validate", "save"],
    ...     step_params={"split": {"test_size": 0.1}, "save": {"path": "my_model"}},
    ...     description="Mój własny workflow.",
    ... ))
    """
    _REGISTRY[preset.name] = preset


def load_workflows_from_toml(path: str) -> None:
    """
    Wczytaj i zarejestruj presety z własnego pliku TOML.

    Format pliku jest identyczny jak ``softnet/workflows/presets.toml``.
    Nowe presety są dodawane do rejestru (nie nadpisują wbudowanych,
    chyba że mają tę samą nazwę).

    Parameters
    ----------
    path : str
        Ścieżka do pliku TOML.

    Examples
    --------
    Utwórz plik ``my_workflows.toml``:

    .. code-block:: toml

        [fraud_full]
        description = "Pełny cykl do fraud detection"
        steps = ["split", "scale", "grid_search", "validate", "save"]

        [step_defaults.split]
        test_size = 0.15

        [step_defaults.save]
        path = "models/fraud_model"

    Następnie wczytaj:

    >>> from softnet.workflows import load_workflows_from_toml
    >>> load_workflows_from_toml("my_workflows.toml")
    >>> list_workflows()   # fraud_full jest teraz dostępny
    """
    p = Path(path).expanduser()
    with open(p, "rb") as f:
        data = tomllib.load(f)

    step_defaults = data.pop("step_defaults", {})

    for name, cfg in data.items():
        _REGISTRY[name] = WorkflowPreset(
            name=name,
            steps=cfg["steps"],
            step_params=step_defaults,
            description=cfg.get("description", ""),
        )
    print(f"[soft-net] Wczytano {len(data)} workflow preset(ów) z {p}")

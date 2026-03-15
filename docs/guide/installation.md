# Instalacja

## Wymagania

- Python ≥ 3.10
- scikit-learn ≥ 1.3
- Keras ≥ 3.0 + TensorFlow ≥ 2.15

---

## pip

```bash
pip install soft-net
```

Z obsługą obrazów (Pillow, OpenCV):

```bash
pip install "soft-net[image]"
```

---

## uv (zalecane)

```bash
uv add soft-net
# lub z opcją image:
uv add "soft-net[image]"
```

---

## Instalacja deweloperska (edytowalna)

Gdy chcesz modyfikować kod frameworka i widzieć zmiany natychmiast:

```bash
git clone https://github.com/lechplace/soft-net
cd soft-net
uv venv && source .venv/bin/activate
uv pip install -e .
```

Osobny sandbox do eksperymentów (kod frameworka pozostaje czysty):

```bash
mkdir ~/sandbox && cd ~/sandbox
uv venv && source .venv/bin/activate
uv pip install -e /ścieżka/do/soft-net
```

---

## Budowanie dokumentacji lokalnie

```bash
# zainstaluj zależności docs
uv pip install -e ".[docs]"

# podgląd na żywo
mkdocs serve
# → http://127.0.0.1:8000

# statyczny build
mkdocs build
# → site/
```

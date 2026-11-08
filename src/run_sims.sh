#!/bin/bash
PY="../.venv/Scripts/python.exe"   # adjust if your .venv lives elsewhere

$PY load_schedule.py  || exit 1; echo "Schedule loaded."
$PY srs_model.py      || exit 1; echo "SRS model run."
$PY srs_scatter.py    || exit 1; echo "SRS scatter plotted."
$PY predict_probs.py  || exit 1; echo "Game probabilities calculated."
$PY simulate.py       || exit 1; echo "Simulations run."
$PY last_updated.py
.PHONY: sweep replot presentation help

## Full experiment run (train → benchmark → TTA → budget curves)
sweep:
	uv run python scripts/run.py sweep --clean

## Re-run benchmark and extras on existing trained weights (skip training)
replot:
	uv run python scripts/run.py sweep --plots-only

## Generate the PowerPoint presentation
presentation:
	uv run --group dev python scripts/run.py presentation

## Show all run.py options
help:
	uv run python scripts/run.py help

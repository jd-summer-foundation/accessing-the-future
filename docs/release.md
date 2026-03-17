# Release Checklist

Use this checklist before tagging a preprint companion release or creating a Zenodo snapshot.

## Required commands

```bash
make reproduce
make release-check
python3 -m pytest
```

## Release payload

- `results/baseline/` contains the canonical baseline run outputs and manifest
- `reports/` contains the canonical table, figures, and report manifest
- `data/checksums.sha256` matches the raw source workbooks used for reproduction
- `CITATION.cff` and `.zenodo.json` are present and up to date

## Metadata checks

- The project title matches across `CITATION.cff` and `.zenodo.json`
- The package version in `pyproject.toml` matches `CITATION.cff`
- The first author in `CITATION.cff` matches the first creator in `.zenodo.json`
- `README.md` still documents the canonical reproduction and release-check commands

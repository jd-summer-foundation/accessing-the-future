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

## Licensing checks

- `CITATION.cff` declares an SPDX identifier (`CC0-1.0`, covering the software) rather than a free-text pointer
- `.zenodo.json` declares `cc-by-4.0` — the deposit bundles ABS material licensed CC BY 4.0, so the record as a whole cannot be CC0 even though the code is
- `README.md` carries a `## Licensing` section and `data/README.md` carries `## Licensing and attribution`
- Every file in `data/raw/` is named in `data/README.md` with its licensing note

`make release-check` enforces all of the above. When depositing to a repository that asks for a single licence (Zenodo, openICPSR, Dataverse), select CC BY 4.0 and reproduce the ABS attribution wording from [data/README.md](../data/README.md#required-attribution) in the deposit description.

# sedfit

Roster-driven SED assembly and fitting for galaxy samples: eazy-py and
Prospector backends behind one shared core. One declaration of your
targets and data files drives photometry assembly, fitting, plotting,
and an append-only run manifest, with full provenance — every run
directory is self-contained and reproducible from its own contents.

Highlights:

- **One assembly path.** Per-target source CSVs (catalog pulls, your
  own aperture photometry, SPHEREx visit tables) are combined by
  declarative recipes: pick a reference frame, stitch or tilt
  instruments onto it, and write a fit-ready table plus a provenance
  sidecar.
- **SPHEREx ingest.** Strict quality-flag filtering and model-matched
  binning of IRSA per-visit spectrophotometry; per-object channels
  become exact narrow-band filters in both backends.
- **Two backends, one contract.** eazy-py (an official engine plus a
  vectorized quick engine implemented directly in numpy and scipy) and
  Prospector (seeded dynesty/emcee, checkpointing, stellar-library
  assertion). The same photometry table, data policy, and run-identity
  machinery feed both.
- **Run identity.** Each run directory is named by a hash of the
  resolved configuration, the photometry bytes, and the bandpass
  definitions. A rerun replaces its own directory in place, and a run
  produced by different package versions is reported and requires
  `--force` to overwrite.

## Install

Requires Python 3.11 or newer on Linux, macOS, or Windows.

    pip install -e .                 # core: assembly, quick engine, plots
    pip install -e ".[eazy]"         # + the official eazy-py engine
    pip install -e ".[prospector]"   # + Prospector/FSPS/dynesty/emcee

Ready-made conda environments live in `envs/`. Prospector fitting needs
an FSPS data checkout at `$SPS_HOME`:

    git clone https://github.com/cconroy20/fsps ~/fsps
    export SPS_HOME=~/fsps
    conda env create -f envs/sedfit_miles.yml

FSPS compiles from Fortran source, so the Prospector extra is best
supported on Linux and macOS. The core install and the eazy backends
run anywhere.

### C3K spectral library

The FSPS spectral library is a compile-time choice in python-fsps; the
plain pip wheel builds MILES. For a C3K build, create the second
environment and recompile python-fsps with the C3K flags (see the
python-fsps installation docs for library selection via `FFLAGS`):

    conda env create -f envs/sedfit_c3k.yml
    conda activate sedfit_c3k
    FFLAGS="-DMILES=0 -DC3K=1" pip install --force-reinstall \
        --no-cache-dir --no-binary fsps fsps==0.4.7

Verify with `python -c "import fsps;
print(fsps.StellarPopulation(zcontinuous=1).libraries)"` — the second
entry should be `c3k_a`. Every fit configuration declares its
`stellar_library`, and the fit asserts it against the live FSPS build
at start, so a mismatched environment fails loudly. Results from
different libraries are not interchangeable.

## Usage

    sedfit roster --catalog SAMPLE.csv --campaign CAMPAIGN.json [--out ROSTER]
    sedfit build --roster ROSTER [--target NAME] [--recipe NAME] [--deredden]
    sedfit fit --roster ROSTER --target NAME --recipe NAME --config CFG
    sedfit fit --roster ROSTER --jobs JOBS.json
    sedfit batch --roster ROSTER --config CFG [--targets CSV] [--workers N]
    sedfit run --catalog SAMPLE.csv --campaign CAMPAIGN.json --config CFG
    sedfit plot --run-dir RUN_DIR
    sedfit manifest --roster ROSTER

`sedfit --help` lists the verbs; `sedfit <verb> --help` documents every
option.

- **`roster`** generates a roster from a sample catalog (one row per
  target) and a campaign config (the declarations shared by every
  target), writing `roster.json` beside the campaign plus a
  `.report.csv` naming what each target supplied. The roster is a
  generated artifact — edit the catalog or the campaign, not the roster.
- **`build`** writes `<target.dir>/Photometry/<prefix>_sed_<recipe>.csv`
  plus a provenance sidecar. With `--deredden` it writes the
  Milky-Way-corrected `..._dered.csv` alongside, using a per-target SFD
  lookup unless `--mw-ebv` supplies one value.
- **`fit`** locates that table, cross-checks the sidecar against the
  live band registry, applies the shared data policy (inclusion, S/N
  gate, one clamped error floor), and writes a self-contained run
  directory under
  `<target.dir>/SED/<recipe>/<backend>/<run_id>-<label>/` with figures
  in `plots/` (`--no-plots` opts out), appending one row to the central
  manifest.
- **`batch`** runs every applicable target and recipe under one config
  across a worker pool, resuming on `run_id` so an interrupted campaign
  restarts where it stopped. It writes a per-job report CSV and
  per-job logs, and isolates failures so one bad target does not stop
  the sweep. `--build` rebuilds each table first.
- **`run`** chains `roster`, `build` and `batch` in one call, reporting
  each stage separately.
- **`plot`** re-renders the figures from a run directory's own contents.
- **`manifest`** summarizes the central manifest and reports rows whose
  run directory has gone missing.

## A worked example

`examples/` holds a complete, runnable campaign: three synthetic
galaxies with broadband photometry and SPHEREx spectra, the three input
files annotated field by field, and four recipes covering every
reference frame and every source role.

    cd examples
    sedfit run --catalog sample_catalog.csv --campaign campaign.json \
        --config fit_quick.json

Copy those three files and replace their contents with your own.

## Data model

Three declarations drive everything; `examples/` has a working instance
of each.

- **Registry** (`sedfit/data/registry.json`): every band the package
  knows — one label, one bandpass (a sedpy filter name or a packaged
  curve). Bandpass content is hashed into run identity.
- **Roster** (one JSON per campaign): a `sample` name, the `data_root`
  everything hangs from, an optional `reference_redshift` and
  `manifest_path`, and the targets — positions, reference redshifts,
  per-target directories, an optional SPHEREx visits table, and named
  sources. Each source is a CSV path, its expected bands, and its
  provider. Source CSVs carry four columns (`band`, `flux_uJy`,
  `flux_err_uJy`, `source`); the `source` column's prefix vocabulary
  (see `sedfit/core/sources.py`) ties every row to its provider, and
  rows that fail the declared expectations are hard errors at load
  time.
- **Recipes** (in the roster): how to assemble a fit table — the
  reference frame (SPHEREx, anchor bands, or none), which sources
  stitch, tilt, or float, and the SPHEREx binning.

A target's `z_ref_kind` is `reference` (adopt the sample's
`reference_redshift`), `spec`, or `phot`. The manifest defaults to
`<data_root>/sed_fitting/runs.jsonl`; declare `manifest_path` in the
campaign to put it elsewhere.

Fit configurations are strict JSON (unknown keys are errors) with a
discriminated `backend` field; resolution materializes every default,
the redshift reference, and the seed, and echoes the result into the
run directory.

## Layout

    sedfit/core/       registry, roster, recipe, sources, generate,
                       validate, table, spherex, synth, stitch, build,
                       dered, policy, fitconfig, provenance, runs
                       (light stack only)
    sedfit/backends/   eazy (official engine + the quick workhorse),
                       prospector (dynesty/emcee, seeded, checkpointed)
    sedfit/analysis/   the unified SED figure, line catalogs, EW
    sedfit/jobs.py     fit-job orchestration (the fit verb's engine)
    sedfit/batch.py    the batch verb: worker pool, resume, report
    sedfit/data/       registry.json, filter curves, templates, TEF

## Tests

    python -m pytest tests/

The test suite is synthetic end to end: fixtures are generated (see
`tests/data/make_spherex_fixture.py`) and golden values are pinned
regressions on those fixtures.

## Packaged data

An eazy configuration names its template set in `templates`, either as
a filesystem path or as the bare name of one of the sets below, so a
configuration carries no machine-specific path. **There is no default:
an omitted `templates` is an error.** The basis is the largest
systematic a fit carries — the choice of template set can move the
derived redshifts by more than any other configuration choice — so it
may not arrive silently.

- `data/templates/brown14_vac_cosmos160/` — **the adopted basis**: the
  129 Brown+14 spectra corrected from air to vacuum (Morton 2000/SDSS,
  applied above 2000 Å), plus the 31 COSMOS (Ilbert+09 / Polletta+07)
  templates.
- `data/templates/brown14/` — the Brown et al. (2014, ApJS 212, 18)
  galaxy spectral atlas as distributed.
- `data/templates/brown14_vac/` — the same atlas, air-to-vacuum
  corrected, without the COSMOS additions.
- `data/templates/ssp_miles/`, `data/templates/ssp_c3k_a/` — FSPS
  simple-stellar-population grids on the two spectral libraries.
- `data/TEMPLATE_ERROR.eazy_v1.0` — the eazy template error function
  (Brammer, van Dokkum, & Coppi 2008; distributed with eazy-py).
- `data/filters/` — J-PLUS filter curves, vendored from the SVO Filter
  Profile Service.
- `data/emlines_info.dat` — the FSPS emission-line list.

## License

MIT

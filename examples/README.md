# A worked example

Everything here is synthetic and self-contained. Run it as-is to see what
the outputs look like, then copy the three input files and replace their
contents with your own.

```bash
cd examples
sedfit run --catalog sample_catalog.csv --campaign campaign.json \
    --config fit_quick.json
```

That generates the roster, builds a table per galaxy and recipe, fits all
twelve, and writes `batch_report.csv`. It takes about a minute.

## The three files you write

| File | What it is | How often it changes |
|---|---|---|
| `sample_catalog.csv` | one row per galaxy | every time the sample grows |
| `campaign.json` | which files to expect, how to combine them | rarely |
| `fit_quick.json` | how to fit | once per fitting choice |

Everything else — `roster.json`, the built tables, the run directories —
is generated. Never edit a generated file; change an input and re-run.

### `sample_catalog.csv`

```
name,ra_deg,dec_deg,dir,label,z_ref_kind,z_ref,notes
gal_0001,217.510000,56.990000,gal_0001,,spec,0.1043,red sequence
```

| Column | Required | Meaning |
|---|---|---|
| `name` | yes | the galaxy's name, unique within the file (`target` also accepted) |
| `ra_deg`, `dec_deg` | yes | position in degrees (`ra`/`dec` also accepted) |
| `z_ref_kind` | yes | `spec`, `phot`, or `reference` |
| `z_ref` | see below | the comparison redshift |
| `dir` | no | its directory under `data_root`; defaults to `name` |
| `label` | no | the filename stem; defaults to a sanitized `name` |

`z_ref_kind` says where the comparison redshift comes from. Use `spec` or
`phot` when the galaxy has its own measured value and put it in `z_ref`.
Use `reference` when it should adopt the sample-wide
`reference_redshift` from the campaign config — and then **leave `z_ref`
blank**, because the roster derives it and refuses a target that states
it twice.

The `notes` column is not one sedfit recognizes. It is carried along and
reported rather than refused, so you can keep your own bookkeeping in the
same file. The `roster` verb prints which columns it used and which it
ignored, which is how you catch a typo in an optional column name.

### `campaign.json`

Everything shared across galaxies. Adding a galaxy to the catalog cannot
change what is fit or how — that separation is the point of having two
files.

**Top level**

| Key | Required | Meaning |
|---|---|---|
| `schema_version` | yes | `1` |
| `sample` | yes | a name for this set of targets |
| `data_root` | yes | where the galaxy directories live, relative to this file |
| `position_authority` | yes | free text: where the positions came from |
| `sources` | yes | the photometry files to expect |
| `recipes` | yes | how to combine them |
| `reference_redshift` | only if some target uses `z_ref_kind: reference` | the sample-wide redshift |
| `manifest_path` | no | the central run log, relative to `data_root`. Default `sed_fitting/runs.jsonl` |
| `spherex` | no | the per-visit spectrophotometry table: `{table, model, provenance}`, all three required if present. `table` is a glob pattern; `model` is `psf` or `sersic` |
| `description` | no | free text, copied into the roster |
| `position_frame` | no | the astrometric frame. Default and only legal value `icrs` |

`reference_redshift` becomes required the moment any catalog row uses
`z_ref_kind: reference` — and **generation does not check it**. The
roster writes cleanly and the *next* verb fails with `z_ref_kind
'reference' requires the roster to declare reference_redshift`. Several
enums behave the same way (a source's `kind`, `spherex.model`,
`position_frame`): they are validated when the roster loads, not when
the campaign does.

**A source** is one photometry file plus the bands you expect from it:

```json
"legacy": {
  "path": "Photometry/{label}_catalog.csv",
  "bands": ["Legacy_g", "Legacy_r", "Legacy_z"],
  "kind": "catalog",
  "provider": "legacy"
}
```

`path` is relative to **the galaxy's own directory**, not to `data_root`.
`{name}`, `{label}` and `{prefix}` are substituted per galaxy. You may
give a list of paths instead of one string, and they are tried in order —
the first that actually supplies a declared band wins.

`provider` is the important one. One file can hold rows from several
archives, and sedfit tells them apart by the start of each row's `source`
column: `legacy` matches `Legacy_`, `sdss` matches `SDSS_`, `unwise`
matches `unWISE_Legacy_`, and so on. That is how the three sources here
all read the same CSV and each take different rows. The full vocabulary
is in `sedfit/core/sources.py`.

`kind` is `catalog` (published archive photometry) or `measured` (your
own aperture photometry). It is recorded in the provenance sidecar.

**A recipe** is one way to combine those sources into a table. The four
here cover every combination the package offers:

| Recipe | `reference` | What it demonstrates |
|---|---|---|
| `broadband_only` | `none` | no common scale; every source at its measured level |
| `spherex_only` | `spherex` | the spectrum alone |
| `stitched` | `spherex` | the spectrum is the color truth; overlapping instruments are rescaled onto it |
| `tilted` | `anchors` | broadbands are the color truth; the spectrum is tilted onto them |

Each source in a recipe takes a **role**:

- **`anchor`** — it defines the color. Legal only under
  `reference: anchors`, which needs one or two of them. One anchor
  rescales the spectrum achromatically; two tilt it.
- **`stitch`** — its whole instrument group is multiplied by one factor
  measured in the reddest band it shares with the reference. Legal under
  `spherex` and `anchors`.
- **`float`** — carried through untouched. Legal everywhere, and the only
  role legal under `reference: none`.

A recipe also takes an optional `description` and `min_coverage`
(default 0.98, in (0, 1]) — the fraction of a band the spectrum must
cover before that band can carry a stitch scale. A source entry may
carry an optional `bands` subset restricting which of that source's
declared bands this recipe consumes.

A source can only stitch if it actually overlaps the reference. SDSS
floats in `stitched` and `tilted` because SDSS *ugi* lies blueward of
SPHEREx's 0.75 µm blue end — a stitch there would have nothing to measure
against, and the build says so rather than guessing.

### `fit_quick.json`

```json
{
  "schema_version": 2,
  "backend": "eazy",
  "name": "quick_example",
  "err_floor": 0.05,
  "eazy": { "engine": "quick", "z_min": 0.01, "z_max": 0.60,
            "z_step": 0.001, "templates": "brown14_vac_cosmos160",
            "template_pattern": "*.dat" }
}
```

(The real file also sets `min_snr_broadband`, `min_valid_bands`, `tef`,
`tef_scale` and `z_step_type` explicitly, all at their defaults — read
it alongside this.)

| Key | Default | Meaning |
|---|---|---|
| `schema_version` | — | **required**; `2` |
| `backend` | — | **required**; `eazy` or `prospector` |
| `name` | — | **required**; names the run directory and the figures |
| `err_floor` | 0.05 | fractional error floor, added in quadrature. May be a per-instrument map with an optional `"default"` key |
| `min_snr_broadband` | 2.0 | drop broadbands below this signal-to-noise |
| `min_valid_bands` | 5 | fail the fit below this many surviving bands |
| `mu_lensing` | 1.0 | divide fluxes by this magnification |
| `bands_include` | `null` (every band) | instrument names and/or band names; an entry matching nothing is an error |
| `z_ref` | `null` | filled from the roster target; an explicit value that disagrees is an error |
| `qa_gates` | `null` | map of `qa_flags` token → `{min, max}` |

Under `eazy`: `engine` is `quick` (no eazy-py needed) or `eazy-py`;
`z_min`/`z_max`/`z_step` set the redshift grid; `templates` names the
basis; `template_pattern` picks which files in it are spectra.

**The grid defaults are narrow and sample-specific**: `z_min` 0.05,
`z_max` 0.16, `z_step` 0.001. A config that omits them fits that window
and says nothing about it, so set them for your own sample as
`fit_quick.json` does. The other `eazy` defaults: `mode` `combo`
(`single` is what writes `singles.csv`), `z_step_type` `linear`,
`z_fixed` `null`, `tef` `true`, `tef_file` `null`, `tef_scale` `1.0`,
`tef_lnp` `true`, `prior` `false`, `fitter` `nnls`, `n_proc` `4`,
`extra_params` `{}`, `save_zcoeffs` `false`. Under `engine: quick`, a
`prior`, a non-`nnls` `fitter` and any `extra_params` are errors that
name `engine: eazy-py` as the remedy.

`template_pattern` defaults to `*_spec.dat`, which is the naming the
Brown+14 spectra use. The adopted set adds 31 COSMOS templates named
differently, so **fitting the whole 160-template basis needs
`"template_pattern": "*.dat"`** — as in `fit_quick.json` here. A pattern
that selects only part of a set raises a warning naming the count.

**`templates` has no default and must be named.** The choice of template
set moves the derived redshifts more than any other setting, so it is
never implicit. Name one of the sets in `sedfit/data/templates/` or give
a path to your own directory of two-column spectra.

`fit_prospector.json` is the equivalent for the Prospector backend, which
needs the `prospector` extra and an FSPS checkout at `$SPS_HOME`.

The `prospector` block has 31 keys and only one required
(`stellar_library`), but its shape is conditional, so the minimal config
that implies does **not** load. `fit_redshift` defaults `true`, and
`true` requires a `zred` block — start from `fit_prospector.json` rather
than from scratch. The other three conditionals: `nebular` off requires
the three `gas_logu_*` keys to stay null; `sfh: continuity` requires the
parametric keys (`tau_*`, `tage_*`, `tie_tage_to_tuniv`) to stay null
and a parametric `sfh` inverts that; and whichever of `dynesty` /
`emcee` the `sampler` does not select must be null. Full key list and
defaults are in `docs/technical_manual.md` §4.3.

## What the example produces

```
examples/
    roster.json                    generated; do not edit
    roster.json.report.csv         why each source was kept or dropped
    batch_report.csv               one row per job
    batch_logs/                    per-job stdout
    data/
        gal_0001/
            Photometry/
                gal_0001_catalog.csv          the input
                gal_0001_sed_tilted.csv       the built table
                gal_0001_sed_tilted.provenance.json
                SPHEREx/table_photometry.csv  the input spectrum
            SED/tilted/eazy/<run_id>-quick_example/
                config.json  phot.csv  phot.provenance.json
                manifest.json  summary.csv  arrays.npz
                catalog.csv  template_error.dat  templates.param
                engine.info  plots/
        sed_fitting/runs.jsonl     every run, appended
```

## About the numbers

The photometry was synthesized from a packaged Brown+14 template at a
known redshift, with 3% noise and a small deliberate SDSS calibration
offset for the stitch recipes to remove. The fitted redshifts land near
the input ones but not on them — with eight broadbands and a 5% error
floor, that scatter is what a photometric redshift looks like. Compare
`zred_p50` in `batch_report.csv` against `z_ref` in `sample_catalog.csv`.

The SPHEREx table is a coarse stand-in, not a model of the instrument.
Real SPHEREx covers 0.75–5.0 µm continuously in 102 channels across six
LVF bands, each channel's passband abutting its neighbors'. This one
carries 21 channels on a linear grid, spaced about 2.5 times their own
widths, and holds nothing between 1.13 and 2.7 µm. The two segments
straddle the recipes' `split_um` of 2.6, so both binning regimes are
exercised. The gap is how the file was written, not a failed extraction.

Channel labels are positional. `SPHEREx_NNN` numbers the rows of the
spectrum it came from, so one label is different wavelengths across
targets that deliver different channels: `SPHEREx_006` is 1.13 µm for
`gal_0001` and 2.7 µm for `gal_0003`, which is short one blue channel.
Filters are built per row from `wave_um`. Never key on the label.

Nothing here is real data. Do not use it for anything but learning the
file formats.

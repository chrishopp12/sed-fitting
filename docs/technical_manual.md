# sedfit Technical Manual

For a reader who will extend, debug, or audit the package. It assumes
astronomy, not this codebase. Paths are repository-relative.

**Contents**

1. [Overview](#1-overview)
2. [Glossary](#2-glossary)
3. [Architecture](#3-architecture)
4. [Module reference](#4-module-reference)
5. [The output contract](#5-the-output-contract)
6. [Extension points](#6-extension-points)
7. [Behavior worth knowing](#7-behavior-worth-knowing)
8. [Testing](#8-testing)

---

## 1. Overview

`sedfit` turns per-target photometry files into fitted redshifts. It has
two halves joined by one table format:

- **Assembly** (`sedfit/core/`) — combine a target's source CSVs and its
  SPHEREx visit table into one fit-ready SED table, on one flux scale,
  with a provenance sidecar.
- **Fitting** (`sedfit/backends/`) — hand that table to eazy-py or to
  Prospector through a shared data policy, and write a self-contained
  run directory.

Everything is a subcommand of one CLI (`sedfit/__main__.py`):

| Verb | Driver | Produces |
|---|---|---|
| `roster` | `core.generate.generate_roster` + `write_roster` | `roster.json` + `roster.json.report.csv` |
| `build` | `core.build.build_target` + `write_build` | `<prefix>_sed_<recipe>.csv` + sidecar |
| `fit` | `jobs.run_job` (`run_jobs` under `--jobs`) | one run directory + one manifest row per job |
| `batch` | `batch.run_batch` | many run directories + a report CSV |
| `run` | `__main__._run` | `roster`, then `batch --build` (each job builds its own table) |
| `plot` | backend `generate_plots` | re-rendered figures |
| `manifest` | `core.runs.read_manifest` | a manifest summary |

Three declarations drive everything, and each is validated strictly —
unknown keys are errors, not warnings:

- the **band registry**, packaged with the code, naming every band the
  package knows and where its bandpass comes from;
- the **roster**, one per campaign, naming the targets and their files;
- a **fit configuration**, naming the backend and its settings.

Fluxes are microjansky throughout, AB. Errors are statistical as they
arrive; the fitting-time error floor is applied once, in one place.

**Exit status.** `batch` exits 1 if any job failed or the batch aborted
on `--stop-after-failures`; `manifest` exits 1 if the central log has a
torn line or any run directory is missing; `run` exits 1 if either stage
failed. `roster`, `build`, `fit` and `plot` exit 0 unless they raise.

The load-bearing idea is **run identity**. A run directory is named by a
hash over the resolved configuration, the photometry bytes, and the
bandpass definitions. The same science inputs always produce the same
`run_id`, on any machine and under any directory layout, so a rerun
replaces its own directory rather than accumulating siblings.

---

## 2. Glossary

Read this before the deep sections; several words are overloaded.

### The declaration layer

| Term | Owner | Meaning |
|---|---|---|
| **sample catalog** | `core/generate.py` | The hand-edited CSV, one row per target: name, position, redshift, directory. The only file a campaign operator edits when adding galaxies. |
| **campaign config** | `core/generate.py` | The JSON holding every policy choice shared across targets: which sources to expect, which bands, which recipes. Adding a galaxy cannot change it. |
| **roster** | `core/roster.py` | The **generated** join of the two, resolved against the tree. Never hand-edited: regenerate it instead. |
| **generation report** | `core/generate.py` | `roster.json.report.csv`, one row per (target, source) with status `kept` / `partial` / `dropped` and the reason. The artifact to read after generating. |
| **target** | `core/roster.py` | One galaxy: a position, a reference redshift, a directory, an optional SPHEREx table, and named sources. |
| **source** | `core/roster.py` | One declared photometry file for one target, plus the bands it is expected to supply, its `kind` (`catalog` or `measured`), and the provider whose prefix its rows must carry. |
| **provider / source prefix** | `core/sources.py` | A provider token (`legacy`, `jplus`, `aperture`, …) maps to a string prefix that every one of its rows carries in the *source CSV's* `source` column. This is what lets two providers supply the same band from one file. |
| **recipe** | `core/recipe.py` | How to assemble one SED table: a reference frame, a role for each source, and the SPHEREx handling. |

### Assembly

| Term | Owner | Meaning |
|---|---|---|
| **reference frame** | `core/recipe.py` | What defines the flux scale and color. One of `spherex` (the raw spectrum is truth), `anchors` (one or two named broadband groups are truth), or `none` (no common scale; everything floats). |
| **role** | `core/recipe.py` | What a source does in a recipe. `anchor` — it defines the reference. `stitch` — its instrument group is rescaled onto the reference. `float` — its fluxes are carried through untouched. |
| **scale** | `core/stitch.py` | One multiplicative factor per instrument group, measured as synthetic-over-measured flux in the reddest eligible overlapping band, applied to the whole group. |
| **tilt** | `core/build.py` | The two-anchor case. With anchor pivots `lam_b < lam_r` and correction factors `c = 1/scale`, the spectrum is multiplied by `t(lam) = c_b * (c_r/c_b) ** (ln(lam/lam_b) / ln(lam_r/lam_b))`, a log-linear interpolation in log wavelength, extrapolated by the same power law. A **rescale** is the degenerate one-anchor case: achromatic, no color rotation. |
| **channel** | `core/spherex.py` | One binned SPHEREx wavelength bin, with its own center, width, and exposure count. Channels are per object: their bandpasses are built on the fly, not looked up. |
| **uniform error convention** | `core/build.py` | Any flux transformation multiplies flux, error, and — where the row has one — scatter by the same factor. Broadband rows carry no scatter, so only the SPHEREx transforms touch all three. |

### Fitting

| Term | Owner | Meaning |
|---|---|---|
| **data policy** | `core/policy.py` | The single implementation of band inclusion, the signal-to-noise gate, QA-flag gating, lensing magnification, and the error floor. Both backends receive its output; neither reimplements any of it. |
| **fittable** | `core/policy.py` | A band that survived inclusion, the S/N gate, and the QA gates. Excluded rows keep transformed values, and each backend substitutes its own missing representation. |
| **error floor** | `core/policy.py` | A fractional floor added in quadrature to the statistical error, with the flux clamped at zero so a negative measurement contributes no floor. Applied exactly once. |
| **run identity (`run_id`)** | `core/provenance.py` | A hash over the resolved config (minus execution-only fields), the photometry bytes, and the bandpass definitions. |
| **execution-only field** | `core/fitconfig.py` | A config key that says *where* something lives or *how fast* to compute it, not *what* is computed: `name`, `n_proc`, `templates`, `template_pattern`, `tef_file`. Stripped before hashing. |
| **content digest** | `backends/eazy/templates.py` | A per-file hash of every resolved template spectrum and the template-error curve, keyed by basename. This is what puts the template *content* into `run_id` while the *path* stays out. |
| **machinery stamp** | `core/runs.py` | Package version, git revision, FSPS libraries, and dependency versions. Compared when replacing an existing run; **not** part of `run_id`. |
| **engine** | `backends/eazy/` | Which eazy implementation runs: `quick` (vectorized, in this package) or `eazy-py` (the official code). Both consume the same policy output and write the same *results* products, so `results.load_run` reads either directory; the engine-specific files differ (§5.3). |

### Ambiguous words, resolved

| Word | Senses |
|---|---|
| **reference** | (1) A recipe's **reference frame**. (2) The sample's **`reference_redshift`**, adopted by targets of `z_ref_kind: reference`. (3) The **anchor band** inside `stitch.measure_scale` (`info['anchor']`), the band a group's scale is measured in — a different sense of *anchor* from the recipe role. |
| **source** | (1) A roster **source** — one declared file. (2) The **source CSV's `source` column**, a provenance string carrying the provider prefix; it is consumed at build time and does **not** survive into the built SED table (§5.1), which records it in the sidecar's `source_strings` instead. (3) The `sedphot` `source` prefix vocabulary. |
| **scale** | An instrument group's achromatic factor (`stitch`), never the tilt. |
| **prefix** | (1) A target's **filename stem** (`Target.prefix`). (2) A provider's **source-column prefix** (`sources.SOURCE_PREFIXES`). Unrelated. |
| **manifest** | (1) The **central** append-only log at the roster's `manifest_path` — relative to `data_root`, default `sed_fitting/runs.jsonl`. (2) The **per-run** `manifest.json`, one row, the same schema. |

---

## 3. Architecture

### 3.1 A build, end to end

```
CLI: sedfit build --roster ROSTER --target X --recipe R
  |
  __main__._build -> core.roster.load_roster        <- opens EVERY declared
  |                                                    source file and checks
  |                                                    its declarations
  +-- core.build.check_applicability(recipe, target)
  |     recipe's sources all declared by the target; SPHEREx agreement
  |
  +-- core.build.build_target(roster, target_name, recipe_name, *,
  |                            registry=..., mw_ebv=...)
        |
        1. sources.select_rows        per recipe entry: the declared bands,
        |                             selected by provider prefix, exactly
        |                             one row each
        1b. sources.check_position    once per distinct file: the table's
        |                             target_ra/target_dec against the roster
        |                             position; a table carrying neither
        |                             column warns rather than passing
        1c. build._check_dered_states consumed rows must share one extinction
        |                             state, and a dereddening build refuses
        |                             rows that are already dereddened
        2. spherex.ingest             visit table -> quality cuts -> binning
        |                             -> a channel spectrum + counts
        3. dered.deredden             (only when mw_ebv is not None) per
        |                             source, in its native frame, before
        |                             any transform
        4. reference resolution       reference == "spherex": the raw
        |                             spectrum is truth
        |                             reference == "anchors": measure each
        |                             anchor group's scale, then rescale
        |                             (one anchor) or tilt (two)
        5. stitch.measure_scale       per stitch-role instrument group:
        |                             synth/measured in the reddest
        |                             eligible overlapping band
        6. concatenate                broadband parts + the spectrum
        7. sidecar                    every input hashed, every choice recorded
        |
        -> BuildResult(frame, sidecar, out_path)     nothing written yet
  |
  +-- core.build.write_build(result, registry)
        validates the frame against the registry, then writes the CSV and
        the .provenance.json beside it
```

`build_target` returns before writing so a caller can inspect or discard
the result; `--dry-run` stops after `check_applicability`.

### 3.2 A fit, end to end

```
CLI: sedfit fit --roster ROSTER --target X --recipe R --config CFG
  |
  fitconfig.load_fit_config       <- the CLI or the batch worker parses
  |                                  the config; unknown keys are errors
  |                                  and every default is materialized here
  |
  jobs.plan_job(...)              <- everything that can fail without
  |                                  writing anything
  |   fitconfig.resolve_config    fill the roster-derived fields: z_ref, a
  |                               null normal zred.mean, the sampler seed
  |   phot_path_for               locate the built table
  |   table.validate_sed_table    schema + registry check
  |   _check_sidecar              recipe, z_ref, per-band bandpass hashes
  |   policy.apply_policy         inclusion, magnification, the broadband
  |                               S/N gate, the QA gates, the error floor
  |   templates.content_digests   hash every template + the TEF curve
  |   provenance.run_id           hash the projection -> run_id
  |
  jobs.run_job(...)
  |   (prospector only) build the SPS and assert the stellar library
  |                               BEFORE staging, so a library mismatch
  |                               leaves no directory and no manifest row
  |   runs.stage_run              build the directory under a temp name,
  |   |                           with config.json, phot.csv, the sidecar
  |   |                           copy and an initial manifest stamp, then
  |   |                           os.replace it into place atomically
  |   +-- backend dispatch
  |         eazy       -> quick.run_quick | fitting.run_official
  |         prospector -> obs.build_obs + model.build_model
  |                       + attach_norm_masks -> fitting.run_sampler
  |                       -> save_results          (the SPS was built above)
  |   +-- auto-plots              eazy: plots.generate_plots
  |                               prospector: plots.plot_run
  |                               failures here are reported, not fatal
  |   runs.finalize_run           write manifest.json, then append one
  |                               locked line to the central runs.jsonl
```

The staging-then-rename step is what makes a crashed fit leave a
self-describing directory rather than a half-written one: the directory
that appears is already complete except for the results.

### 3.3 A batch

`batch.run_batch` expands the target × recipe grid against
`check_applicability`, then dispatches to a `spawn` process pool whose
workers each load the registry and roster exactly once
(`batch._init_worker`). With `--workers 1` there is no pool at all and
the jobs run in the calling process, which is how a traceback survives
intact — that is the setting to reach for when debugging.

Every outcome is recorded rather than raised, so one target that fails
does not stop the sweep, and the run ends with a sorted per-job report.
The exception is `--stop-after-failures` (default 20): once that many
jobs have failed the batch cancels its pending work and reports
`aborted`.

Resume is by `run_id`. `_completed_run` looks for an existing directory
with that id whose own `manifest.json` says `status: ok`, and skips the
job if it finds one — **before** any staging, so a resumed batch never
touches a finished run.

`--mw-ebv` is legal only with both `--build` and `--deredden`, checked
before any job runs. When both are set, `run_batch` resolves `E(B-V)`
once per *target* in the parent process — not per job, and not in the
workers — so every recipe a target runs under gets the same value and
the SFD service sees one query per galaxy.

### 3.4 Tracing a flux to a fitted redshift

1. A row in one of the target's declared source CSVs — the path comes
   from the roster and resolves against `target.dir` — selected by
   `sources.select_rows` because its `source` column starts with the
   declared provider's prefix.
2. Possibly multiplied by a dereddening factor (`dered.band_factor`).
3. Possibly multiplied by its instrument group's scale
   (`stitch.measure_scale`), or the spectrum tilted onto the anchors.
4. Written to `<target.dir>/Photometry/<prefix>_sed_<recipe>.csv`
   (`build.SED_TABLE_SUBDIR`) as one row of `band`, `flux_uJy`,
   `flux_err_uJy`.
5. Copied byte-for-byte into the run directory as `phot.csv`.
6. Divided by `mu_lensing`, then the error floor added in quadrature
   (`policy.apply_policy`).
7. Integrated through its bandpass by the engine, and compared with the
   template grid.

Steps 1–4 are recorded in the build sidecar; steps 5–7 in the run
directory's `config.json` and `manifest.json`.

---

## 4. Module reference

### 4.1 The declaration layer (`sedfit/core/`)

#### `registry.py`
The band-identity authority. Loads `sedfit/data/registry.json`: an
`instruments` list, a `spherex` block giving the channel-name prefix, and
a `bands` map where each entry names its `instrument` and exactly one of
`sedpy` (a filter name resolved through sedpy) or `curve` (a two-column
file path relative to the registry).

`get_bandpass(band)` returns `(wave_AA, throughput)`. It **refuses**
SPHEREx channel names — those have per-object geometry and are built by
`core.synth`. `bandpass_hash()` is one digest over the registry file
*and* every band's resolved bandpass arrays, and it enters `run_id`:
edit a filter curve or the registry itself and every run identity
changes.

#### `roster.py`
Loads, resolves and validates a roster. Resolution is positional:
`data_root` is relative to the roster file, target directories to
`data_root`, and source paths and SPHEREx tables to the target
directory. Loading is not cheap by design — it **opens every declared
source file** and requires each declared band to select exactly one row
under its provider prefix, so a roster that loads is one whose
declarations are true of the tree right now.

`cluster`, `cluster_redshift`, and `z_ref_kind: cluster` are accepted as
deprecated spellings of `sample`, `reference_redshift`, and `reference`.
Declaring both spellings of one field is an error.

`prefix` is a **required** target key — it is the filename stem every
built table and sidecar is named from. `manifest_path` must be relative;
an absolute path is a hard error. `aperture_arcsec` is accepted and
parsed onto `Target`, but nothing in the package reads it.

#### `generate.py`
The `roster` verb. Joins a sample catalog with a campaign config and
resolves both against the tree. Its rule is that generation declares only
what the tree actually supplies: a source whose file is absent is
omitted, and a declared band no row supplies is dropped from that source
rather than written and left to fail later.

A source may declare several filename templates (`{name}`, `{label}`,
`{prefix}` are substituted). They are tried in order, and **a candidate
wins by supplying at least one declared band**, not merely by existing —
so a tree carrying both a combined catalog and per-provider files
resolves to whichever actually holds the bands.

The SPHEREx table is matched by **glob pattern** rather than an exact
name, and a pattern matching more than one table is a hard error rather
than a pick. It is the one place generation refuses to guess.

Campaign keys. Required: `schema_version`, `sample`, `data_root`,
`position_authority`, `sources`, `recipes`. Optional: `description`,
`reference_redshift`, `manifest_path`, `position_frame` (default and
only legal value `icrs`), `spherex`. `cluster` and `cluster_redshift`
are accepted as deprecated spellings of `sample` and
`reference_redshift`.

**`reference_redshift` becomes required the moment any catalog row uses
`z_ref_kind: reference`, and generation does not check it** — the roster
writes cleanly and the *next* verb raises `z_ref_kind 'reference'
requires the roster to declare reference_redshift`. The same deferral
applies to several enums: `load_campaign` does not validate a source's
`kind`, the campaign `spherex.model`, or `position_frame`, all of which
are checked at roster load instead.

**Two different `spherex` blocks.** The campaign's top-level one is
`{table, model, provenance}`, all three required, where `table` is a
glob pattern (or list) with `{name}`/`{label}`/`{prefix}` substitution
and `model` is `psf` or `sersic`. A *recipe's* `spherex` is the
tri-state `{cuts, binning}` block described under `recipe.py`. They
share a name and nothing else.

#### `recipe.py`
Parses and validates one recipe. Checks the frame model: `anchor` roles
are legal only under `reference: anchors`, which requires one or two of
them; `stitch` is legal under `spherex` and `anchors`; `float`
everywhere. No band may be *listed* by two source entries; the same rule
on implicitly-consumed bands is enforced at build time by
`build.effective_bands`.

The `spherex` block is tri-state: **absent** (the build hard-errors when
the target declares a SPHEREx table), **null** (deliberate exclusion), or
a block of cuts and binning — `cuts` being `fit_ql_max` (null),
`flags_reject` (`DEFAULT_FLAGS_REJECT`) and `drop_local_bkg` (false),
and `binning` being `split_um` (null), `blue_merge_dlam_um` (0.001) and
`red_bin_resolution` (1.0).

Recipe keys. Required: `reference`, `sources`. Optional: `description`
and `min_coverage` (default 0.98, in (0, 1]). A source entry takes
`source` and `role`, plus an optional `bands` subset restricting which
of that source's declared bands this recipe consumes.

#### `sources.py`
The provider-prefix vocabulary and the row selection built on it.
`SOURCE_PREFIXES` maps ten provider tokens to the string prefixes that
appear verbatim in data files. The set must stay **prefix-free** — no
entry may be a prefix of another — so `startswith` matching is
unambiguous.

`select_rows` is the error-raising selection used at build time;
`match_counts` is the counting version used at roster load. `check_position`
cross-checks a table's `target_ra`/`target_dec` against the roster
position within 0.5", and a table carrying neither column raises a
`UserWarning` rather than passing silently.

`REQUIRED_COLUMNS` is `band`, `flux_uJy`, `flux_err_uJy`, `source`.
Three further columns are consumed when present: `target_ra`/`target_dec`
for the position check, and `flags` — on `aperture` and `sersic`
provider rows only — which becomes the SED table's `qa_flags` and is
what `qa_gates` gates on. Flux and error validity is enforced on
**selected rows only**, so a malformed row nothing selects is inert.

#### `validate.py`
`require_keys`, `require_enum`, `apply_aliases`, `describe_columns`. The
JSON loaders refuse unknown keys; the CSV readers do not, because one
sample catalog is shared with other tools — `describe_columns` reports
the split instead, which is what makes a typo in an optional column
visible next to the default it silently took.

### 4.2 Assembly (`sedfit/core/`)

#### `spherex.py`
Ingests an IRSA per-visit spectrophotometry table. `load_visits` reads
and requires the mandatory columns; `apply_quality_cuts` applies the
flag mask and the optional `fit_ql` and `local_bkg` cuts; `bin_spectrum`
merges surviving visits into channels.

`DEFAULT_FLAGS_REJECT` rejects a visit if any bit below 40 other than
`SOURCE` is set, documented or not — `SOURCE` is set on every
forced-photometry row by construction, so it must be exempt.

Two behaviors matter downstream and are easy to miss: a **single-visit
bin reports its statistical error as its scatter**, and `bandwidth_um`
is the larger of the channel width and the bin's actual wavelength span.

#### `synth.py`
The synthetic-photometry primitives: a photon-counting AB band average
of a spectrum through a bandpass, the pivot wavelength, and the SPHEREx
tophat generator. Used by the stitch scales, both backends' SPHEREx
channels, and the plot positions.

`synth_fnu` returns a `coverage` fraction alongside the flux. Samples
the spectrum does not cover are dropped before integrating, so an
**interior** gap is bridged by the trapezoid rule and counts as covered
— coverage measures the ends, not holes.

`SPHEREX_TOPHAT_SAMPLES` (101) is the **quadrature resolution**, not
merely the bandpass shape. The Prospector backend's exact filters
integrate the source *on* the filter grid, so an under-resolved tophat
aliases the ~1 Å FSPS grid: at 25 samples the two bluest channels' model
fluxes ran +0.2% high. It is a module constant, not a config key, and it
is **not** in `run_id` — so the manifest row records the value that ran
(§5.4).

#### `stitch.py`
`measure_scale` returns one instrument group's achromatic factor. A band
is eligible if it clears `min_coverage` — a per-recipe key, default 0.98
— has a finite synthetic flux, **and** has a positive measured flux; the
reddest eligible band wins. Returns `None` when no band is covered (the
group's level floats at fit time), and raises when bands are covered but
none has positive flux.

#### `build.py`
The assembly pipeline (§3.1) and the sidecar. `check_applicability` is
the cheap pre-flight; `build_target` does the work and returns a
`BuildResult` without writing; `write_build` validates and writes.

`SED_TABLE_SUBDIR` is the one definition of where a target's tables live;
`jobs.phot_path_for` resolves the same path for the fit verb.

#### `dered.py`
Foreground Milky Way extinction, applied per source in its native frame
**before** any reference or stitch transform. The curve is three
branches in `x = 1/lambda [1/micron]`: the CCM (1989) near-IR power law
below `x = 1.1`, the O'Donnell (1994) optical polynomial to `x = 3.3`,
and the CCM (1989) ultraviolet form to `x = 8`, above which it raises.
Broadband corrections are photon-counting bandpass integrals matching
`synth`; SPHEREx channels are corrected at their center wavelength.

`fetch_ebv_sfd` pulls the Schlafly & Finkbeiner recalibrated `E(B-V)`
from IRSA, which is what the curve above expects.

#### `table.py`
The SED table schema (§5.1) and its validator. Validation runs on every
read and is the authoritative unknown-band and error check.

### 4.3 Fitting and orchestration (`sedfit/core/`, `sedfit/jobs.py`, `sedfit/batch.py`, `sedfit/__main__.py`)

#### `policy.py`
The one implementation of everything between the table and a backend.
In order: inclusion by `bands_include`, lensing magnification, the
broadband signal-to-noise gate (computed on the magnification-divided
*statistical* errors, before the floor, and SPHEREx channels are
exempt), the QA-flag gates, then the clamped error floor. Magnification
and the floor commute.

`PolicyResult` carries the transformed vectors at full table length plus
the complete accounting — which bands were excluded and why — and
`accounting()` projects that into the manifest row. `floor_by_band`
reports the floor that was actually applied.

`min_valid_bands` (default 5) is the one policy setting that aborts a
fit outright: fewer surviving bands than that and `apply_policy`
raises.

#### `spectrum.py`
The observed-spectrum input for joint fits: `read_spectrum` enforces the
file contract (§6.6) and reads the bytes exactly once, returning a
`Spectrum` whose sha256 enters the run identity;
`apply_spectrum_policy` mirrors `policy.py` for the spectrum —
magnification, the clamped fractional floor, and the config's
`mask_windows` on top of the file's own mask. Light-stack: no
`prospect` import, so the layering test covers it.

#### `fitconfig.py`
Strict parsing, defaulting, and resolution of a fit configuration, plus
the hash projection. `parse_fit_config` refuses unknown keys, validates every vocabulary,
and materializes every default; `resolve_config` fills only the
roster-derived fields — `z_ref`, a null normal `zred.mean` centered on
the sample's `reference_redshift`, and a null sampler seed;
`hash_projection` strips the execution-only fields and merges the
content digests.

An eazy config **must** name its `templates`. There is no default,
because the template basis is the dominant systematic a fit carries.

**Top level.** Required: `schema_version` (`2`), `backend`, `name`.
Optional, with defaults: `bands_include` `null` (accepts registry
*instrument* names as well as band names; an entry matching no band in
the table is a hard error), `min_valid_bands` `5`, `min_snr_broadband`
`2.0`, `err_floor` `0.05` (a scalar, or a per-instrument map with an
optional `"default"` key), `mu_lensing` `1.0`, `z_ref` `null` (filled
from the roster target; an explicit value that disagrees is a hard
error), `qa_gates` `null` (a map of `qa_flags` token → `{min, max}`,
tokens restricted to `policy.QA_GATE_TOKENS`).

**The `eazy` block.** `templates` is required. The rest default:
`engine` `quick`, `mode` `combo` (`single` is what writes
`singles.csv`), **`z_min` `0.05`, `z_max` `0.16`**, `z_step` `0.001`,
`z_step_type` `linear`, `z_fixed` `null` (must lie strictly inside the
grid), `template_pattern` `*_spec.dat`, `tef` `true`, `tef_file` `null`,
`tef_scale` `1.0`, `tef_lnp` `true`, `prior` `false`, `prior_file`
`null`, `prior_filter` `null`, `fitter` `nnls`, `n_proc` `4`,
`extra_params` `{}`, `save_zcoeffs` `false`. Under `engine: quick`, a
`prior`, a non-`nnls` `fitter` and any `extra_params` are hard errors
naming `engine: eazy-py` as the remedy.

The redshift-grid defaults are a **narrow, sample-specific window**. A
config that omits them fits 0.05–0.16 and says nothing about it; set
them deliberately for any other sample.

**The `prospector` block.** `stellar_library` is the only required key,
but the shape is conditional in four places, and the minimal config that
suggests — `{"stellar_library": "miles"}` — does **not** load:
`fit_redshift` defaults `true`, and `true` requires a `zred` block
(`prior` `uniform`|`normal`, `mean`, `sigma`, `bounds` `[0.0, 1.0]`; a
`normal` prior needs a positive `sigma`, and a null `mean` is centered
on the roster's `reference_redshift` at resolve time). The other three
conditionals: `nebular` off requires `gas_logu_free`/`gas_logu_prior`/
`gas_logu_init` to stay null and on defaults them to `false` /
`[-4.0, -1.0]` / `-2.0`; `sfh: continuity` defaults `n_agebins` to `7`
and requires the parametric keys (`tie_tage_to_tuniv`, `tau_range`,
`tau_init`, `tage_range`, `tage_init`, `tage_tuniv_init`) to stay null,
while a parametric `sfh` inverts that; and the unselected `sampler`
block of `dynesty` / `emcee` must be null.

The remaining defaults: `sfh` `continuity`, `nebular` `false`, `agn`
`false`, `dust_emission` `false`, `free_norm_instruments` `[]`,
`free_norm_prior` `[0.1, 10.0]`, `exact_filters` `true`, `sampler`
`dynesty`, `dynesty` `{}`, `seed` `null`, `mass_range`
`[1.0e10, 1.0e13]`, `mass_init` `3.0e11`, `logzsol_prior`
`[0.0, 0.3, -1.0, 0.5]`, `logzsol_prior_type` `clipped_normal`
(`uniform` takes two values, `clipped_normal` four), `logzsol_init`
`0.0`, `dust2_prior` `[0.15, 0.2, 0.0, 1.0]`, `dust2_prior_type`
`clipped_normal`, `dust2_init` `0.15`.

A null `seed` draws 31 bits from `SystemRandom` at resolve time and
records the draw — which is why an unseeded dry run reports a `run_id`
the real run will not use (§7).

#### `provenance.py`
`canonical_json` and `run_id`, plus file and byte digests and
`git_state`. The canonical form is version-stable: changing it changes
every run identity. `git_state`'s `dirty` comes from `git status
--porcelain`, so untracked files count.

#### `runs.py`
Run-directory lifecycle and the central manifest. `stage_run` builds the
skeleton under a temporary name and `os.replace`s it into place;
`finalize_run` writes the per-run `manifest.json` and appends one locked
line to the central manifest at `roster.manifest_path` — the campaign's
`manifest_path`, resolved against `data_root`, defaulting to
`sed_fitting/runs.jsonl`. The lock is `fcntl.flock` on POSIX and
`msvcrt.locking` on Windows, taken on a separate descriptor so the
append can use its own.

`read_manifest` tolerates a torn final line and reports it, because an
interrupted append is a real outcome.

#### `jobs.py`
The fit verb's engine. `plan_job` is the whole validation prefix —
config resolution, photometry read, sidecar cross-checks, policy, run
identity — with nothing written, and `fit --dry-run` is exactly a call
to it. `run_job` stages, dispatches, plots, and finalizes.

`_machinery` stamps the versions a run's numbers actually depend on: an
eazy-py version only for the official engine, since the quick engine is
implemented here in numpy and scipy.

#### `batch.py`
The `batch` verb (§3.3). Worker pool, exact resume on `run_id`, failure
isolation, per-job logs, and a sorted report. Workers are spawned on
every platform, so a worker's state is exactly what its initializer
builds.

#### `__main__.py`
The argparse tree and the seven verb handlers. `build_parser` assembles
one subparser per verb; `_add_fit_args` defines the options shared by
`fit`, `batch` and `run` once, so the three cannot drift apart.

`_run` implements the `run` verb by mutating its own `args` and calling
`_roster` then `_batch`. It forces `build=True`, `no_resume=False` and
`dry_run=False`, so **`sedfit run` always rebuilds and always resumes**,
and neither is overridable from the command line. Reach for `roster` and
`batch` separately when you need either off.

### 4.4 The eazy backend (`sedfit/backends/eazy/`)

| Module | Role |
|---|---|
| `templates.py` | Resolves a config's `templates` to spectra or a `.param` file, resolves the TEF curve, and produces the content digests that enter `run_id`. |
| `filters_io.py` | Writes the run-local `FILTER.RES` and the catalog-to-filter translate file. |
| `fitting.py` | The official eazy-py engine. Lazy import; `PhotoZ` is driven directly rather than through `standard_output`. |
| `quick.py` | The vectorized engine, implemented in numpy and scipy. |
| `results.py` | The shared `FitResult`, the summary and singles tables, `arrays.npz`, and `load_run` for rehydrating a directory. |
| `plots.py` | The redshift-scan and SED figures. |

**Why the quick engine exists.** eazy-py rebuilds its template–filter
grid whenever the filter set changes, and per-object SPHEREx tophats
force exactly that on every object. `quick.py` reproduces the eazy-py
0.8.6 template-error treatment and chi-square while rebuilding only what
changed.

It is a reimplementation, not a wrapper, and it deviates in three
documented ways: **no IGM absorption**; bandpass integrals evaluated on
refined filter-curve nodes rather than template nodes, giving sub-grid
agreement rather than bit-identical numbers; and best-fit and fixed-z
designs taken as direct projections rather than spline interpolations.
`fitconfig` also forbids priors, non-NNLS fitters, and `extra_params`
under the quick engine, naming `engine: eazy-py` as the remedy. Both
engines share `results.percentiles_from_lnp` rather than eazy-py's own
percentile routine — see the fence below.

**The percentile fence.** `results.percentiles_from_lnp` replaces
`PhotoZ.pz_percentiles`, which resamples `ln P(z)` onto its own
log-spaced zoom grid with an Akima spline; when a zoom endpoint lands one
float ULP outside the fit grid the spline returns NaN there, a leading
NaN poisons the cumulative integral, and every percentile collapses to
the grid start (eazy-py 0.8.6).

### 4.5 The Prospector backend (`sedfit/backends/prospector/`)

| Module | Role |
|---|---|
| `obs.py` | Converts the policy's microjansky vectors to maggies and builds the sedpy filter list; attaches a prepared spectrum when the config declares one. |
| `exact_filter.py` | A `Filter` subclass whose default projection is redirected to the fixed-grid `obj_counts_lores`. |
| `model.py` | Builds the `SpecModel` (or `PolySpecModel` for joint fits) and the SPS object; asserts the live FSPS library against the config. |
| `fitting.py` | Runs dynesty or emcee with a recorded seed and checkpointing; builds the spectral noise model when jitter is configured. |
| `results.py` | Loads an h5, detects the sampler, flattens the chain, rebuilds the model. |
| `plots.py` | Corner, trace, and SED figures, plus the spectrum-fit figure on joint fits. |

**The exact-filter fence.** sedpy's default high-resolution projection
quantizes onto its own grid; for ordinary broadbands the two projections
agree, but the difference matters for narrow SPHEREx channels.
`make_exact` is applied both before and after `fix_obs`, which may
rebuild the filter list. Both calls are gated on the config's
`prospector.exact_filters` (default `true`); turning it off restores
sedpy's own projection.

**The `depends_on` ordering.** prospect propagates `depends_on` in dict
order, so `agebins` (derived from `zred`) must precede `mass` (derived
from `agebins`). The continuity builder orders them explicitly.

**Stellar library.** Every config declares `stellar_library`, and
`assert_stellar_library` compares it against the live FSPS build at fit
start. FSPS's spectral library is a compile-time choice, so a mismatched
environment is a silent scientific error unless caught here.

**Joint photometry+spectrum fits.** A `prospector.spectrum` config block
names a prepared spectrum file (see §6.6 for the contract) and turns the
fit joint: `core/spectrum.py` reads and validates the file, `build_obs`
fills the obs dictionary's `wavelength`/`spectrum`/`unc`/`mask` keys
(microjanskys to maggies, the shared `mu_lensing` divided out, the
block's `err_floor` and `mask_windows` applied), and `build_model`
returns the `PolySpecModel` concretion, whose maximum-likelihood
Chebyshev calibration vector (order `polyorder`) absorbs the smooth
ratio between the observed spectrum and the model — aperture losses and
flux-calibration drift — so the spectrum constrains features while the
photometry anchors the absolute scale. Smoothing is always declared
explicitly (`smooth_sigma_prior`/`smooth_sigma_init` or
`smooth_sigma_fixed`, km/s): prospect silently smooths by 100 km/s when
`sigma_smooth` is absent, so absence is never allowed to choose the
physics. `jitter_prior` frees a `spec_jitter` multiplier on the spectral
uncertainty through an uncorrelated noise model; `outlier_prior` frees
an `f_outlier_spec` mixture fraction (with fixed `nsigma_outlier_spec`)
that de-weights channels the model cannot reach. A photometry-only
config builds exactly the objects it always did.

### 4.6 Analysis (`sedfit/analysis/`)

`plots.py` holds the unified SED figure and its axis conventions;
producers live in the backends and hand it plain arrays. `lines.py`
carries the line catalogs — all vacuum, ordered by wavelength — and the
matplotlib annotation helpers; `load_emission_lines` refines the curated
emission wavelengths against the packaged FSPS `emlines_info.dat`.
`ew.py` measures equivalent widths against a two-sided fitted continuum.

`ew.compute_ew_batch` and `ew.print_ew_table` have **no caller inside
this package**. They serve the posterior-spectra figure suite in the
standalone `prospector_sed_fitting` engine, which is kept live for
exactly that reason.

---

## 5. The output contract

### 5.1 The SED table schema

`<target.dir>/Photometry/<prefix>_sed_<recipe>[_dered].csv`. Eight
columns, defined in `core/table.py`. Note there is **no `source`
column**: the provider strings are consumed at build time and recorded
in the sidecar's `source_strings`.

| # | Column | Contract |
|---|---|---|
| 1 | `band` | A registry band name, or a SPHEREx channel name carrying the registry's channel prefix |
| 2 | `flux_uJy` | Microjansky, AB |
| 3 | `flux_err_uJy` | Statistical, as supplied; the fitting floor is applied later |
| 4 | `scatter_uJy` | SPHEREx only: visit-to-visit scatter. Empty for broadbands |
| 5 | `wave_um` | SPHEREx only: channel center |
| 6 | `bandwidth_um` | SPHEREx only: the larger of the channel width and the bin's span |
| 7 | `n_exp` | SPHEREx only: visits in the bin |
| 8 | `qa_flags` | Broadband rows only (SPHEREx rows must leave it empty); the build fills it for the measured providers |

Broadband rows leave columns 4–7 empty — their bandpasses come from the
registry. SPHEREx rows must carry all four: `wave_um` and `bandwidth_um`
build the per-object tophat, and `scatter_uJy` and `n_exp` record the
bin's statistics for the fit policy and the provenance record.

`validate_sed_table` enforces the exact column set and order, a non-empty
table, known bands, finite fluxes, **finite and strictly positive**
errors, no duplicate bands, all four SPHEREx-only columns empty on
broadband rows, positive `wave_um`/`bandwidth_um` and non-negative
`scatter_uJy` and an **integral** `n_exp >= 1` on SPHEREx rows, an empty `qa_flags` on
SPHEREx rows, and a parseable `;`-joined `key=value` string where
`qa_flags` is present.

### 5.2 The build sidecar

`<prefix>_sed_<recipe>[_dered].provenance.json`, written beside every
table and named from it, so a dereddening build produces its own pair:

```
builder, package_version, git, generated
roster, data_root, target, position, z_ref, z_ref_kind
recipe, recipe_sha256_16
registry     {path, bandpass_sha256_16, bands: {band: hash}}
sources      [{source, role, path, sha256_16, kind, provider,
               bands, source_strings}, ...]
spherex      {table, sha256_16, model, provenance, cuts, binning, counts}
             or null
dered        {ebv, law, r_v, broadband_A_mag, spherex_A_mag}  or null
reference    {kind, anchors, tilt: {blue, red, c_blue, c_red,
              factor_min, factor_max} or null}
stitch       [{source, instrument, anchor, scale, bands}, ...]
min_coverage, n_rows, n_broadband
```

Every input is hashed, which is what makes a run directory auditable
without the tree it came from.

**The hashes are a record, not a live check.** `jobs._check_sidecar`
compares only the recipe, the `z_ref`, and the per-band bandpass hashes
against the live registry. Nothing in the fit path re-reads or re-hashes
the source CSVs, so a source file that changed after the build is **not**
detected — rebuild deliberately when the photometry moves.

### 5.3 The run directory

```
<target.dir>/SED/<recipe>/<backend>/<run_id>-<label>/
    config.json             the fully resolved configuration
    phot.csv                the photometry, byte-for-byte
    phot.provenance.json    the build sidecar, copied
    manifest.json           this run's manifest row
    plots/                  eazy:       sed_<name>.png, zscan_<name>.png,
    |                                   sed_fixed_<name>.png (z_fixed only)
    |                       prospector: corner.png, trace.png, map_sed.png
    # both eazy engines add:
    catalog.csv  template_error.dat  summary.csv  arrays.npz
    templates.param         directory-mode template sets only
    singles.csv             single mode only
    # the quick engine adds:
    engine.info
    # the official engine adds:
    FILTER.RES  FILTER.RES.info  zphot.translate  zphot.param.echo
    # prospector adds:
    result.h5               chain, model stamps, run_params
    checkpoint.save         dynesty only; kept after a successful run
    # joint fits add:
    spectrum.csv            the spectrum, byte-for-byte
    spectrum.provenance.json   its preparation sidecar, copied
    plots/spectrum_fit.png  observed vs calibrated model, chi, calibration
```

The directory carries everything a re-render needs except the band
registry, which comes from the installed package. `sedfit plot
--run-dir` on an eazy run needs no eazy-py — `results.load_run`
rehydrates a `FitResult` from `arrays.npz` — but the Prospector path
rebuilds the model and the SPS object, so it needs the Prospector/FSPS
stack and a valid `$SPS_HOME`.

The `<backend>` level is the **backend** (`eazy` / `prospector`), not
the engine: `quick` and `eazy-py` both write into `eazy/` and are told
apart by `engine.info` and the manifest's `engine` field.

**The eazy tables.** `summary.csv` carries `id, n_bands, z_ml, z_chi2,
chi2_best, n_active, redchi2, z025, z160, z500, z840, z975`, plus
`z_fixed, chi2_fixed, n_active_fixed, redchi2_fixed` under `z_fixed` and
`single_template, z_single, chi2_single` in single mode. `singles.csv`
carries `id, template, z_best, chi2_min, ampl`; `catalog.csv` carries
`id` and an `f_<band>` / `e_<band>` pair per band.

**One rename to know.** The percentiles are `z160 / z500 / z840` in
`summary.csv` and `zred_p16 / zred_p50 / zred_p84` in the manifest row
and the batch report — the same numbers under two vocabularies, mapped
in `jobs.py`. Searching a `summary.csv` for `zred_p50` finds nothing.

### 5.4 The manifest row

One JSON object per finalized run, written both to `manifest.json` and
appended to the central manifest at `roster.manifest_path` (relative to
`data_root`, default `sed_fitting/runs.jsonl`). `path` is the run
directory **relative to `data_root`**, which is how `sedfit manifest`
rejoins it:

```
run_id, path, written, target, recipe, backend, engine, sampler, seed
package_version, git_rev, git_dirty, fsps_libraries, versions
phot_sha256_16, config_sha256_16, bandpass_sha256_16
spherex_tophat_samples
bands_include, err_floor, mu_lensing, z_ref
templates      {n, set_sha256_16, source}        (eazy)
spectrum_sha256_16, n_spec_channels, n_spec_fit  (joint fits)
status         "ok" | "failed"
estimates      eazy:       {zred_p50, zred_p16, zred_p84,
                            z_ml, z_chi2, chi2_best}
               prospector: {zred_p16, zred_p50, zred_p84,
                            logmass_p16, logmass_p50, logmass_p84;
                            joint fits add sigma_smooth, spec_jitter,
                            f_outlier_spec percentiles when free}
n_bands_table, n_bands_fit, n_excluded, n_low_snr, n_qa_rejected
included, excluded_by_set, low_snr, qa_rejected, warnings
error          present only on failure
```

A failed run is finalized too, with `status: failed` and the exception
string, so the manifest records attempts rather than only successes. A
run whose fit died before finalization keeps the smaller staging stamp
instead — `{run_id, status: "staged", written, phot_sha256_16, …
machinery}` — which is how an interrupted run is told apart from a
completed one.

### 5.5 Identity, precisely

`run_id(hash_projection(resolved_config, digests=digests),
sha256(phot_bytes), registry.bandpass_hash())`.

**In** the hash: every scientific config field, the photometry bytes,
the per-band bandpass digests, the per-file template and TEF content
digests, and — on joint fits — the spectrum file's content digest.

**Out** of the hash: `name`, `n_proc`, `templates`, `template_pattern`,
`tef_file`, `prospector.spectrum.file` — these locate curves or tune
execution. Identical curves
under any path therefore share one identity, which is what lets a config
name a packaged template set rather than an absolute path. Note the
asymmetry on `template_pattern`: the pattern string is stripped, but the
files it resolves to are hashed, so changing it still changes `run_id`
whenever it changes *which* spectra are selected.

**Separate** from the hash: the machinery stamp. Two runs with the same
`run_id` and different package versions are the same science computed by
different software; `stage_run` reports that and requires `force` to
replace. Also separate: `synth.SPHEREX_TOPHAT_SAMPLES`, a module
constant rather than a config field, which changes the fluxes without
changing the identity — hence its manifest row entry.

### 5.6 The batch report

`batch_report.csv` beside the roster by default, one row per **job** —
one target under one recipe, not one per galaxy. Fourteen columns,
`batch.REPORT_COLUMNS`:

```
target, recipe, backend, status, stage, run_id, path, seconds
zred_p50, zred_p16, zred_p84, n_bands_fit, n_bands_table, error
```

`status` is `ok` / `failed` / `skipped`. `stage` names the phase a job
died in — `roster`, `applicable`, `build`, `plan`, `fit` — and is
populated only on failed and skipped rows; it is the first column to
read when a sweep goes wrong. `seconds` is null on resume-skipped rows.

`<report>.partial` carries the same rows in **completion** order and is
unlinked on a clean exit, so a `.partial` left on disk is the record of
a batch that died. Per-job logs are `<log_dir>/<target>__<recipe>.log`
(double underscore), in `batch_logs/` beside the report by default.

### 5.7 The generation report

`<roster path>.report.csv` — the suffix is appended, so a roster at
`roster.json` reports to `roster.json.report.csv`. Five columns,
`generate.REPORT_COLUMNS`: `target, source, status, detail, bands`,
where `status` is `kept` / `partial` / `dropped`, `bands` is
space-joined, and `detail` carries the per-template "absent" trail when
a source was dropped.

---

## 6. Extension points

### 6.1 Adding a band

Add an entry to `sedfit/data/registry.json` naming its `instrument` and
either a `sedpy` filter name or a `curve` path relative to the registry
file. The `instrument` must already appear in the registry's top-level
`instruments` list — the loader hard-errors on an unknown one, so add it
there in the same edit if it is new — and a band name may not begin with
the SPHEREx channel prefix. Note that `bandpass_hash()` is a global
digest, so adding a band changes every subsequent `run_id`.

When the new bands serve a different campaign, prefer a **campaign
registry** instead: a separate registry JSON passed as `--registry` to
`roster`, `build`, `fit`, `plot` and `manifest` (the Python API takes
`load_registry(path)`). Each campaign then owns its band identities, and
editing one never re-keys the other's runs. Use the same registry for
every verb of a campaign — the build sidecar's per-band hashes are
cross-checked at fit time.

### 6.2 Adding a provider

Add a token and its prefix to `sources.SOURCE_PREFIXES`. The prefix must
appear verbatim in the data files' `source` column, and the set must
stay prefix-free.

### 6.3 Adding a template set

Drop a directory of two-column ASCII spectra under
`sedfit/data/templates/<name>/` and configs can name it as
`"templates": "<name>"`. Any filesystem path still works, and paths are
tried first. The quick engine additionally requires plain two-column
ASCII and points at `engine: eazy-py` for anything else.

`resolve_spectra` globs the config's `template_pattern` (default
`*_spec.dat`) and falls back to `*.dat` **only when the pattern matches
nothing at all**. A pattern matching *part* of a directory does not fall
back: it under-selects the basis and raises a `UserWarning` naming both
counts, because a basis silently missing members is a scientific error
rather than a preference. This is live for the adopted set —
`brown14_vac_cosmos160` holds 129 `*_spec.dat` Brown spectra plus 31
differently-named COSMOS files, so fitting all 160 requires
`"template_pattern": "*.dat"`.

### 6.4 A new recipe

Add it to the campaign config's `recipes` map and regenerate the roster.
Recipes are pure declaration — no code changes — as long as the frame
model and role legality in `recipe.py` cover what you need.

### 6.5 A new backend

A backend is a runner that `jobs.run_job` dispatches to: it consumes a
`PolicyResult` and a resolved config, writes into a supplied run
directory, and returns an `estimates` dict. Adding one is not a single
registration — the backend name is branched on in several places, and
two of them currently fall through to Prospector rather than erroring:

- `fitconfig.py`: `BACKENDS` and a `_parse_<backend>` block wired into
  `parse_fit_config`. The two-backend assumptions here were fixed on
  2026-08-25: `TOP_KEYS` is now `TOP_KEYS_BASE + BACKENDS`, the exclusion
  rejects every non-selected block rather than one named "other", and the
  dispatch raises instead of falling through to Prospector. Adding a name to
  `BACKENDS` is now enough for the key allowance and the exclusion;
- `fitconfig.py`: a `USES_PHOTOMETRY` entry saying whether the backend reads
  a built photometry table. False makes `parse_fit_config` refuse the
  photometry-only top-level fields (`bands_include`, `min_valid_bands`,
  `min_snr_broadband`, `err_floor`, `qa_gates`) when written and null them
  otherwise, so they drop out of canonical JSON instead of carrying their
  photometry defaults into the run identity. A test asserts the entry exists;
- `fitconfig.py` again: an `EXECUTION_ONLY` entry, which strips the block's
  path-valued fields from the identity hash. The table is keyed by the path
  from the config root — `()` is the root, `("prospector", "spectrum")` a
  nested block — and a test asserts every name in `BACKENDS` appears, so a
  missing entry fails rather than silently hashing machine paths as literal
  strings and forking every `run_id` when the repo moves;
- `jobs.py`: `_backend_versions` and the `run_job` dispatch. Fixed 2026-08-25 —
  the four branches (those two plus `plan_job`'s digests and `run_job`'s
  manifest row) named no backend and fell through to Prospector; each now
  raises `_undispatchable`, and `plan_job` refuses an unknown backend before
  it reads photometry or stages anything. Until a backend is added to those
  four branches it is refused, by name, with a message rather than a
  `TypeError` from deep inside `apply_policy`;
- `batch.py`: `MAX_WORKERS_BY_BACKEND`;
- `__main__.py`: the `plot` verb's branch.

Keep the implementation out of `sedfit/core/`: `tests/test_layering.py`
forbids `core/` and `analysis/` from importing `eazy`, `prospect` or
`fsps` **anywhere in the file** — it walks the full AST, so a lazy
import inside a function body fails it exactly as a top-level one does.
Note it does not forbid `core/` importing
`sedfit.backends.*` — that layering is a convention, not a test.

`backends/linear/` (2026-08-25) is the worked counter-example: a complete
backend that deliberately **skips** this list. It has the config half — a
`LINEAR_KEYS` block, `_parse_linear`, and its `EXECUTION_ONLY` and
`USES_PHOTOMETRY` entries — and then writes its own runs through
`backends/linear/runner.py` rather than through `jobs.py`, because `jobs.py`
gates on photometry at five points and a spectrum-only fit has no photometry
table, no roster recipe, and nothing to put in fifteen of the manifest's
columns. A `linear` config parses, resolves and hashes like any other;
`plan_job` then refuses it by name. Its shared NNLS solve moved to
`core/nnls.py` rather than being imported across from `eazy`. Its `linear`
block covers every term of the model equation, `transmission` included — a
backend whose config cannot reach one of its own terms is a trap, and that
one silently defaulted to ones until review caught it.

The cost of that split is that linear runs are absent from the central
manifest and from `sedfit fit` / `sedfit batch`. Three seams keep it
reversible: `provenance.run_id`'s two hashes are optional, `runs.stage_run`
takes photometry optionally, and `USES_PHOTOMETRY` already marks which
backends would need conditional gates. See DESIGN.md 16.3, 16.4a and 16.4b.

### 6.6 Fitting an observed spectrum

The Prospector backend fits a spectrum jointly with the photometry when
the config's prospector block carries a `spectrum` object. The package
deliberately owns none of the spectrum's preparation — flux calibration,
telluric handling, error inflation, frame conversion and quality masking
happen in target-specific tooling — and consumes one prepared file:

- a CSV with exactly the columns `wave_A, flux_uJy, flux_err_uJy, mask`:
  observed-frame **vacuum** wavelengths in Angstroms (FSPS wavelengths
  are vacuum; feeding air wavelengths misplaces every line), f_nu flux
  densities in microjanskys, and a 0/1 fit-inclusion mask. Masked
  channels may be non-finite; unmasked ones may not.
- a required sidecar `<stem>.provenance.json` declaring
  `"wave_frame": "vacuum"` and `"flux_unit": "uJy"`; everything else in
  it is free-form preparation provenance, staged verbatim into the run
  directory.

The config block: `file` (relative paths resolve against the config
file's directory), `polyorder` (default 12; 0 disables the calibration
vector), `smooth_sigma_prior`/`smooth_sigma_init` or
`smooth_sigma_fixed` [km/s], optional `jitter_prior` (frees
`spec_jitter`), optional `outlier_prior` + `outlier_nsigma` (frees
`f_outlier_spec`), `err_floor` (fractional, on the spectrum),
`mask_windows` (observed-frame vacuum `[lo, hi]` intervals excluded on
top of the file's mask — analysis choices, where the file mask records
data quality), and optional `eline_sigma_kms` (requires `nebular`):
turns FSPS's in-spectrum nebular lines off (`nebemlineinspec: false`)
and has prospect draw them at this fixed total width instead. FSPS
inserts lines at a library-resolution convention width that need not
match the data; when line profiles carry information, set this to the
measured intrinsic-plus-instrumental width. Band predictions still
receive the line fluxes through `nebline_photometry`. A null `spectrum` (the default) leaves every existing
photometry-only identity untouched, because canonical JSON drops null
keys; the file's path is execution-only while its content digest enters
`run_id`.

### 6.7 A blind redshift from a spectrum alone

`backends/linear` solves for redshift with no photometry and no input
redshift. Four config blocks turn that into a usable blind fitter; each is
null by default, so a config that omits them keeps the identity it had.

**`gas`** adds emission lines as extra NNLS columns, built analytically as
Gaussians at observed wavelength rather than resampled from files or baked
into nebular-on SSPs. Against an absorption-only basis a blue emission-line
galaxy fits to the wrong redshift *silently* — the sigma clip then deletes
the lines, which were the only redshift information in the spectrum. Fields:
`lines` (**required**, a packaged list name such as `optical`, or a path — a
short list is silent, so there is no default), `sigma_kms` (the nebular
width, fixed rather than free: a symmetric kernel does not move a line
centroid, so the redshift is insensitive to it at first order while the
amplitude is not), and `ratio_locked` (groups entering as **one** column at a
ratio atomic physics fixes — [OIII] 5007/4959 = 2.98 and [NII] 6584/6548 =
3.05 by default; Balmer lines stay free, because the decrement is a dust
measurement). Line fluxes are reported in `gas_fluxes`, apart from
`light_fractions`: an amplitude on a unit-integrated line column is a flux,
not a share of a continuum. Gas columns count in `dof` whether or not they
take amplitude.

**`lsf`** and **`template_resolution`** supply the instrument line spread and
the basis's own resolution. The kernel is the quadrature *difference*,

    sigma_kernel(l_obs)^2 = sigma_LSF(l_obs)^2 - [sigma_lib(l_obs/(1+z)) * (1+z)]^2

applied in the observed frame on the design matrix, after the redshift shift.
Both blocks share one shape: exactly one of `constant` or `file`, a `unit`
from `R`, `fwhm_A`, `sigma_A`, `fwhm_kms`, `sigma_kms`, and `wave_frame` when
it is a file. `template_resolution` is required whenever `lsf` is set, and
its `file` accepts a packaged set name, resolving to the `resolution.txt`
that set ships. A constant-R LSF against a constant-R library gives one
velocity width, which folds into the cached rest-frame broadening for free;
anything else becomes a per-pixel convolution on the spectrum's own grid.
Ignoring a real LSF folds the instrument into the reported dispersion — 80
km/s intrinsic reads as ~93 behind a rising R = 1500 to 4000 — while leaving
the redshift alone.

`lsf.on_undersampled` has **no default**, because neither shipped set can be
matched to MUSE over roughly half the band and all three answers are
defensible: `raise` refuses, naming the wavelengths and the redshift;
`degrade_data` smooths the data up to the library resolution (a normalized
convolution over the fitted channels, with the variance propagated as
`sum(w^2 v)/sum(w)^2` — the neglected off-diagonal terms are why a degraded
run's chi-square does not compare with an undegraded one's, and it refuses
outright when the library's velocity width moves with redshift, as a
constant-FWHM library's does); `ignore` leaves both alone, which costs the
dispersion and not the redshift.

**`scan`** runs the two-stage blind scan: a coarse pass at
`n_poly_iter_coarse` and one representative sigma, then `fit_spectrum`
unmodified over a `window_steps`-wide window of the configured grid. A
single-stage scan at production settings over a wide redshift range is tens
of thousands of evaluations. `z_step_coarse` defaults to **null, meaning
derived from the basis** rather than fixed: with gas present the narrowest
feature is a line rather than a stellar absorption trough, and half a coarse
step must stay inside one line width. The scan returns its coarse grid and
ranked minima separately from the fit, so a coarse chi-square and a final one
never share a field.

Every fit — scanned or not — now keeps the grid it scanned and reports the
distinct local minima with their `delta_chi2` to the winner. A catastrophic
redshift is a minimum-selection failure, which the Hessian error cannot
express: it states the precision of the minimum the fit landed in, not
whether that was the right one. Distinctness is a **velocity** separation,
`|dz|/(1+z) > dv/c`, default 1000 km/s. `delta_chi2` is a discriminant, not a
probability — the variance is diagonal and a spectrum is thousands of
correlated channels, so any threshold gating a catalog must be calibrated
against a truth sample first.

---

## 7. Behavior worth knowing

**Roster load is expensive on purpose.** Every verb that takes
`--roster` opens every declared source CSV. On a network or
cloud-synced filesystem with evicted files, a cold load takes minutes.
That is the cost of the guarantee that a loaded roster's declarations
are true. A batch pays it once in the parent **and once per worker**, so
a six-worker sweep loads the roster seven times.

**`--deredden` selects a different table.** A dereddening build writes
`..._dered.csv` *instead of* the as-measured table — one call writes one
table — and the fit looks for whichever the flag selects. Run `build`
twice to have both on disk. Inside one batch the flag drives both the
build and the selection, so they cannot disagree; the trap is across
invocations, where `sedfit batch --build` followed by `sedfit batch
--deredden` finds no `_dered` table.

**A `template_pattern` can silently shrink your basis.** The default
`*_spec.dat` selects only the 129 Brown spectra of the 160-template
adopted set. It warns, but the warning is a `UserWarning` on stdout
inside a batch worker's per-job log, which is easy to miss. The template
basis is the dominant systematic a fit carries.

**`SPHEREX_TOPHAT_SAMPLES` is quadrature resolution, not cosmetics.**
Prospector's exact filters integrate the source on the filter grid, so
the tophat sample count sets the projection accuracy. It is not in
`run_id`, so two runs at different sample counts share an id and differ
numerically; the manifest row records the value that ran.

**Excluded bands are not dropped from the table.** The policy marks them
and each backend substitutes its own missing representation — a sentinel
for eazy, omission for Prospector. The table always has every row.

**An unseeded Prospector dry run reports an id the real run will not
use.** `fit --dry-run` resolves and reports a `run_id`, but if
`prospector.seed` is null the real run draws a fresh seed and therefore
a different id. The dry run says so on the line it prints.

**Batch resume is by `run_id`, not by target.** Changing anything that
enters the hash — a config field, the photometry, a bandpass — makes
every job a new run, and a resumed batch will run all of them. `--build`
in particular can move `run_id` if the measured inputs changed since the
table was written.

**Batch resume interacts with `--force`.** `resume` is computed as
`resume and not force`, so `--force` re-runs every job whether or not
`--no-resume` was given.

**Windows is supported but unverified.** The core install and both eazy
engines are pure Python with no platform-specific code paths. The two
places that do branch are the worker pool, which uses `spawn`
everywhere, and the manifest lock, which uses `msvcrt.locking` on
Windows and `fcntl.flock` elsewhere. The Windows lock takes its byte at
`runs.LOCK_OFFSET`, far past any manifest, because Windows byte-range
locks are mandatory and would otherwise block the append itself. There
is no CI and no Windows-conditional test, so treat the platform as
untested. FSPS compiles from Fortran, so the Prospector extra is best
supported on Linux and macOS.

**Matplotlib backend.** The CLI and the batch worker initializer select
`Agg`. Importing `sedfit.analysis.plots` leaves the caller's backend
alone — with one exception: `run_batch(workers=1)` runs its initializer
in the calling process, and therefore switches that process to `Agg`.

---

## 8. Testing

```bash
python -m pytest tests/
```

The suite is synthetic end to end: no observational data is committed.
`tests/data/spherex_visits.csv` is checked in but wholly synthetic, and
`tests/data/make_spherex_fixture.py` regenerates it from a seeded
generator. `tests/synthetic.py` holds the shared template and photometry
builders, and imports no backend stack so the modules that use it run on
a core install.

| Module | Covers |
|---|---|
| `test_registry.py`, `test_roster.py`, `test_recipe.py`, `test_sources.py`, `test_generate.py` | The declaration layer, including the deprecated key spellings |
| `test_spherex.py`, `test_synth.py`, `test_stitch.py`, `test_table.py`, `test_build.py`, `test_dered.py` | Assembly |
| `test_policy.py`, `test_fitconfig.py`, `test_provenance.py`, `test_runs.py` | The fitting contract and run identity |
| `test_eazy_backend.py` | Both engines, including quick-versus-official agreement |
| `test_prospector_backend.py` | Model construction, the error vector, exact filters |
| `test_jobs.py`, `test_batch.py`, `test_cli.py` | Orchestration, the CLI surface, and the error messages |
| `test_layering.py` | That `core/` and `analysis/` import no backend stack |
| `test_analysis.py` | Line catalogs and equivalent widths |
| `test_package.py` | That the package imports and carries a version |

`test_eazy_backend.py` skips without the `eazy` extra and
`test_prospector_backend.py` without `prospector`; everything else,
including the quick engine's path through the orchestration and the CLI,
runs on a core install.

Three tests will fail loudly on an unintended change, and each is
deliberate to touch:
`tests/test_eazy_backend.py::test_quick_vs_official_agreement` pins the
two engines to each other, and
`tests/test_fitconfig.py::test_projection_golden` and
`tests/test_provenance.py::test_run_id_golden` freeze the hash
projection and the canonical form behind literal run ids.

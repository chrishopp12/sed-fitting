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
| `fit` | `jobs.run_job` | one run directory + one manifest row |
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
| **provider / source prefix** | `core/sources.py` | A provider token (`legacy`, `jplus`, `aperture`, …) maps to a string prefix that every one of its rows carries in the table's `source` column. This is what lets two providers supply the same band from one file. |
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
| **source** | (1) A roster **source** — one declared file. (2) The SED table's **`source` column**, a provenance string. (3) The `sedphot` `source` prefix vocabulary. |
| **scale** | An instrument group's achromatic factor (`stitch`), never the tilt. |
| **prefix** | (1) A target's **filename stem** (`Target.prefix`). (2) A provider's **source-column prefix** (`sources.SOURCE_PREFIXES`). Unrelated. |
| **manifest** | (1) The **central** append-only `runs.jsonl`. (2) The **per-run** `manifest.json`, one row, the same schema. |

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
  +-- core.build.build_target(roster, target, recipe, registry=..., mw_ebv=...)
        |
        1. sources.select_rows        per recipe entry: the declared bands,
        |                             selected by provider prefix, exactly
        |                             one row each
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
CLI: sedfit fit --roster R --target X --recipe R --config CFG
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
  |         prospector -> model.build_model + build_sps -> fitting.run_sampler
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

### 3.4 Tracing a flux to a fitted redshift

1. A row in one of the target's declared source CSVs — the path comes
   from the roster and resolves against `target.dir` — selected by
   `sources.select_rows` because its `source` column starts with the
   declared provider's prefix.
2. Possibly multiplied by a dereddening factor (`dered.band_factor`).
3. Possibly multiplied by its instrument group's scale
   (`stitch.measure_scale`), or the spectrum tilted onto the anchors.
4. Written to `<prefix>_sed_<recipe>.csv` as one row of `band`,
   `flux_uJy`, `flux_err_uJy`.
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

#### `recipe.py`
Parses and validates one recipe. Checks the frame model: `anchor` roles
are legal only under `reference: anchors`, which requires one or two of
them; `stitch` is legal under `spherex` and `anchors`; `float`
everywhere. No band may be *listed* by two source entries; the same rule
on implicitly-consumed bands is enforced at build time by
`build.effective_bands`.

The `spherex` block is tri-state: **absent** (the build hard-errors when
the target declares a SPHEREx table), **null** (deliberate exclusion), or
a block of cuts and binning.

#### `sources.py`
The provider-prefix vocabulary and the row selection built on it.
`SOURCE_PREFIXES` maps ten provider tokens to the string prefixes that
appear verbatim in data files. The set must stay **prefix-free** — no
entry may be a prefix of another — so `startswith` matching is
unambiguous.

`select_rows` is the error-raising selection used at build time;
`match_counts` is the counting version used at roster load. `check_position`
cross-checks a table's `target_ra`/`target_dec` against the roster
position, and a table carrying neither column raises a `UserWarning`
rather than passing silently.

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

### 4.3 Fitting (`sedfit/core/`, `sedfit/jobs.py`)

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

#### `provenance.py`
`canonical_json` and `run_id`, plus file and byte digests and
`git_state`. The canonical form is version-stable: changing it changes
every run identity. `git_state`'s `dirty` comes from `git status
--porcelain`, so untracked files count.

#### `runs.py`
Run-directory lifecycle and the central manifest. `stage_run` builds the
skeleton under a temporary name and `os.replace`s it into place;
`finalize_run` writes the per-run `manifest.json` and appends one locked
line to the central `runs.jsonl`. The lock is `fcntl.flock` on POSIX and
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
| `obs.py` | Converts the policy's microjansky vectors to maggies and builds the sedpy filter list. |
| `exact_filter.py` | A `Filter` subclass whose default projection is redirected to the fixed-grid `obj_counts_lores`. |
| `model.py` | Builds the `SpecModel` and the SPS object; asserts the live FSPS library against the config. |
| `fitting.py` | Runs dynesty or emcee with a recorded seed and checkpointing. |
| `results.py` | Loads an h5, detects the sampler, flattens the chain, rebuilds the model. |
| `plots.py` | Corner, trace, and SED figures. |

**The exact-filter fence.** sedpy's default high-resolution projection
quantizes onto its own grid; for ordinary broadbands the two projections
agree, but the difference matters for narrow SPHEREx channels.
`make_exact` is applied both before and after `fix_obs`, which may
rebuild the filter list.

**The `depends_on` ordering.** prospect propagates `depends_on` in dict
order, so `agebins` (derived from `zred`) must precede `mass` (derived
from `agebins`). The continuity builder orders them explicitly.

**Stellar library.** Every config declares `stellar_library`, and
`assert_stellar_library` compares it against the live FSPS build at fit
start. FSPS's spectral library is a compile-time choice, so a mismatched
environment is a silent scientific error unless caught here.

### 4.6 Analysis (`sedfit/analysis/`)

`plots.py` holds the unified SED figure and its axis conventions;
producers live in the backends and hand it plain arrays. `lines.py`
carries the line catalogs — all vacuum, ordered by wavelength — and the
matplotlib annotation helpers. `ew.py` measures equivalent widths
against a two-sided fitted continuum.

---

## 5. The output contract

### 5.1 The SED table schema

Eight columns, defined in `core/table.py`.

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
`scatter_uJy` and `n_exp >= 1` on SPHEREx rows, an empty `qa_flags` on
SPHEREx rows, and a parseable `;`-joined `key=value` string where
`qa_flags` is present.

### 5.2 The build sidecar

`<prefix>_sed_<recipe>.provenance.json`, written beside every table:

```
builder, package_version, git, generated
roster, data_root, target, position, z_ref, z_ref_kind
recipe, recipe_sha256_16
registry     {path, bandpass_sha256_16, bands: {band: hash}}
sources      [{source, role, path, sha256_16, kind, provider,
               bands, source_strings}, ...]
spherex      {table, sha256_16, model, provenance, cuts, binning, counts}
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
    plots/                  the figures
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
```

The directory carries everything a re-render needs except the band
registry, which comes from the installed package. `sedfit plot
--run-dir` on an eazy run needs no eazy-py — `results.load_run`
rehydrates a `FitResult` from `arrays.npz` — but the Prospector path
rebuilds the model and the SPS object, so it needs the Prospector/FSPS
stack and a valid `$SPS_HOME`.

### 5.4 The manifest row

One JSON object per finalized run, written both to `manifest.json` and
appended to the central `runs.jsonl`:

```
run_id, path, written, target, recipe, backend, engine, sampler, seed
package_version, git_rev, git_dirty, fsps_libraries, versions
phot_sha256_16, config_sha256_16, bandpass_sha256_16
bands_include, err_floor, mu_lensing, z_ref
templates      {n, set_sha256_16, source}        (eazy)
status         "ok" | "failed"
estimates      eazy:       {zred_p50, zred_p16, zred_p84,
                            z_ml, z_chi2, chi2_best}
               prospector: {zred_p16, zred_p50, zred_p84,
                            logmass_p16, logmass_p50, logmass_p84}
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
the per-band bandpass digests, and the per-file template and TEF content
digests.

**Out** of the hash: `name`, `n_proc`, `templates`, `template_pattern`,
`tef_file` — these locate curves or tune execution. Identical curves
under any path therefore share one identity, which is what lets a config
name a packaged template set rather than an absolute path.

**Separate** from the hash: the machinery stamp. Two runs with the same
`run_id` and different package versions are the same science computed by
different software; `stage_run` reports that and requires `force` to
replace.

---

## 6. Extension points

### 6.1 Adding a band

Add an entry to `sedfit/data/registry.json` naming its `instrument` and
either a `sedpy` filter name or a `curve` path relative to the registry
file. Note that `bandpass_hash()` is a global digest, so adding a band
changes every subsequent `run_id`.

### 6.2 Adding a provider

Add a token and its prefix to `sources.SOURCE_PREFIXES`. The prefix must
appear verbatim in the data files' `source` column, and the set must
stay prefix-free.

### 6.3 Adding a template set

Drop a directory of two-column ASCII spectra under
`sedfit/data/templates/<name>/` and configs can name it as
`"templates": "<name>"`. Any filesystem path still works, and paths are
tried first. `resolve_spectra` globs the config's `template_pattern`
(default `*_spec.dat`) and falls back only to `*.dat`, so name the files
accordingly or set the pattern. The quick engine additionally requires
plain two-column ASCII and points at `engine: eazy-py` for anything
else.

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

- `fitconfig.py`: `BACKENDS`, `TOP_KEYS`, a `_parse_<backend>` block
  wired into `parse_fit_config`, and the `other = "prospector" if
  backend == "eazy" else "eazy"` exclusion, which assumes exactly two;
- `jobs.py`: `_backend_versions` and the `run_job` dispatch — **both
  else-branches route to Prospector today**;
- `batch.py`: `MAX_WORKERS_BY_BACKEND`;
- `__main__.py`: the `plot` verb's branch.

Keep the implementation out of `sedfit/core/`: `tests/test_layering.py`
forbids `core/` and `analysis/` from importing `eazy`, `prospect` or
`fsps` at module scope. Note it does not forbid `core/` importing
`sedfit.backends.*` — that layering is a convention, not a test.

---

## 7. Behavior worth knowing

**Roster load is expensive on purpose.** Every verb that takes
`--roster` opens every declared source CSV. On a network or
cloud-synced filesystem with evicted files, a cold load takes minutes.
That is the cost of the guarantee that a loaded roster's declarations
are true. A batch pays it once in the parent **and once per worker**, so
a six-worker sweep loads the roster seven times.

**`--deredden` selects a different table.** The build writes
`..._dered.csv` alongside the as-measured one, and the fit looks for
whichever the flag selects. A batch that builds without `--deredden` and
fits with it will not find its table.

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

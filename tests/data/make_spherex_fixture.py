"""Regenerate spherex_visits.csv, the synthetic SPHEREx per-visit fixture.

Every value is generated here (seeded); the position, identifiers, and
fluxes are invented. The layout exercises the ingest paths: a blue
near-duplicate pair that merges, adaptive red bins with a four-visit
reddest bin, and fifteen visits carrying reject-worthy flag bits.

    python tests/data/make_spherex_fixture.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "spherex_visits.csv"
SOURCE_FLAG = 1 << 21
RA, DEC = 150.11875, 2.20514


def _flux(lam: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    smooth = 2400.0 * (lam / 0.75) ** -0.35
    return smooth * rng.normal(1.0, 0.02, size=lam.shape)


def main() -> None:
    rng = np.random.default_rng(42)

    blue = list(np.round(np.linspace(0.752, 2.60, 32), 5))
    blue.append(blue[4] + 0.0008)          # merges with its neighbor
    blue = np.array(sorted(blue))

    red_centers = np.linspace(2.75, 4.95, 9)
    red_counts = [5, 5, 5, 5, 5, 5, 4, 4, 4]
    red = np.concatenate([
        center + 0.008 * np.arange(n)
        for center, n in zip(red_centers, red_counts)
    ])

    lam_kept = np.concatenate([blue, red])
    lam_rejected = np.concatenate([
        np.round(np.linspace(0.80, 2.45, 8), 5) + 0.0004,
        red_centers[:7] + 0.016,
    ])
    reject_bits = [0, 2, 6, 7, 17, 19, 32, 35, 0, 2, 17, 32, 35, 19, 6]

    rows = []
    for i, lam in enumerate(np.concatenate([lam_kept, lam_rejected])):
        kept = i < len(lam_kept)
        flags = SOURCE_FLAG
        if not kept:
            flags |= 1 << reject_bits[i - len(lam_kept)]
        width = 0.018 if lam <= 2.65 else 0.040
        flux = float(_flux(np.array([lam]), rng)[0])
        rows.append({
            "ra": RA, "dec": DEC,
            "x_image": round(float(rng.uniform(100, 1900)), 4),
            "y_image": round(float(rng.uniform(100, 1900)), 4),
            "mjd": round(float(rng.uniform(60000, 60400)), 6),
            "flux_bkg": round(float(rng.normal(0.15, 0.02)), 8),
            "local_bkg_flg": bool(i % 17 == 0),
            "flags": flags,
            "fit_ql": round(float(rng.uniform(0.5, 4.0)), 6),
            "deep_flg": False,
            "det_id": 1 + (i % 6),
            "lvf_id": 202000000000000 + i,
            "obs_publisher_did": f"ivo://test/spherex_fixture?row_{i:03d}",
            "lambda": round(float(lam), 8),
            "lambda_width": width,
            "flux": round(flux, 4),
            "flux_err": round(0.03 * flux, 6),
        })

    frame = pd.DataFrame(rows).sample(frac=1.0, random_state=7)
    frame.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(frame)} rows)")


if __name__ == "__main__":
    main()

"""V1.3 - requirement_breadth composite-score correlation check (PRE-COMMITTED).

For each component, compute Pearson correlation with description_cleaned_length.
If ANY component r > 0.3, flag. Compute length-residualized breadth and re-report effect.
"""
import duckdb
import pandas as pd
import numpy as np
from scipy import stats
import json
import math
from pathlib import Path

OUT_DIR = Path("/home/jihgaboot/gabor/job-research/exploration/artifacts/V1")
T11_PARQ = "/home/jihgaboot/gabor/job-research/exploration/artifacts/T11/T11_posting_features.parquet"
CLEAN_PARQ = "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_cleaned_text.parquet"


def main():
    con = duckdb.connect()

    # Load T11 features joined with cleaned-text length
    # Note: T11 has desc_len_chars already (from LLM-labeled set). Let's also join to cleaned text to get both.
    df = con.execute(f"""
        SELECT t.*, LENGTH(c.description_cleaned) AS desc_cleaned_length
        FROM '{T11_PARQ}' t
        LEFT JOIN '{CLEAN_PARQ}' c USING (uid)
    """).df()
    print(f"Loaded {len(df)} rows from T11 features")

    COMPONENTS = ["tech_count", "soft_skill_count", "org_scope_count",
                  "education_level", "yoe_numeric", "management_STRICT_count",
                  "ai_count"]
    LEN_COLS = ["desc_len_chars", "desc_cleaned_length"]

    print("\n=== Pearson correlations with desc_cleaned_length ===")
    rows = []
    for col in COMPONENTS:
        for lcol in LEN_COLS:
            sub = df[[col, lcol]].dropna()
            if len(sub) < 10 or sub[col].var() == 0 or sub[lcol].var() == 0:
                continue
            r, p = stats.pearsonr(sub[col], sub[lcol])
            flag = " <-- FLAG r>0.3" if abs(r) > 0.3 else ""
            print(f"  {col:30s}  vs {lcol:22s}  r={r:+.3f}  p={p:.1e}  n={len(sub)}{flag}")
            rows.append({"component": col, "length_col": lcol, "r": float(r), "p": float(p), "n": int(len(sub))})

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(OUT_DIR / "V1_3_correlations.csv", index=False)

    # Also check requirement_breadth <> length
    print("\n=== requirement_breadth vs length ===")
    for lcol in LEN_COLS:
        sub = df[["requirement_breadth", lcol]].dropna()
        r, p = stats.pearsonr(sub["requirement_breadth"], sub[lcol])
        print(f"  requirement_breadth vs {lcol:22s}  r={r:+.3f}  p={p:.1e}  n={len(sub)}")

    # Length-residualize requirement_breadth
    # Regress breadth on length (cleaned); take residuals.
    print("\n=== Length residualization ===")
    # Use LLM-labeled + available cleaned length rows
    d2 = df.dropna(subset=["requirement_breadth", "desc_cleaned_length"]).copy()
    X = d2["desc_cleaned_length"].values.reshape(-1, 1)
    y = d2["requirement_breadth"].values
    # OLS via numpy
    X_ones = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X_ones, y, rcond=None)
    intercept, slope = coef[0], coef[1]
    y_pred = X_ones @ coef
    resid = y - y_pred
    d2["breadth_residual"] = resid
    d2["breadth_pred"] = y_pred
    print(f"  breadth = {intercept:.3f} + {slope:.6f} * desc_cleaned_length")
    # R^2
    ss_tot = np.sum((y - y.mean())**2)
    ss_res = np.sum(resid**2)
    r2 = 1 - ss_res / ss_tot
    print(f"  R^2 = {r2:.3f}  (proportion of breadth variance explained by length)")

    # Now compute +39% effect on raw breadth vs residualized breadth
    d2["period_group"] = d2["period"].apply(lambda p: "2024" if str(p).startswith("2024") else "2026")

    def snr_stat(col):
        g24 = d2[d2.period_group == "2024"][col]
        g26 = d2[d2.period_group == "2026"][col]
        p24 = g24.mean()
        p26 = g26.mean()
        v24 = g24.var() / len(g24)
        v26 = g26.var() / len(g26)
        se = math.sqrt(v24 + v26)
        delta = p26 - p24
        snr = delta / se if se > 0 else float("inf")
        pct = (p26 - p24) / p24 if p24 != 0 else float("nan")
        return {"p24": float(p24), "p26": float(p26), "delta": float(delta),
                "pct": float(pct), "se": float(se), "snr": float(snr),
                "n24": int(len(g24)), "n26": int(len(g26))}

    raw = snr_stat("requirement_breadth")
    resid_ = snr_stat("breadth_residual")
    print("\n=== Breadth effect: raw vs residualized ===")
    print(f"  RAW breadth:   2024={raw['p24']:.3f}  2026={raw['p26']:.3f}  Δ={raw['delta']:+.3f} ({raw['pct']:+.1%})  SNR={raw['snr']:.2f}")
    print(f"  RESID breadth: 2024={resid_['p24']:.3f}  2026={resid_['p26']:.3f}  Δ={resid_['delta']:+.3f}  SNR={resid_['snr']:.2f}")

    # Also per-component residualized & effect
    print("\n=== Per-component effect (raw) ===")
    component_effects = {}
    for col in COMPONENTS + ["desc_cleaned_length"]:
        if col not in d2.columns:
            continue
        sub = d2.dropna(subset=[col])
        if len(sub) < 10:
            continue
        g24 = sub[sub.period_group == "2024"][col]
        g26 = sub[sub.period_group == "2026"][col]
        p24 = g24.mean()
        p26 = g26.mean()
        delta = p26 - p24
        pct = delta / p24 if p24 != 0 else float("nan")
        print(f"  {col:30s}  2024={p24:.3f}  2026={p26:.3f}  Δ={delta:+.3f} ({pct:+.1%})")
        component_effects[col] = {"p24": float(p24), "p26": float(p26),
                                   "delta": float(delta), "pct": float(pct)}

    # Summary
    summary = {
        "raw_breadth_effect": raw,
        "resid_breadth_effect": resid_,
        "length_vs_breadth_r2": float(r2),
        "length_vs_breadth_slope": float(slope),
        "length_vs_breadth_intercept": float(intercept),
        "component_effects": component_effects,
        "flagged_components": corr_df[corr_df["r"].abs() > 0.3].to_dict("records"),
    }
    with open(OUT_DIR / "V1_3_correlation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {OUT_DIR / 'V1_3_correlation_summary.json'}")

    # Verdict
    raw_snr = raw["snr"]
    resid_snr = resid_["snr"]
    raw_pct = raw["pct"]
    resid_delta_sign = resid_["delta"] > 0
    # If resid SNR still high and positive delta, breadth rise is content-diversification not length-driven
    verdict = "VERIFIED" if resid_snr > 10 and resid_delta_sign else (
        "CORRECTED (length-residualized signal weakened but persists)" if resid_snr > 3 and resid_delta_sign else
        "FAILED (length-residualized signal disappears)"
    )
    print(f"\nVerdict on Wave 2 finding 6 (breadth rise): {verdict}")
    print(f"  Raw: +{raw['delta']:.2f} ({raw_pct:+.1%}), SNR {raw_snr:.2f}")
    print(f"  Residualized: +{resid_['delta']:.3f}, SNR {resid_snr:.2f}")


if __name__ == "__main__":
    main()

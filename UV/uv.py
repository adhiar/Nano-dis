"""
Publication-ready UV-Vis and Tauc plotting with automatic band-gap estimation.

Examples
--------
UV-Vis only:
    python uvvis_tauc_auto.py --input "NiO.WData" --sample NiO

UV-Vis + automatic direct Tauc gap:
    python uvvis_tauc_auto.py --input "NiO.WData" --sample NiO --tauc --baseline-zero

Automatic indirect Tauc gap:
    python uvvis_tauc_auto.py --input "NiO.WData" --sample NiO --tauc --transition indirect

Manual override of the linear Tauc region:
    python uvvis_tauc_auto.py --input "NiO.WData" --sample NiO --tauc \
        --fit-range "3.4,4.0"

Important
---------
The automatic routine identifies the *lowest robust linear Tauc edge* in the
measured spectrum. If defect/sub-gap absorption is present, that lowest edge
may not represent the intrinsic fundamental gap. Always inspect the fitted
region and use --fit-range to override it when scientifically justified.
"""

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def read_wdata(path):
    lines = Path(path).read_text(errors="ignore").splitlines()

    data_start = None
    for i, line in enumerate(lines):
        if line.strip().strip('"') == "Data":
            data_start = i
            break
    if data_start is None:
        raise ValueError(f'Could not find a "Data" section in {path}')

    header_idx = None
    for i in range(data_start + 1, min(data_start + 12, len(lines))):
        if "Wavelength" in lines[i] and "Absorbance" in lines[i]:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find Wavelength,Absorbance header.")

    rows = []
    for line in lines[header_idx + 1:]:
        parts = [p.strip().strip('"') for p in line.strip().split(",")]
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue

    if not rows:
        raise ValueError(f"No numerical UV-Vis data found in {path}")

    return pd.DataFrame(rows, columns=["Wavelength_nm", "Absorbance"])


def read_uvvis(path):
    path = Path(path)

    if path.suffix.lower() == ".wdata":
        df = read_wdata(path)
    else:
        raw = pd.read_csv(path)
        cleaned = {
            c: re.sub(r"[^a-z0-9]+", "", str(c).lower())
            for c in raw.columns
        }

        wl_col = None
        ab_col = None
        for original, c in cleaned.items():
            if c in {"wavelength", "wavelengthnm", "lambda", "lambdanm"}:
                wl_col = original
            if c in {"absorbance", "abs", "a"}:
                ab_col = original

        if wl_col is None or ab_col is None:
            raise ValueError(
                "CSV must contain wavelength and absorbance columns.\n"
                f"Found: {list(raw.columns)}"
            )

        df = raw[[wl_col, ab_col]].copy()
        df.columns = ["Wavelength_nm", "Absorbance"]

    df["Wavelength_nm"] = pd.to_numeric(df["Wavelength_nm"], errors="coerce")
    df["Absorbance"] = pd.to_numeric(df["Absorbance"], errors="coerce")
    df = df.dropna()
    df = df[df["Wavelength_nm"] > 0]
    return df.sort_values("Wavelength_nm").reset_index(drop=True)



def baseline_shift_to_zero(df):
    """
    Shift the absorbance baseline so the minimum absorbance becomes zero.

    A_shifted = A_raw - min(A_raw)

    This removes negative absorbance values without rescaling the spectrum
    to 0-1, so the relative spectral shape and amplitude are preserved.
    """
    out = df.copy()
    amin = float(out["Absorbance"].min())
    out["Absorbance_raw"] = out["Absorbance"]
    out["Absorbance"] = out["Absorbance"] - amin
    return out, amin


def publication_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 18,
        "axes.labelweight": "bold",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.5,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def bold_ticks(ax):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")


def style_axes(ax):
    ax.tick_params(
        axis="both", which="major",
        direction="in", top=True, right=True,
        width=1.5, length=8,
    )
    ax.tick_params(
        axis="both", which="minor",
        direction="in", top=True, right=True,
        width=1.2, length=4,
    
    )
    ax.tick_params(top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    bold_ticks(ax)


def compact_number(x, pos=None):
    ax = abs(x)
    if ax >= 1e6:
        value, suffix = x / 1e6, "M"
    elif ax >= 1e3:
        value, suffix = x / 1e3, "k"
    else:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:g}"

    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}{suffix}"
    return f"{value:.1f}".rstrip("0").rstrip(".") + suffix


def maybe_compact_y(ax):
    ylim = ax.get_ylim()
    if max(abs(ylim[0]), abs(ylim[1])) >= 1e3:
        ax.yaxis.set_major_formatter(FuncFormatter(compact_number))
        bold_ticks(ax)


def plot_uvvis(df, sample, out_png, out_pdf):
    publication_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.3))

    ax.plot(
        df["Wavelength_nm"],
        df["Absorbance"],
        color="black",
        linewidth=1.8,
    )

    ax.set_xlabel("Wavelength (nm)", fontweight="bold")
    ax.set_ylabel("Absorbance (a.u.)", fontweight="bold")

    # ax.text(
    #     0.97, 0.95, sample,
    #     transform=ax.transAxes,
    #     ha="right", va="top",
    #     fontsize=15, fontweight="bold",
    # )
    ax.tick_params(top=False, right=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def tauc_transform(df, transition="direct", path_length_cm=1.0):
    wavelength = df["Wavelength_nm"].to_numpy(dtype=float)
    absorbance = df["Absorbance"].to_numpy(dtype=float)

    energy = 1239.841984 / wavelength
    alpha = 2.303 * absorbance / path_length_cm

    # Negative background-corrected absorbance cannot contribute to the
    # positive absorption edge in a Tauc construction.
    alpha_hv = np.clip(alpha * energy, 0.0, None)

    if transition == "direct":
        y = alpha_hv ** 2
    elif transition == "indirect":
        y = np.sqrt(alpha_hv)
    else:
        raise ValueError("transition must be direct or indirect")

    out = pd.DataFrame({
        "Energy_eV": energy,
        "Tauc_y": y,
        "Alpha_cm-1": alpha,
        "Absorbance": absorbance,
        "Wavelength_nm": wavelength,
    })

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out.sort_values("Energy_eV").reset_index(drop=True)


def linear_fit(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept

    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    eg = -intercept / slope if slope != 0 else np.nan

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "R2": float(r2),
        "Eg_eV": float(eg),
    }


def manual_tauc_fit(tauc_df, fit_range):
    lo, hi = fit_range
    fit_df = tauc_df[
        (tauc_df["Energy_eV"] >= lo) &
        (tauc_df["Energy_eV"] <= hi)
    ].copy()

    if len(fit_df) < 3:
        raise ValueError("Too few points inside --fit-range.")

    result = linear_fit(
        fit_df["Energy_eV"].to_numpy(),
        fit_df["Tauc_y"].to_numpy(),
    )

    if result["slope"] <= 0:
        raise ValueError("Selected Tauc region has a non-positive slope.")

    result["fit_df"] = fit_df
    result["method"] = "manual"
    result["fit_min_eV"] = float(fit_df["Energy_eV"].min())
    result["fit_max_eV"] = float(fit_df["Energy_eV"].max())
    return result


def cluster_candidates(candidates, gap_separation=0.35):
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda c: c["Eg_eV"])
    clusters = [[ordered[0]]]

    for item in ordered[1:]:
        if item["Eg_eV"] - clusters[-1][-1]["Eg_eV"] <= gap_separation:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    return clusters


def auto_tauc_fit(tauc_df, min_r2=0.99):
    """
    Detect the lowest robust linear Tauc edge.

    Strategy:
      - scan multiple contiguous energy windows
      - require a positive, mostly monotonic rise
      - avoid the near-zero baseline and the upper strong-transition region
      - require high linearity
      - extrapolate each candidate to y=0
      - group candidates by Eg and select the lowest well-populated edge cluster
      - within that cluster, select the highest-R^2 fit

    This deliberately favors the lowest reproducible absorption edge rather
    than a higher-energy transition.
    """
    x = tauc_df["Energy_eV"].to_numpy(dtype=float)
    y = tauc_df["Tauc_y"].to_numpy(dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    if len(x) < 20:
        raise ValueError("Too few valid points for automatic Tauc fitting.")

    ymax = float(np.max(y))
    if ymax <= 0:
        raise ValueError("No positive Tauc signal found.")

    candidates = []
    max_window = min(60, len(x))

    for nwin in range(12, max_window + 1):
        for i in range(0, len(x) - nwin + 1):
            xx = x[i:i + nwin]
            yy = y[i:i + nwin]

            energy_span = float(xx[-1] - xx[0])
            if not (0.12 <= energy_span <= 0.60):
                continue

            y0 = float(yy[0])
            y1 = float(yy[-1])

            # Ignore baseline/noise and strong upper transitions.
            if y0 < 0.003 * ymax:
                continue
            if y1 > 0.25 * ymax:
                continue
            if (y1 - y0) < 0.003 * ymax:
                continue

            monotonic_fraction = float(np.mean(np.diff(yy) >= 0))
            if monotonic_fraction < 0.75:
                continue

            fit = linear_fit(xx, yy)

            if fit["slope"] <= 0:
                continue
            if fit["R2"] < min_r2:
                continue

            eg = fit["Eg_eV"]

            # Extrapolated intercept should lie near and below the fitted edge.
            if not (0.2 < eg <= xx[0] + 0.15):
                continue
            if eg < xx[0] - 1.0:
                continue

            fit.update({
                "fit_min_eV": float(xx[0]),
                "fit_max_eV": float(xx[-1]),
                "n_points": int(nwin),
                "monotonic_fraction": monotonic_fraction,
                "fit_df": pd.DataFrame({
                    "Energy_eV": xx,
                    "Tauc_y": yy,
                }),
            })
            candidates.append(fit)

    if not candidates:
        raise RuntimeError(
            "Automatic Tauc fitting could not identify a robust linear edge. "
            "Use --fit-range Emin,Emax to select the region manually."
        )

    clusters = cluster_candidates(candidates, gap_separation=0.35)

    # Require several independently overlapping windows to support an edge.
    robust_clusters = [c for c in clusters if len(c) >= 5]
    chosen_cluster = robust_clusters[0] if robust_clusters else clusters[0]

    # Highest linearity within the lowest robust edge cluster.
    best = max(
        chosen_cluster,
        key=lambda c: (c["R2"], c["n_points"])
    ).copy()
    best["method"] = "automatic"
    best["candidate_count"] = len(candidates)
    best["cluster_count"] = len(chosen_cluster)

    # Record the centers of all robust gap clusters for diagnostics.
    cluster_summary = []
    for cluster in robust_clusters:
        best_in_cluster = max(cluster, key=lambda c: c["R2"])
        cluster_summary.append({
            "Eg_eV": best_in_cluster["Eg_eV"],
            "R2": best_in_cluster["R2"],
            "fit_min_eV": best_in_cluster["fit_min_eV"],
            "fit_max_eV": best_in_cluster["fit_max_eV"],
            "n_candidates": len(cluster),
        })
    best["edge_clusters"] = cluster_summary

    return best


def plot_tauc(tauc_df, fit_result, sample, transition, out_png, out_pdf):
    publication_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.3))

    ax.plot(
        tauc_df["Energy_eV"],
        tauc_df["Tauc_y"],
        color="black",
        linewidth=1.8,
    )

    x1 = fit_result["fit_min_eV"]
    x2 = fit_result["fit_max_eV"]
    eg = fit_result["Eg_eV"]

    # Extend the regression to the x-axis intercept.
    xfit = np.linspace(max(0.0, eg), x2, 300)
    yfit = fit_result["slope"] * xfit + fit_result["intercept"]

    ax.plot(
        xfit, yfit,
        color="black",
        linestyle="--",
        linewidth=1.5,
    )
    ax.axvline(
        eg,
        color="black",
        linestyle=":",
        linewidth=1.1,
    )

    if transition == "direct":
        ylabel = r"$(\alpha h\nu)^2$ (a.u.)"
    else:
        ylabel = r"$(\alpha h\nu)^{1/2}$ (a.u.)"

    ax.set_xlabel(r"Photon energy(eV)", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.tick_params(top=False, right=False)
    ax.text(
        0.05, 0.95,
        rf"$E_g$ = {eg:.2f} eV",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=14, fontweight="bold",
    )

    # ax.text(
    #     0.97, 0.95, sample,
    #     transform=ax.transAxes,
    #     ha="right", va="top",
    #     fontsize=15, fontweight="bold",
    # )

    style_axes(ax)
    maybe_compact_y(ax)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def parse_fit_range(text):
    if text is None:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError('--fit-range must be "Emin,Emax", e.g. "3.2,3.7"')
    lo, hi = map(float, parts)
    if lo >= hi:
        raise ValueError("fit-range Emin must be lower than Emax.")
    return lo, hi


def main():
    parser = argparse.ArgumentParser(
        description="Publication-ready UV-Vis and automatic Tauc-gap plotting."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--outdir", default=".")

    parser.add_argument(
        "--tauc", action="store_true",
        help="Generate Tauc plot and estimate Eg automatically."
    )
    parser.add_argument(
        "--baseline-zero", action="store_true",
        help=(
            "Shift absorbance so the minimum value becomes zero: "
            "A_shifted = A_raw - min(A_raw). Recommended when the spectrum "
            "contains small negative baseline values."
        )
    )
    parser.add_argument(
        "--transition",
        choices=["direct", "indirect"],
        default="direct",
    )
    parser.add_argument(
        "--path-length", type=float, default=1.0,
        help="Optical path length in cm. Default = 1 cm."
    )
    parser.add_argument(
        "--fit-range", default=None,
        help='Manual Tauc fit range in eV, e.g. "3.2,3.7". Overrides auto fit.'
    )
    parser.add_argument(
        "--min-r2", type=float, default=0.99,
        help="Minimum R^2 for automatic Tauc linear-region detection."
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.path_length <= 0:
        raise ValueError("--path-length must be > 0.")

    sample_safe = "".join(
        ch if ch.isalnum() or ch in "-_" else "_"
        for ch in args.sample
    )

    df = read_uvvis(input_path)

    baseline_shift = 0.0
    if args.baseline_zero:
        df, baseline_shift = baseline_shift_to_zero(df)

    uv_png = outdir / f"{sample_safe}_UVVis_black.png"
    uv_pdf = outdir / f"{sample_safe}_UVVis_black.pdf"
    uv_csv = outdir / f"{sample_safe}_UVVis_data.csv"

    df.to_csv(uv_csv, index=False)
    plot_uvvis(df, args.sample, uv_png, uv_pdf)

    print(f"\nSample: {args.sample}")
    print(f"Input: {input_path}")
    print(f"Wavelength range: {df['Wavelength_nm'].min():.1f}"
          f"–{df['Wavelength_nm'].max():.1f} nm")
    print(f"Points: {len(df)}")
    if args.baseline_zero:
        print(f"Baseline shift applied: {-baseline_shift:+.6g} absorbance units")
        print("Minimum corrected absorbance = 0")
    print("\nSaved UV-Vis:")
    print(uv_png)
    print(uv_pdf)

    if args.tauc:
        if args.sample.strip().lower() in {"ni", "nickel", "nikel"}:
            print(
                "\nWARNING: Tauc band-gap analysis is generally not "
                "appropriate for metallic Ni."
            )

        tauc_df = tauc_transform(
            df,
            transition=args.transition,
            path_length_cm=args.path_length,
        )

        fit_range = parse_fit_range(args.fit_range)

        if fit_range is None:
            fit_result = auto_tauc_fit(
                tauc_df,
                min_r2=args.min_r2,
            )
        else:
            fit_result = manual_tauc_fit(
                tauc_df,
                fit_range,
            )

        tauc_png = outdir / f"{sample_safe}_Tauc_{args.transition}_black.png"
        tauc_pdf = outdir / f"{sample_safe}_Tauc_{args.transition}_black.pdf"
        tauc_csv = outdir / f"{sample_safe}_Tauc_{args.transition}_data.csv"
        summary_txt = outdir / f"{sample_safe}_Tauc_{args.transition}_summary.txt"

        tauc_df.to_csv(tauc_csv, index=False)
        plot_tauc(
            tauc_df,
            fit_result,
            args.sample,
            args.transition,
            tauc_png,
            tauc_pdf,
        )

        with open(summary_txt, "w", encoding="utf-8") as fh:
            fh.write(f"Tauc analysis: {args.sample}\n")
            fh.write("=" * 50 + "\n")
            fh.write(f"Input: {input_path}\n")
            fh.write(f"Transition: {args.transition}\n")
            fh.write(f"Method: {fit_result['method']}\n")
            fh.write(f"Path length: {args.path_length} cm\n")
            fh.write(f"Baseline-zero correction: {args.baseline_zero}\n")
            if args.baseline_zero:
                fh.write(f"Original minimum absorbance: {baseline_shift:.8g}\n")
                fh.write("Correction: A_corrected = A_raw - min(A_raw)\n")
            fh.write(f"Fit range: {fit_result['fit_min_eV']:.6f}"
                     f"–{fit_result['fit_max_eV']:.6f} eV\n")
            fh.write(f"Eg = {fit_result['Eg_eV']:.8f} eV\n")
            fh.write(f"R^2 = {fit_result['R2']:.8f}\n")
            fh.write(f"Slope = {fit_result['slope']:.8g}\n")
            fh.write(f"Intercept = {fit_result['intercept']:.8g}\n")

            if fit_result["method"] == "automatic":
                fh.write("\nDetected robust edge clusters:\n")
                for i, edge in enumerate(fit_result["edge_clusters"], 1):
                    fh.write(
                        f"  Edge {i}: Eg={edge['Eg_eV']:.6f} eV, "
                        f"R2={edge['R2']:.6f}, "
                        f"fit={edge['fit_min_eV']:.4f}"
                        f"–{edge['fit_max_eV']:.4f} eV, "
                        f"candidate windows={edge['n_candidates']}\n"
                    )

        print("\nTauc band-gap result:")
        print(f"Method = {fit_result['method']}")
        print(f"Fit range = {fit_result['fit_min_eV']:.4f}"
              f"–{fit_result['fit_max_eV']:.4f} eV")
        print(f"Eg = {fit_result['Eg_eV']:.4f} eV")
        print(f"R^2 = {fit_result['R2']:.6f}")

        if fit_result["method"] == "automatic":
            print("\nOther robust Tauc edge clusters detected:")
            for edge in fit_result["edge_clusters"]:
                print(
                    f"  Eg ~ {edge['Eg_eV']:.3f} eV "
                    f"(R^2={edge['R2']:.4f}; "
                    f"{edge['fit_min_eV']:.3f}–{edge['fit_max_eV']:.3f} eV)"
                )

        print("\nSaved Tauc:")
        print(tauc_png)
        print(tauc_pdf)
        print(tauc_csv)
        print(summary_txt)


if __name__ == "__main__":
    main()

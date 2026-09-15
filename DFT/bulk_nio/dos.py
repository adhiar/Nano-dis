#!/usr/bin/env python3
"""
QE DOS/PDOS + band plotter, with energies referenced to EF = 0 eV.

Expected files in the current directory (defaults):
  bulk_pdos.pdos_tot
  bulk_pdos.pdos_atm#...(Ni...)_wfc#...(d)
  bulk_pdos.pdos_atm#...(O...)_wfc#...(p)
  nio_band.out
  nio_n.out / nio.out  (used only to detect EF if available)

Outputs:
  dos_pdos_fermi.png
  band_fermi.png

Examples:
  python3 plot_qe_dos_band.py
  python3 plot_qe_dos_band.py --fermi 18.7498
  python3 plot_qe_dos_band.py --emin -8 --emax 8
"""

from pathlib import Path
import argparse
import re
import sys
import numpy as np
import matplotlib.pyplot as plt


def read_numeric_table(path):
    """Load a whitespace-separated QE data file, ignoring # comments."""
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals = []
            ok = True
            for x in s.split():
                try:
                    vals.append(float(x))
                except ValueError:
                    ok = False
                    break
            if ok and vals:
                rows.append(vals)
    if not rows:
        raise RuntimeError(f"No numeric data found in {path}")
    ncol = min(len(r) for r in rows)
    return np.array([r[:ncol] for r in rows], dtype=float)


def detect_fermi(files):
    """
    Try, in order:
      1. 'the Fermi energy is ... ev'
      2. midgap from 'highest occupied, lowest unoccupied level'
      3. highest occupied level as fallback reference
    Returns (energy, description).
    """
    fermi_re = re.compile(
        r"(?:the\s+)?Fermi\s+energy\s+is\s+([-+0-9.Ee]+)\s*ev",
        re.I,
    )
    gap_re = re.compile(
        r"highest\s+occupied.*?lowest\s+unoccupied.*?\(ev\)\s*:\s*"
        r"([-+0-9.Ee]+)\s+([-+0-9.Ee]+)",
        re.I,
    )
    ho_re = re.compile(
        r"highest\s+occupied\s+level.*?\(ev\)\s*:\s*([-+0-9.Ee]+)",
        re.I,
    )

    texts = []
    for p in files:
        p = Path(p)
        if p.exists():
            try:
                texts.append((p, p.read_text(errors="ignore")))
            except Exception:
                pass

    for p, txt in texts:
        vals = fermi_re.findall(txt)
        if vals:
            return float(vals[-1]), f"Fermi energy from {p.name}"

    for p, txt in texts:
        vals = gap_re.findall(txt)
        if vals:
            vbm, cbm = map(float, vals[-1])
            return 0.5 * (vbm + cbm), (
                f"mid-gap reference from VBM/CBM in {p.name} "
                f"(VBM={vbm:.6f}, CBM={cbm:.6f} eV)"
            )

    for p, txt in texts:
        vals = ho_re.findall(txt)
        if vals:
            vbm = float(vals[-1])
            return vbm, (
                f"VBM used as zero-reference from {p.name}; "
                "no explicit Fermi energy was found"
            )

    return None, None



def detect_gap_from_output(path):
    """
    Try to read an explicit QE VBM/CBM pair from output:
      highest occupied, lowest unoccupied level (ev): ...
    Returns (vbm, cbm, gap, source) or None.
    """
    path = Path(path)
    if not path.exists():
        return None

    txt = path.read_text(errors="ignore")
    gap_re = re.compile(
        r"highest\s+occupied.*?lowest\s+unoccupied.*?\(ev\)\s*:\s*"
        r"([-+0-9.Ee]+)\s+([-+0-9.Ee]+)",
        re.I,
    )
    vals = gap_re.findall(txt)
    if not vals:
        return None

    vbm, cbm = map(float, vals[-1])
    return vbm, cbm, max(0.0, cbm - vbm), f"explicit VBM/CBM from {path.name}"


def calculate_gap_from_eigenvalues(eigs, ef, tol=1e-6):
    """
    Determine VBM/CBM from all eigenvalues relative to an energy reference EF.
    For an insulator with EF inside the gap:
        VBM = highest E <= EF
        CBM = lowest  E >  EF

    Returns (vbm_abs, cbm_abs, gap, is_metallic).
    """
    flat = np.asarray(eigs, dtype=float).ravel()
    below = flat[flat <= ef + tol]
    above = flat[flat > ef + tol]

    if below.size == 0 or above.size == 0:
        return None

    vbm = float(np.max(below))
    cbm = float(np.min(above))
    gap = cbm - vbm

    # If states effectively touch the reference, treat as metallic/zero-gap.
    metallic = gap <= 1e-4
    if metallic:
        gap = 0.0

    return vbm, cbm, gap, metallic


def interp_to(x_src, y_src, x_target):
    if len(x_src) == len(x_target) and np.allclose(x_src, x_target, atol=1e-8):
        return y_src
    return np.interp(x_target, x_src, y_src, left=0.0, right=0.0)


def orbital_ldos(files, target_energy):
    """
    QE spin-polarized atomic PDOS files normally have:
      E  LDOS_up  LDOS_down  [resolved m components...]
    We use LDOS_up/down (columns 2/3) so all m components of that orbital
    are included exactly once.
    """
    up = np.zeros_like(target_energy)
    dn = np.zeros_like(target_energy)
    used = []
    for p in files:
        arr = read_numeric_table(p)
        if arr.shape[1] < 3:
            print(f"Warning: skipping {p}, expected >=3 columns", file=sys.stderr)
            continue
        e = arr[:, 0]
        up += interp_to(e, arr[:, 1], target_energy)
        dn += interp_to(e, arr[:, 2], target_energy)
        used.append(p.name)
    return up, dn, used


def plot_dos(prefix, ef, emin, emax, outname):
    total_file = Path(f"{prefix}.pdos_tot")
    if not total_file.exists():
        raise FileNotFoundError(f"Missing {total_file}")

    total = read_numeric_table(total_file)
    if total.shape[1] < 3:
        raise RuntimeError(
            f"{total_file} has {total.shape[1]} columns; "
            "spin-resolved DOS requires at least E, DOS_up, DOS_down."
        )

    energy_abs = total[:, 0]
    energy = energy_abs - ef
    dos_up = total[:, 1]
    dos_dn = total[:, 2]

    pdos_files = list(Path(".").glob(f"{prefix}.pdos_atm#*"))

    ni_d_files = sorted(
        p for p in pdos_files
        if re.search(r"\(Ni[^)]*\).*_wfc#.*\(d\)", p.name, re.I)
    )
    o_p_files = sorted(
        p for p in pdos_files
        if re.search(r"\(O[^)]*\).*_wfc#.*\(p\)", p.name, re.I)
    )

    ni_d_up, ni_d_dn, ni_used = orbital_ldos(ni_d_files, energy_abs)
    o_p_up, o_p_dn, o_used = orbital_ldos(o_p_files, energy_abs)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    ax.plot(energy, dos_up, label="Total DOS ↑")
    ax.plot(energy, -dos_dn, label="Total DOS ↓")

    if ni_used:
        ax.plot(energy, ni_d_up, label="Ni 3d ↑")
        ax.plot(energy, -ni_d_dn, label="Ni 3d ↓")

    if o_used:
        ax.plot(energy, o_p_up, label="O 2p ↑")
        ax.plot(energy, -o_p_dn, label="O 2p ↓")

    ax.axvline(0.0, linestyle="--", linewidth=1.0, label=r"$E_F$")
    ax.axhline(0.0, linewidth=0.8)

    ax.set_xlim(emin, emax)
    ax.set_xlabel(r"$E-E_F$ (eV)")
    ax.set_ylabel("DOS / PDOS (states/eV)")
    ax.set_title("NiO DOS / PDOS")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(outname, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved {outname}")
    print(f"  Ni-d files summed: {len(ni_used)}")
    print(f"  O-p files summed:  {len(o_used)}")


def parse_pw_bands(path):
    """
    Parse eigenvalues from a QE pw.x bands output.
    Returns kpoints (Nk,3), eigs (Nk,Nb).
    """
    text = Path(path).read_text(errors="ignore").splitlines()

    nbnd = None
    nbnd_re = re.compile(r"number of Kohn-Sham states\s*=\s*(\d+)", re.I)
    for line in text:
        m = nbnd_re.search(line)
        if m:
            nbnd = int(m.group(1))
            break

    k_re = re.compile(
        r"k\s*=\s*([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)"
    )

    kpts = []
    bands = []
    i = 0
    while i < len(text):
        line = text[i]
        if "bands (ev)" in line.lower():
            m = k_re.search(line)
            if not m:
                # sometimes k= line precedes the "bands (ev)" line
                for back in range(max(0, i-3), i+1):
                    m = k_re.search(text[back])
                    if m:
                        break
            if not m:
                i += 1
                continue

            kpts.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])

            vals = []
            j = i + 1
            while j < len(text):
                s = text[j].strip()

                if "occupation numbers" in s.lower():
                    break
                if "bands (ev)" in s.lower() or re.search(r"\bk\s*=", s):
                    break

                nums = []
                for tok in s.split():
                    try:
                        nums.append(float(tok))
                    except ValueError:
                        nums = []
                        break

                if nums:
                    vals.extend(nums)
                    if nbnd is not None and len(vals) >= nbnd:
                        vals = vals[:nbnd]
                        break
                elif vals and s == "":
                    break

                j += 1

            if vals:
                bands.append(vals)
            else:
                kpts.pop()

            i = j
        else:
            i += 1

    if not bands:
        raise RuntimeError(
            f"No 'bands (ev)' blocks found in {path}. "
            "This script expects the output of pw.x with calculation='bands' "
            "or an SCF/NSCF output that prints band eigenvalues."
        )

    nmin = min(len(b) for b in bands)
    eigs = np.array([b[:nmin] for b in bands], dtype=float)
    kpts = np.array(kpts[:len(eigs)], dtype=float)

    return kpts, eigs


def kpath_distance(kpts):
    """
    Cumulative distance in the coordinates printed by pw.x.
    This is adequate for plotting sequence/path shape.
    For publication-quality reciprocal-space distances, use the reciprocal
    lattice metric or bands.x output.
    """
    if len(kpts) == 1:
        return np.array([0.0])
    dk = np.linalg.norm(np.diff(kpts, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(dk)))


def plot_bands(band_out, ef, emin, emax, outname):
    kpts, eigs = parse_pw_bands(band_out)
    x = kpath_distance(kpts)
    shifted = eigs - ef

    # Prefer explicit VBM/CBM printed by QE.
    explicit = detect_gap_from_output(band_out)
    if explicit is not None:
        vbm, cbm, gap, gap_source = explicit
        metallic = gap <= 1e-4
    else:
        calc = calculate_gap_from_eigenvalues(eigs, ef)
        if calc is None:
            vbm = cbm = gap = None
            metallic = None
            gap_source = "could not determine gap"
        else:
            vbm, cbm, gap, metallic = calc
            gap_source = "computed from band eigenvalues around the energy reference"

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for ib in range(shifted.shape[1]):
        ax.plot(x, shifted[:, ib], linewidth=0.9)

    ax.axhline(0.0, linestyle="--", linewidth=1.0)

    if gap is not None:
        vbm_rel = vbm - ef
        cbm_rel = cbm - ef
        ax.axhline(vbm_rel, linestyle=":", linewidth=0.9)
        ax.axhline(cbm_rel, linestyle=":", linewidth=0.9)

        if metallic:
            gap_label = "Metallic / zero gap"
        else:
            gap_label = f"$E_g$ = {gap:.3f} eV"

        ax.text(
            0.02, 0.97, gap_label,
            transform=ax.transAxes,
            va="top", ha="left",
        )

    ax.set_ylim(emin, emax)
    ax.set_xlim(x[0], x[-1] if len(x) > 1 else 1.0)
    ax.set_xlabel("k-path")
    ax.set_ylabel(r"$E-E_F$ (eV)")
    ax.set_title("NiO band structure")
    fig.tight_layout()
    fig.savefig(outname, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved {outname}")
    print(f"  k-points parsed: {len(kpts)}")
    print(f"  bands parsed:    {eigs.shape[1]}")

    if gap is None:
        print("  Band gap: could not be determined automatically")
    else:
        print(f"  VBM = {vbm:.6f} eV  ({vbm-ef:+.6f} eV relative to E_ref)")
        print(f"  CBM = {cbm:.6f} eV  ({cbm-ef:+.6f} eV relative to E_ref)")
        print(f"  Band gap Eg = {gap:.6f} eV")
        print(f"  Gap method: {gap_source}")
        if metallic:
            print("  Classification: metallic / zero-gap within tolerance")
        else:
            print("  Classification: finite-gap system")

    return {
        "vbm_eV": vbm,
        "cbm_eV": cbm,
        "gap_eV": gap,
        "metallic": metallic,
        "source": gap_source,
    }



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="bulk_pdos",
                    help="projwfc filpdos prefix (default: bulk_pdos)")
    ap.add_argument("--band-out", default="nio_band.out",
                    help="pw.x band output (default: nio_band.out)")
    ap.add_argument("--fermi", type=float, default=None,
                    help="Fermi energy in eV. If omitted, detect from QE outputs.")
    ap.add_argument("--emin", type=float, default=-8.0)
    ap.add_argument("--emax", type=float, default=8.0)
    ap.add_argument("--dos-out", default="dos_pdos_fermi.png")
    ap.add_argument("--band-fig", default="band_fermi.png")
    args = ap.parse_args()

    ef = args.fermi
    source = "command line"
    if ef is None:
        ef, source = detect_fermi([
            args.band_out,
            "nio_n.out",
            "nio.out",
            "nio_band.out",
        ])

    if ef is None:
        raise RuntimeError(
            "Could not determine EF automatically. "
            "Run again with e.g. --fermi 18.7498"
        )

    print(f"Energy reference = {ef:.8f} eV ({source})")
    print("All plotted energies are shifted as E - E_ref, so E_ref = 0 eV.")

    plot_dos(args.prefix, ef, args.emin, args.emax, args.dos_out)

    if Path(args.band_out).exists():
        gap_info = plot_bands(args.band_out, ef, args.emin, args.emax, args.band_fig)

        summary = Path("band_gap_summary.txt")
        with summary.open("w") as f:
            f.write(f"Energy reference (eV): {ef:.8f}\n")
            f.write(f"Reference source: {source}\n")
            if gap_info["gap_eV"] is None:
                f.write("Band gap: could not be determined automatically\n")
            else:
                f.write(f"VBM (eV): {gap_info['vbm_eV']:.8f}\n")
                f.write(f"CBM (eV): {gap_info['cbm_eV']:.8f}\n")
                f.write(f"Band gap Eg (eV): {gap_info['gap_eV']:.8f}\n")
                f.write(f"Method: {gap_info['source']}\n")
                f.write(f"Metallic/zero-gap: {gap_info['metallic']}\n")
        print(f"Saved {summary}")
    else:
        print(f"Warning: {args.band_out} not found; DOS plot only.", file=sys.stderr)


if __name__ == "__main__":
    main()

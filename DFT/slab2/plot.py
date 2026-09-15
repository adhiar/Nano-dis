import argparse, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_table(path):
    data=[]
    for line in Path(path).read_text(errors="ignore").splitlines():
        s=line.strip()
        if not s or s.startswith("#"): 
            continue
        try:
            data.append([float(x) for x in s.split()])
        except ValueError:
            pass
    if not data:
        raise RuntimeError(f"No numeric data in {path}")
    n=min(len(r) for r in data)
    return np.array([r[:n] for r in data],float)

def detect_fermi(files):
    fre=re.compile(r"(?:the\s+)?Fermi\s+energy\s+is\s+([-+0-9.Ee]+)\s*ev",re.I)
    gre=re.compile(r"highest\s+occupied.*?lowest\s+unoccupied.*?\(ev\)\s*:\s*([-+0-9.Ee]+)\s+([-+0-9.Ee]+)",re.I)
    for f in files:
        p=Path(f)
        if p.exists():
            txt=p.read_text(errors="ignore")
            m=fre.findall(txt)
            if m:
                return float(m[-1]),f"Fermi energy from {p.name}"
    for f in files:
        p=Path(f)
        if p.exists():
            txt=p.read_text(errors="ignore")
            m=gre.findall(txt)
            if m:
                v,c=map(float,m[-1])
                return 0.5*(v+c),f"mid-gap from {p.name}"
    return None,None

def interp(x,y,xt):
    return y if len(x)==len(xt) and np.allclose(x,xt) else np.interp(xt,x,y,left=0,right=0)

def orbital_sum(files,E):
    up=np.zeros_like(E); dn=np.zeros_like(E)
    for f in files:
        a=load_table(f)
        if a.shape[1] >= 3:
            up += interp(a[:,0],a[:,1],E)
            dn += interp(a[:,0],a[:,2],E)
    return up,dn

def plot_dos(prefix,ef,emin,emax):
    total=load_table(f"{prefix}.pdos_tot")
    Eabs=total[:,0]; E=Eabs-ef
    dup=total[:,1]; ddn=total[:,2]

    files=list(Path(".").glob(f"{prefix}.pdos_atm#*"))
    nid=sorted(p for p in files if re.search(r"\(Ni[^)]*\).*_wfc#.*\(d\)",p.name,re.I))
    op =sorted(p for p in files if re.search(r"\(O[^)]*\).*_wfc#.*\(p\)",p.name,re.I))
    niu,nidn=orbital_sum(nid,Eabs)
    ou,odn=orbital_sum(op,Eabs)

    fig,ax=plt.subplots(figsize=(7.2,5.2))
    ax.plot(E,dup,label="Total DOS ↑")
    ax.plot(E,-ddn,label="Total DOS ↓")
    if nid:
        ax.plot(E,niu,label="Ni 3d ↑")
        ax.plot(E,-nidn,label="Ni 3d ↓")
    if op:
        ax.plot(E,ou,label="O 2p ↑")
        ax.plot(E,-odn,label="O 2p ↓")
    ax.axvline(0,ls="--",lw=1,label=r"$E_F$")
    ax.axhline(0,lw=.8)
    ax.set_xlim(emin,emax)
    ax.set_xlabel(r"$E-E_F$ (eV)")
    ax.set_ylabel("DOS / PDOS (states/eV)")
    ax.set_title("NiO DOS / PDOS")
    ax.legend(frameon=False,ncol=2,loc="upper right")
    fig.tight_layout()
    fig.savefig("dos_pdos_fermi.png",dpi=300,bbox_inches="tight")
    print("Saved dos_pdos_fermi.png")
    plt.show()

def read_gnu(path):
    bands=[]; cur=[]
    for line in Path(path).read_text(errors="ignore").splitlines():
        s=line.strip()
        if not s:
            if cur:
                bands.append(np.array(cur,float)); cur=[]
            continue
        if s.startswith("#"): 
            continue
        sp=s.split()
        if len(sp)>=2:
            try:
                cur.append([float(sp[0]),float(sp[1])])
            except ValueError:
                pass
    if cur: bands.append(np.array(cur,float))
    if not bands:
        raise RuntimeError(f"No bands parsed from {path}")
    return bands

def spin_gap(bands,ef,tol=1e-5):
    # If any continuous band crosses EF, spin channel is metallic.
    for b in bands:
        r=b[:,1]-ef
        if r.min() < -tol and r.max() > tol:
            return dict(vbm=ef,cbm=ef,gap=0.0,metal=True)
    allE=np.concatenate([b[:,1] for b in bands])
    below=allE[allE<=ef+tol]
    above=allE[allE>ef+tol]
    if not len(below) or not len(above):
        return dict(vbm=None,cbm=None,gap=None,metal=None)
    v=float(below.max()); c=float(above.min())
    return dict(vbm=v,cbm=c,gap=max(0,c-v),metal=(c-v)<=1e-4)


def _pretty_klabel(label):
    """Convert common text labels to publication-style symbols."""
    s = label.strip().strip("!#").strip()
    aliases = {
        "G": r"$\Gamma$",
        "GM": r"$\Gamma$",
        "GAMMA": r"$\Gamma$",
        "Γ": r"$\Gamma$",
        "DELTA": r"$\Delta$",
        "SIGMA": r"$\Sigma$",
        "LAMBDA": r"$\Lambda$",
    }
    key = s.upper()
    if key in aliases:
        return aliases[key]
    # Allow labels such as X1, L1, etc.
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", s)
    if m:
        return rf"${m.group(1)}_{{{m.group(2)}}}$"
    return rf"${s}$" if len(s) <= 3 else s


def parse_high_symmetry_from_qe_input(path, kx):
    """
    Parse a QE K_POINTS {crystal_b} or {tpiba_b} path.

    Expected style, e.g.
        K_POINTS {crystal_b}
        6
        0.0 0.0 0.0  30 ! GAMMA
        0.5 0.0 0.5  30 ! X
        ...

    The fourth integer is QE's interpolation count for the segment that starts
    at that node. We map those node indices onto the x-coordinate already
    present in bands.x *.gnu.

    Returns (tick_positions, tick_labels) or (None, None).
    """
    p = Path(path)
    if not p.exists():
        return None, None

    lines = p.read_text(errors="ignore").splitlines()

    start = None
    mode = None
    for i, line in enumerate(lines):
        m = re.search(r"K_POINTS\s*\{\s*([^}]+)\s*\}", line, re.I)
        if m:
            mode = m.group(1).strip().lower()
            start = i
            break

    if start is None or mode not in ("crystal_b", "tpiba_b"):
        return None, None

    j = start + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    if j >= len(lines):
        return None, None

    try:
        nnode = int(lines[j].split()[0])
    except Exception:
        return None, None

    nodes = []
    counts = []
    labels = []

    for line in lines[j+1:j+1+nnode]:
        body, *comment = line.split("!", 1)
        parts = body.split()
        if len(parts) < 4:
            return None, None
        try:
            k = tuple(float(x) for x in parts[:3])
            n = int(float(parts[3]))
        except Exception:
            return None, None
        nodes.append(k)
        counts.append(n)
        label = comment[0].strip() if comment else ""
        labels.append(label)

    # Need labels from comments to know whether a point is Gamma/X/L/etc.
    if not any(labels):
        return None, None

    # QE path convention: node 0 is first point; each previous node's count
    # determines interpolation toward the following node. Build approximate
    # indices in the explicit k list.
    node_indices = [0]
    idx = 0
    for seg in range(nnode - 1):
        nseg = max(int(counts[seg]), 1)
        idx += nseg
        node_indices.append(idx)

    # bands.x GNU files may include endpoints in a slightly different count;
    # scale the logical indices onto the actual first-band k-axis.
    logical_last = node_indices[-1]
    actual_last_index = len(kx) - 1
    if logical_last <= 0 or actual_last_index <= 0:
        return None, None

    scaled_indices = [
        int(round(ii * actual_last_index / logical_last))
        for ii in node_indices
    ]
    scaled_indices = [min(max(ii, 0), actual_last_index) for ii in scaled_indices]

    ticks = [float(kx[ii]) for ii in scaled_indices]
    pretty = [_pretty_klabel(x if x else f"K{i}") for i, x in enumerate(labels)]

    return ticks, pretty


def manual_high_symmetry(kx, labels_csv=None, fractions_csv=None):
    """
    Optional manual mapping:
      --klabels "GAMMA,X,W,K,GAMMA,L"
      --kfractions "0,0.18,0.34,0.50,0.72,1"

    Fractions are along the plotted k-axis, not reciprocal coordinates.
    """
    if not labels_csv:
        return None, None

    labels = [x.strip() for x in labels_csv.split(",") if x.strip()]
    if len(labels) < 2:
        return None, None

    if fractions_csv:
        vals = [float(x) for x in fractions_csv.split(",")]
        if len(vals) != len(labels):
            raise ValueError("--kfractions must have the same number of entries as --klabels")
        if min(vals) < 0 or max(vals) > 1:
            raise ValueError("--kfractions entries must lie between 0 and 1")
    else:
        # Deliberately only a convenience fallback. Equal spacing is not a
        # substitute for the real reciprocal-space segment lengths.
        vals = np.linspace(0.0, 1.0, len(labels))

    xmin, xmax = float(kx[0]), float(kx[-1])
    ticks = [xmin + f * (xmax - xmin) for f in vals]
    pretty = [_pretty_klabel(x) for x in labels]
    return ticks, pretty


def plot_bands(upfile,dnfile,ef,emin,emax,band_input='nio_band.in',klabels=None,kfractions=None):
    up=read_gnu(upfile); dn=read_gnu(dnfile)
    gu=spin_gap(up,ef); gd=spin_gap(dn,ef)

    if gu["metal"] or gd["metal"]:
        gv=gc=ef; gg=0.0
    elif gu["vbm"] is not None and gd["vbm"] is not None:
        gv=max(gu["vbm"],gd["vbm"])
        gc=min(gu["cbm"],gd["cbm"])
        gg=max(0,gc-gv)
    else:
        gv=gc=gg=None

    fig,ax=plt.subplots(figsize=(7.2,5.4))
    for i,b in enumerate(up):
        ax.plot(b[:,0],b[:,1]-ef,color="black",ls="-",lw=1.0,
                label="Spin up" if i==0 else None)
    for i,b in enumerate(dn):
        ax.plot(b[:,0],b[:,1]-ef,color="black",ls="--",lw=1.0,
                label="Spin down" if i==0 else None)

    ax.axhline(0,color="black",ls=":",lw=1,label=r"$E_F$")
    if gg is not None and gg>1e-4:
        ax.axhline(gv-ef,color="black",ls=":",lw=.7)
        ax.axhline(gc-ef,color="black",ls=":",lw=.7)
        text=rf"$E_g$ = {gg:.3f} eV"
    elif gg is not None:
        text="Metallic / zero gap"
    else:
        text="Band gap undetermined"

    ax.text(.02,.97,text,transform=ax.transAxes,va="top")

    # High-symmetry x-axis mapping.
    # Prefer exact path metadata from QE K_POINTS crystal_b/tpiba_b.
    first_k = up[0][:,0]
    ticks, ticklabels = parse_high_symmetry_from_qe_input(band_input, first_k)

    # Optional manual override.
    if klabels:
        ticks, ticklabels = manual_high_symmetry(first_k, klabels, kfractions)

    if ticks is not None:
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        for xpos in ticks:
            ax.axvline(xpos, color="black", linewidth=0.55, alpha=0.45)
        ax.set_xlabel("High-symmetry k-path")
    else:
        ax.set_xlabel("k-path")
        print(
            f"Warning: no valid high-symmetry path labels found in {band_input}. "
            "Your current QE band input must use K_POINTS {crystal_b} or {tpiba_b} "
            "with labels in comments (e.g. ! GAMMA, ! X). "
            "An automatic Monkhorst-Pack grid cannot be honestly relabeled as Γ-X-L."
        )

    xmin=min(min(b[:,0].min() for b in up),min(b[:,0].min() for b in dn))
    xmax=max(max(b[:,0].max() for b in up),max(b[:,0].max() for b in dn))
    ax.set_xlim(xmin,xmax); ax.set_ylim(emin,emax)
    ax.set_ylabel(r"$E-E_F$ (eV)")
    ax.set_title("NiO spin-resolved band structure")
    ax.legend(frameon=False,loc="upper right")
    fig.tight_layout()
    fig.savefig("band_spin_bw_fermi.png",dpi=300,bbox_inches="tight")

    print("\n=== Band-gap analysis ===")
    for name,g in [("Spin up",gu),("Spin down",gd)]:
        print(name)
        print(f"  VBM = {g['vbm']}")
        print(f"  CBM = {g['cbm']}")
        print(f"  Eg  = {g['gap']} eV")
    print("Overall")
    print(f"  VBM = {gv}")
    print(f"  CBM = {gc}")
    print(f"  Eg  = {gg} eV")

    with open("band_gap_summary.txt","w") as f:
        f.write(f"EF = {ef:.8f} eV\n\n")
        f.write(f"Spin up:   VBM={gu['vbm']} CBM={gu['cbm']} Eg={gu['gap']} eV\n")
        f.write(f"Spin down: VBM={gd['vbm']} CBM={gd['cbm']} Eg={gd['gap']} eV\n")
        f.write(f"Overall:   VBM={gv} CBM={gc} Eg={gg} eV\n")

    print("Saved band_spin_bw_fermi.png")
    print("Saved band_gap_summary.txt")
    plt.show()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prefix",default="bulk_pdos")
    p.add_argument("--spin-up",default="Nio_spin_up.bands.gnu")
    p.add_argument("--spin-down",default="Nio_spin_down.bands.gnu")
    p.add_argument("--fermi",type=float,default=None)
    p.add_argument("--emin",type=float,default=-8)
    p.add_argument("--emax",type=float,default=8)
    p.add_argument(
        "--band-input",
        default="nio_band.in",
        help="QE bands input containing K_POINTS {crystal_b/tpiba_b} labels",
    )
    p.add_argument(
        "--klabels",
        default=None,
        help='Manual labels, e.g. "GAMMA,X,W,K,GAMMA,L"',
    )
    p.add_argument(
        "--kfractions",
        default=None,
        help='Manual tick positions as fractions 0..1, e.g. "0,0.18,0.35,0.52,0.76,1"',
    )
    a=p.parse_args()

    ef=a.fermi
    src="command line"
    if ef is None:
        ef,src=detect_fermi(["nio_band.out","nio_up.out","nio_dw.out","nio.out"])
    if ef is None:
        raise RuntimeError("Cannot detect EF. Use e.g. --fermi 18.7498")

    print(f"Energy reference = {ef:.8f} eV ({src})")
    print("All energies are plotted as E - EF.\n")
    plot_dos(a.prefix,ef,a.emin,a.emax)
    plot_bands(a.spin_up,a.spin_down,ef,a.emin,a.emax,a.band_input,a.klabels,a.kfractions)

if __name__=="__main__":
    main()

"""Module 1 (variant study): parametric 2D RC moment frame, 2-3 storeys, non-ductile detailing.

Implements the variant generator promised in the EOSD26 abstract: user-defined ranges of concrete
strength, column reinforcement ratio, column size, bay width and storey height (plus the number of
storeys), sampled by LHS in lhs_variants.py. No floor equalDOF: the beams are axially stiff, and
chaining floor constraints onto the spring constraints breaks the Transformation handler.

Model (same formulation as the 3D prototype, Sect. 3.2 of the paper):
  - elastic beam-columns with zero-length ModIMKPeakOriented springs at both member ends
    (Ibarra et al. 2005), parameters from Haselton et al. (2016), Eqs. 2, 5, 8, 9, 11 (imk.py);
  - n-factor stiffness split, n = 10 (Zareian & Medina 2010, Eq. 7);
  - My from a fiber moment-curvature analysis of each section at first yield of the tension steel,
    with the column axial load from tributary gravity (Concrete01 unconfined, Steel01);
  - 5% Rayleigh damping at T1 and T3: aM on the whole model, bK*(n+1)/n on the elastic elements only
    (Zareian & Medina 2010, Eq. 9);
  - gravity and mass from 1.05D + 0.25L (FEMA P695 2009, Eq. 6-1), P-Delta on columns.

Fixed assumptions (state them in the paper): 3 bays; out-of-plane tributary width 4.5 m;
floor and roof seismic weight q = 7.0 kPa (D + 0.25L incl. slab, finishes, partitions);
fy = 275 MPa (Grade 40, older stock); ties and stirrups 10 mm at 200 mm, bar diameter 16 mm
(s/db = 12.5): the non-ductile detailing; beams 250 mm wide, depth L/12 >= 400 mm, rho = 1.0%
top and bottom; cover to bar centre 50 mm.
Units: kN, m, s (MPa inputs converted).
"""
import json, os
import numpy as np
import openseespy.opensees as ops
from imk import imk_params

NF = 10.0
MATERIAL = os.environ.get("EOSD_IMK", "IMKPeakOriented")  # replacement for the deprecated ModIMKPeakOriented
# Solver settings for run_gm. Defaults reproduce the settings used so far. The Karpathy-loop
# tuner (tuning/) points EOSD_SOLVER_CONFIG at a JSON file to try other values; the result
# check in tuning/evaluate.py guards that collapse capacities do not move.
SOLVER = dict(tol=1e-5, max_iter=50, primary="Newton",
              fallbacks=[["KrylovNewton", 1], ["Newton", 10], ["KrylovNewton", 50]],
              dt_sub=1, t_extra=5.0, stop_drift=0.20, system="UmfPack", eigen="-fullGenLapack")
if os.environ.get("EOSD_SOLVER_CONFIG"):
    SOLVER.update(json.load(open(os.environ["EOSD_SOLVER_CONFIG"])))
N_BAYS = 3
TRIB = 4.5          # m, out-of-plane tributary width
Q_FLOOR = 7.0       # kPa, 1.05D + 0.25L
FY = 275.0          # MPa
ES = 200e3          # MPa
BAR, TIE, S_TIE = 0.016, 0.010, 0.200
COVER = 0.05
G = 9.81
DETAIL = {}


def _layers(b, h, rho, kind):
    """Steel layers (area m2, y from centroid m). Columns: 3/8 top, 3/8 bottom, 1/4 at mid-depth."""
    As = rho * b * h
    y = h / 2 - COVER
    if kind == "col":
        return [(0.375 * As, y), (0.25 * As, 0.0), (0.375 * As, -y)]
    return [(As, y), (As, -y)]


def yield_moment(b, h, rho, fc, fy, P, kind):
    """First-yield moment (kN m) from a fiber moment-curvature analysis under axial load P (kN, +comp)."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    fcK, fyK = fc * 1e3, fy * 1e3
    ops.uniaxialMaterial("Concrete01", 1, -fcK, -0.002, -0.2 * fcK, -0.0035)
    ops.uniaxialMaterial("Steel01", 2, fyK, ES * 1e3, 0.01)
    ops.section("Fiber", 1)
    ops.patch("rect", 1, 20, 1, -h / 2, -b / 2, h / 2, b / 2)
    for A, y in _layers(b, h, rho, kind):
        ops.layer("straight", 2, 2, A / 2, y, -b / 2 + COVER, y, b / 2 - COVER)
    ops.node(1, 0, 0)
    ops.node(2, 0, 0)
    ops.fix(1, 1, 1, 1)
    ops.fix(2, 0, 1, 0)
    ops.element("zeroLengthSection", 1, 1, 2, 1)
    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, -P, 0, 0)
    ops.system("BandGeneral"); ops.numberer("Plain"); ops.constraints("Plain")
    ops.test("NormUnbalance", 1e-8, 50); ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.0); ops.analysis("Static")
    ops.analyze(1)
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    ops.load(2, 0, 0, 1.0)
    ey = fy / ES
    y_t = h / 2 - COVER
    dk = ey / (0.5 * h) / 400
    ops.integrator("DisplacementControl", 2, 3, dk)
    M = 0.0
    for _ in range(2000):
        if ops.analyze(1) != 0:
            break
        k = ops.nodeDisp(2, 3)
        eps_axial = ops.nodeDisp(2, 1)
        if eps_axial - k * (-y_t) >= ey:      # tension steel strain (bottom fibre side)
            M = ops.getLoadFactor(2)
            break
        M = ops.getLoadFactor(2)
    ops.wipe()
    return M


def build(params):
    p = params
    ns = int(round(p.get("n_storey", 3)))
    L = p.get("bay_m", 5.5)
    H = p.get("storey_m", 3.2)
    bc = p.get("col_m", 0.40)
    rho_c = p.get("rho_col", 0.02)
    fc = p.get("fc_MPa", 21.0)
    ec = 4700 * np.sqrt(fc) * 1e3
    hb = max(0.40, round(L / 12 / 0.05) * 0.05)
    bb = 0.25
    w = Q_FLOOR * TRIB                       # kN/m on beams
    m_ext, m_int = w * L / 2 / G, w * L / G  # t per node
    # axial loads per storey (kN): storey j carries floors j+1..ns
    nu_ext = [w * L / 2 * (ns - j) / (bc * bc * fc * 1e3) for j in range(ns)]
    nu_int = [w * L * (ns - j) / (bc * bc * fc * 1e3) for j in range(ns)]
    # yield moments (fresh section models, before the frame is built)
    My_c = {(j, t): yield_moment(bc, bc, rho_c, fc, FY, (nu_ext if t == "e" else nu_int)[j] * bc * bc * fc * 1e3, "col")
            for j in range(ns) for t in "ei"}
    My_b = yield_moment(bb, hb, 0.010, fc, FY, 0.0, "beam")
    rho_sh_c = 2 * np.pi * TIE**2 / 4 / (S_TIE * bc)
    rho_sh_b = 2 * np.pi * TIE**2 / 4 / (S_TIE * bb)
    sn = S_TIE / BAR
    mult = {k: p.get(k + "_mult", 1.0) for k in ("EIy", "theta_cap_pl", "theta_pc", "lambda_prime")}

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    nid = lambda i, j: 100 * j + i + 1
    for j in range(ns + 1):
        for i in range(N_BAYS + 1):
            ops.node(nid(i, j), i * L, j * H)
            if j == 0:
                ops.fix(nid(i, j), 1, 1, 1)
            else:
                m = m_ext if i in (0, N_BAYS) else m_int
                ops.mass(nid(i, j), m, m, 0.0)
    ops.geomTransf("PDelta", 1)
    ops.geomTransf("Linear", 2)
    tag = [1000]
    elastic, beams = [], []

    def member(ni, nj, b, h, nu, rho, rho_sh, My, length, transf):
        A, I = b * h, b * h**3 / 12
        hp = imk_params(nu, rho_sh, sn, rho, fc, My, ec * I, length / 2, mult)
        EIy = hp["EIy"]
        k_mem = 6 * EIy / length
        a_mem = (hp["Mc_My"] - 1) * My / hp["theta_cap_pl"] / k_mem
        k_spr = (NF + 1) * k_mem
        a_spr = a_mem / (1 + NF * (1 - a_mem))
        g = hp["lamda_opensees"]          # E_t = Lamda * My (rotation units)
        ends = []
        for n0 in (ni, nj):
            tag[0] += 1
            new = 100000 + tag[0]
            x, y = ops.nodeCoord(n0)
            ops.node(new, x, y)
            if MATERIAL == "IMKPeakOriented":
                # spring pre-capping rotation that gives the member hardening (Mc - My)/theta_cap_pl
                dp = (hp["Mc_My"] - 1) * My / (a_spr * k_spr)
                ops.uniaxialMaterial("IMKPeakOriented", tag[0], k_spr, dp, hp["theta_pc"], 0.4, My, hp["Mc_My"], 0.0,
                                     dp, hp["theta_pc"], 0.4, My, hp["Mc_My"], 0.0,
                                     g, 0.0, g, g, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
            else:
                ops.uniaxialMaterial("ModIMKPeakOriented", tag[0], k_spr, a_spr, a_spr, My, -My, g, 0.0, g, g,
                                     1.0, 1.0, 1.0, 1.0, hp["theta_cap_pl"], hp["theta_cap_pl"],
                                     hp["theta_pc"], hp["theta_pc"], 0.0, 0.0, 0.4, 0.4, 1.0, 1.0)
            ops.element("zeroLength", tag[0], n0, new, "-mat", tag[0], "-dir", 6)
            ops.equalDOF(n0, new, 1, 2)
            ends.append(new)
        tag[0] += 1
        ops.element("elasticBeamColumn", tag[0], ends[0], ends[1], A, ec, EIy / ec * (NF + 1) / NF, transf)
        elastic.append(tag[0])
        return tag[0], hp

    hps = []
    for j in range(ns):
        for i in range(N_BAYS + 1):
            t = "e" if i in (0, N_BAYS) else "i"
            nu = (nu_ext if t == "e" else nu_int)[j]
            _, hp = member(nid(i, j), nid(i, j + 1), bc, bc, nu, rho_c, rho_sh_c, My_c[(j, t)], H, 1)
            hps.append(("col", j, t, hp))
    for j in range(1, ns + 1):
        for i in range(N_BAYS):
            e, hp = member(nid(i, j), nid(i + 1, j), bb, hb, 0.0, 0.010, rho_sh_b, My_b, L, 2)
            beams.append(e)
        hps.append(("beam", j, "", hp))
    DETAIL.update(hb=hb, My_b=My_b, My_c=My_c, hps=hps, nu_int=nu_int)

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for e in beams:
        ops.eleLoad("-ele", e, "-type", "-beamUniform", -w)
    ops.constraints("Transformation"); ops.numberer("RCM"); ops.system("UmfPack")
    ops.test("NormDispIncr", 1e-8, 50); ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.1); ops.analysis("Static")
    ops.analyze(10)
    ops.loadConst("-time", 0.0)
    lam = ops.eigen(SOLVER["eigen"], min(3, ns)) if SOLVER["eigen"] else ops.eigen(min(3, ns))
    wn = np.sqrt(lam)
    w1, w3 = wn[0], wn[-1]
    aM = 2 * 0.05 * w1 * w3 / (w1 + w3)
    bK = 2 * 0.05 / (w1 + w3)
    ops.rayleigh(aM, 0.0, 0.0, 0.0)
    ops.region(1, "-ele", *elastic, "-rayleigh", 0.0, 0.0, bK * (NF + 1) / NF, 0.0)
    build.ns, build.H = ns, H
    build.T1 = 2 * np.pi / w1
    return build.T1


def run_gm(acc_g, dt, scale, t_extra=None):
    S = SOLVER
    ops.logFile("/dev/null", "-noEcho")
    ns, H = build.ns, build.H
    t_extra = S["t_extra"] if t_extra is None else t_extra
    ops.timeSeries("Path", 2, "-dt", dt, "-values", *(np.asarray(acc_g) * G * scale))
    ops.pattern("UniformExcitation", 2, 1, "-accel", 2)
    ops.wipeAnalysis()
    ops.constraints("Transformation"); ops.numberer("RCM"); ops.system(S["system"])
    ops.test("NormDispIncr", S["tol"], S["max_iter"]); ops.algorithm(S["primary"])
    ops.integrator("Newmark", 0.5, 0.25); ops.analysis("Transient")
    h = dt / S["dt_sub"]
    t_end = len(acc_g) * dt + t_extra
    floors = [100 * j + 1 for j in range(ns + 1)]
    theta = 0.0
    while ops.getTime() < t_end:
        ok = ops.analyze(1, h)
        if ok != 0:
            for alg, sub in S["fallbacks"]:
                ops.algorithm(alg)
                ok = ops.analyze(sub, h / sub)
                if ok == 0:
                    break
            ops.algorithm(S["primary"])
            if ok != 0:
                return theta, False
        u = [ops.nodeDisp(n, 1) for n in floors]
        theta = max(theta, max(abs(u[k + 1] - u[k]) / H for k in range(ns)))
        if theta > S["stop_drift"]:
            return theta, True
    return theta, True


def pushover(params, roof_drift=0.06, steps=600):
    """First-mode-shaped (triangular) pushover; returns (roof drift ratio, base shear / W)."""
    build(params)
    ns, H = build.ns, build.H
    ops.timeSeries("Linear", 3)
    ops.pattern("Plain", 3, 3)
    for j in range(1, ns + 1):
        ops.load(100 * j + 1, j / ns, 0, 0)
    roof = 100 * ns + 1
    dU = roof_drift * ns * H / steps
    ops.wipeAnalysis()
    ops.constraints("Transformation"); ops.numberer("RCM"); ops.system("UmfPack")
    ops.test("NormDispIncr", 1e-6, 100); ops.algorithm("KrylovNewton")
    ops.integrator("DisplacementControl", roof, 1, dU); ops.analysis("Static")
    W = sum(ops.nodeMass(100 * j + i + 1, 1) for j in range(1, ns + 1) for i in range(N_BAYS + 1)) * G
    out = [(0.0, 0.0)]
    for _ in range(steps):
        if ops.analyze(1) != 0:
            break
        # base shear from the load factor: reactions at the fixed nodes miss the equalDOF forces
        V = ops.getLoadFactor(3) * sum(j / ns for j in range(1, ns + 1))
        out.append((ops.nodeDisp(roof, 1) / (ns * H), V / W))
    return np.array(out)

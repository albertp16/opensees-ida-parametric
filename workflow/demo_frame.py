"""TEST ONLY: 2D one-bay, 3-storey RC frame to exercise the IDA runner end to end.

Not the paper's model. Member sizes loosely follow the exposure B schedules
(columns 500x500, girders 350x500, storey 3.15 m, bay 7.0 m); elastic beam-columns
with zero-length ModIMKPeakOriented (IMK, peak-oriented) springs at every member end, parameters from imk.py.
Spring/element stiffness split uses the n-factor scheme (n = 10): spring K = (n+1)*6EIy/L,
element EI = EIy*(n+1)/n. Damping: mass-proportional on the whole model and
stiffness-proportional on the elastic elements only (region), 5% at T1 and T3.
"""
import numpy as np
import openseespy.opensees as ops
from imk import imk_params

H, L, NST = 3.15, 7.0, 3
EC = 4700 * np.sqrt(27.6) * 1e3  # kN/m^2
NF = 10.0


def _sec(b, h):
    return b * h, b * h**3 / 12


def build(params):
    p = params
    fc = p.get("fc_MPa", 27.6)
    ec = 4700 * np.sqrt(fc) * 1e3
    m_floor = 14.0 * p.get("mass_mult", 1.0)  # t per node: ~3650 kN floor weight / 13 frames / 2 nodes
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    for j in range(NST + 1):
        for i, x in enumerate((0.0, L)):
            ops.node(10 * j + i + 1, x, j * H)
            if j > 0:
                ops.mass(10 * j + i + 1, m_floor, m_floor, 0.0)
    ops.fix(1, 1, 1, 1)
    ops.fix(2, 1, 1, 1)
    ops.geomTransf("PDelta", 1)
    ops.geomTransf("Linear", 2)
    mult = {k: p.get(k + "_mult", 1.0) for k in ("EIy", "theta_cap_pl", "theta_pc", "lambda_prime")}
    tag = [1000]
    elastic = []

    def member(ni, nj, b, h, nu, rho, rho_sh, sn, My, length, transf):
        A, I = _sec(b, h)
        hp = imk_params(nu, rho_sh, sn, rho, fc, My, ec * I, length / 2, mult)
        EIy = hp["EIy"]
        k_mem = 6 * EIy / length
        a_mem = (hp["Mc_My"] - 1) * My / hp["theta_cap_pl"] / k_mem
        k_spr = (NF + 1) * k_mem
        a_spr = a_mem / (1 + NF * (1 - a_mem))
        ends = []
        for n0 in (ni, nj):
            tag[0] += 1
            new = 100000 + tag[0]
            x, y = ops.nodeCoord(n0)
            ops.node(new, x, y)
            mt = tag[0]
            g = hp["lamda_opensees"]
            ops.uniaxialMaterial("ModIMKPeakOriented", mt, k_spr, a_spr, a_spr, My, -My, g, 0.0, g, g, 1.0, 1.0, 1.0, 1.0,
                                 hp["theta_cap_pl"], hp["theta_cap_pl"], hp["theta_pc"], hp["theta_pc"],
                                 0.0, 0.0, 0.4, 0.4, 1.0, 1.0)
            ops.element("zeroLength", tag[0], n0, new, "-mat", mt, "-dir", 6)
            ops.equalDOF(n0, new, 1, 2)
            ends.append(new)
        tag[0] += 1
        ops.element("elasticBeamColumn", tag[0], ends[0], ends[1], A, ec, EIy / ec * (NF + 1) / NF, transf)
        elastic.append(tag[0])

    for j in range(NST):
        for i in (1, 2):
            member(10 * j + i, 10 * (j + 1) + i, 0.5, 0.5, 0.06, 0.0236, 0.0063, 4.0, 420.0, H, 1)
        member(10 * (j + 1) + 1, 10 * (j + 1) + 2, 0.35, 0.5, 0.0, 0.012, 0.0045, 4.0, 260.0, L, 2)
    # gravity
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for j in range(1, NST + 1):
        for i in (1, 2):
            ops.load(10 * j + i, 0.0, -m_floor * 9.81, 0.0)
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormDispIncr", 1e-8, 50)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.1)
    ops.analysis("Static")
    ops.analyze(10)
    ops.loadConst("-time", 0.0)
    lam = ops.eigen("-fullGenLapack", 3)
    w = np.sqrt(lam)
    z = 0.05
    aM = 2 * z * w[0] * w[2] / (w[0] + w[2])
    bK = 2 * z / (w[0] + w[2])
    ops.rayleigh(aM, 0.0, 0.0, 0.0)
    ops.region(1, "-ele", *elastic, "-rayleigh", 0.0, 0.0, bK * (NF + 1) / NF, 0.0)
    build.T1 = 2 * np.pi / w[0]
    return build.T1


def run_gm(acc_g, dt, scale, t_extra=5.0):
    ops.timeSeries("Path", 2, "-dt", dt, "-values", *(np.asarray(acc_g) * 9.81 * scale))
    ops.pattern("UniformExcitation", 2, 1, "-accel", 2)
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormDispIncr", 1e-6, 100)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")
    t_end = len(acc_g) * dt + t_extra
    floors = [10 * j + 1 for j in range(NST + 1)]
    theta = 0.0
    while ops.getTime() < t_end:
        ok = ops.analyze(1, dt)
        if ok != 0:
            for alg, sub in (("KrylovNewton", 1), ("Newton", 10), ("KrylovNewton", 50)):
                ops.algorithm(alg)
                ok = ops.analyze(sub, dt / sub)
                if ok == 0:
                    break
            ops.algorithm("Newton")
            if ok != 0:
                return theta, False
        u = [ops.nodeDisp(n, 1) for n in floors]
        theta = max(theta, max(abs(u[k + 1] - u[k]) / H for k in range(NST)))
        if theta > 0.2:
            return theta, True
    return theta, True


def synthetic_record(seed, duration=20.0, dt=0.01):
    """Filtered, enveloped white noise. A test signal only, NOT a real record."""
    r = np.random.default_rng(seed)
    n = int(duration / dt)
    x = r.standard_normal(n)
    k = np.exp(-np.arange(0, 0.5, dt) / 0.08) * np.sin(2 * np.pi * 2.5 * np.arange(0, 0.5, dt))
    x = np.convolve(x, k)[:n]
    t = np.arange(n) * dt
    env = np.minimum(1, t / 2) * np.exp(-np.maximum(0, t - 8) / 4)
    x = x * env
    return 0.3 * x / np.abs(x).max(), dt

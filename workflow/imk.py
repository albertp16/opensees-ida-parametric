"""IMK hinge parameters for RC members from Haselton et al. (2016), ACI Struct. J. 113(6).

Equation numbers refer to that paper. Inputs in MPa, m, rad.
LHS multipliers (optional) scale the median predictions; the paper's limits are
applied after the multiplier (theta_pc <= 0.10, 0.2 <= EIy/EIg <= 0.6).

OpenSees notation (corrected 2026-10-21): ModIMKPeakOriented and IMKPeakOriented take
Lamda with E_t = Lamda * My (OpenSees wiki/docs: "E_t = Lamda_S * M_y", "Ref_Energy = Lamda * Fy"),
so Lamda has units of rotation: Lamda = lambda' * theta_cap_pl (= gamma * theta_y), NOT gamma.
Checked numerically with a cyclic single-spring test (deterioration matches E_t = Lamda * My).
Earlier versions passed gamma = lambda' theta_cap_pl / theta_y, about 1/theta_y (100-200x) too large,
which switched cyclic deterioration off in practice. imk_params() returns lamda_opensees and gamma.
"""
import math


def theta_cap_pl(nu, rho_sh, sn, rho, fc, a_sl=1.0):
    """Plastic rotation to capping, Eq. (5)."""
    return (0.12 * (1 + 0.55 * a_sl) * 0.16**nu * (0.02 + 40 * rho_sh) ** 0.43
            * 0.54 ** (0.01 * fc) * 0.66 ** (0.1 * sn) * 2.27 ** (10 * rho))


def theta_pc(nu, rho_sh):
    """Post-capping rotation, Eq. (8), before the 0.10 rad limit."""
    return 0.76 * 0.031**nu * (0.02 + 40 * rho_sh) ** 1.02


def lambda_prime(nu):
    """Cyclic energy capacity normalised by My*theta_cap_pl, Eq. (11)."""
    return 30 * 0.3**nu


def eiy_ratio(nu):
    """Effective stiffness to yield EIy/EIg, simplified Eq. (2), before limits."""
    return 0.75 * (0.1 + nu) ** 0.8


def imk_params(nu, rho_sh, sn, rho, fc, My, EIg, Ls, mult=None):
    """Median Haselton (2016) parameters times LHS multipliers, with the paper's limits.

    nu: axial load ratio P/(Ag fc) (use 0 for beams); rho_sh = Ash/(s b); sn = s/db;
    rho = As/(b h); fc in MPa; My in kN*m (from section analysis with sampled fc, fy);
    EIg in kN*m^2; Ls shear span in m (e.g. half the clear length in double curvature).
    mult: dict with keys EIy, theta_cap_pl, theta_pc, lambda_prime (default all 1).
    """
    m = {"EIy": 1.0, "theta_cap_pl": 1.0, "theta_pc": 1.0, "lambda_prime": 1.0}
    m.update(mult or {})
    r = min(max(eiy_ratio(nu) * m["EIy"], 0.2), 0.6)
    EIy = r * EIg
    th_y = My * Ls / (3 * EIy)  # cantilever of length Ls
    th_p = theta_cap_pl(nu, rho_sh, sn, rho, fc) * m["theta_cap_pl"]
    th_pc = min(theta_pc(nu, rho_sh) * m["theta_pc"], 0.10)
    lam_p = lambda_prime(nu) * m["lambda_prime"]
    gamma = lam_p * th_p / th_y  # Haselton gamma (E_t = gamma My theta_y), for reference only
    return dict(EIy_EIg=r, EIy=EIy, theta_y=th_y, theta_cap_pl=th_p, theta_pc=th_pc,
                Mc_My=1.13, lambda_prime=lam_p, gamma=gamma, lamda_opensees=lam_p * th_p, My=My)

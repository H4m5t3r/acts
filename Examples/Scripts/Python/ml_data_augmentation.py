"""
Point of closest approach (perigee) of a charged particle helix to a reference point.

Assumptions
-----------
* Uniform magnetic field along +z (solenoid).  For a general field direction,
  rotate (vertex, momentum, reference) into a frame where B || z, solve, rotate back.
* Vacuum, no energy loss  ->  |p| and pT are constants of motion, exact helix.
* Units: positions in mm, momenta in GeV/c, B in Tesla, q in units of e.

Conventions
-----------
* d0 is the signed transverse impact parameter, d0 = -dx*sin(phi) + dy*cos(phi),
  with (dx, dy) = POCA - reference and phi the track azimuth *at* the POCA.
* s is the signed 3D path length from the vertex to the POCA; s < 0 means the
  perigee lies upstream of the given vertex (backwards extrapolation).
* The perigee is defined by minimising the *transverse* distance, which is the
  standard track-fitting definition.  Pass refine_3d=True to instead minimise
  the true 3D distance (no closed form exists in that case).
"""

from numpy.typing import ArrayLike
import numpy as np

import uproot as ur
import awkward as ak
import sys

# turns [GeV/c] / ([e] * [T]) into metres
KAPPA = 0.299792458


def helix_poca(
    vertex: ArrayLike,
    momentum: ArrayLike,
    q: float,
    Bz: float,
    reference: ArrayLike = (0.0, 0.0, 0.0),
    units_per_metre: float = 1000.0,
    refine_3d: bool = False,
):
    """
    Parameters
    ----------
    vertex          : (vx, vy, vz)   production point               [mm]
    momentum        : (px, py, pz)   momentum at the vertex         [GeV/c]
    q               : charge in units of e (electron: -1)
    Bz              : z-component of B                              [T]
    reference       : (rx, ry, rz)   point to be closest to         [mm]
    units_per_metre : 1000 for mm, 100 for cm, 1 for m
    refine_3d       : minimise true 3D distance instead of transverse distance

    Returns
    -------
    dict with keys poca, d0, z0, phi, theta, qOverP, s, radius, centre
    """
    v = np.asarray(vertex, dtype=float)
    p = np.asarray(momentum, dtype=float)
    ref = np.asarray(reference, dtype=float)

    pT = np.hypot(p[0], p[1])
    if pT == 0.0:
        raise ValueError("pT = 0: track is parallel to B, transverse perigee undefined")

    pmag = np.linalg.norm(p)
    phi0 = np.arctan2(p[1], p[0])
    tan_lambda = p[2] / pT  # dz / ds_T
    theta = np.arctan2(pT, p[2])

    # ---------- straight-line limit ----------
    # Minimise the *transverse* distance, consistent with the helix branch below
    # and with the perigee (line-surface) convention: this is the closest
    # approach to the reference *axis*, not to the reference point.
    if q == 0.0 or Bz == 0.0:
        ux, uy = p[0] / pT, p[1] / pT
        sT = -((v[0] - ref[0]) * ux + (v[1] - ref[1]) * uy)
        poca = np.array([v[0] + sT * ux, v[1] + sT * uy, v[2] + sT * p[2] / pT])
        dx, dy = poca[0] - ref[0], poca[1] - ref[1]
        d0 = -dx * np.sin(phi0) + dy * np.cos(phi0)
        return dict(
            poca=poca,
            d0=d0,
            z0=poca[2] - ref[2],
            phi=phi0,
            theta=theta,
            qOverP=q / pmag,
            s=sT * pmag / pT,
            radius=np.inf,
            centre=None,
        )

    # ---------- signed radius of curvature ----------
    rho = pT / (KAPPA * q * Bz) * units_per_metre  # signed, [mm]
    R = abs(rho)
    turn = -np.sign(rho)  # sense of rotation of the position angle psi

    # ---------- transverse circle centre ----------
    # centripetal acceleration q*(v x B) points from the particle to the centre
    C = np.array([v[0] + rho * np.sin(phi0), v[1] - rho * np.cos(phi0)])

    d = ref[:2] - C
    dist = np.linalg.norm(d)
    if dist < 1e-12:
        raise ValueError("reference point lies on the helix axis; POCA is degenerate")

    poca_xy = C + R * d / dist  # radial projection onto the circle

    # ---------- turning angle -> arc length -> z ----------
    psi0 = np.arctan2(v[1] - C[1], v[0] - C[0])
    psi1 = np.arctan2(poca_xy[1] - C[1], poca_xy[0] - C[0])
    dpsi = (psi1 - psi0 + np.pi) % (2.0 * np.pi) - np.pi  # wrap to (-pi, pi]

    sT = turn * R * dpsi  # signed transverse arc length
    z = v[2] + sT * tan_lambda

    # ---------- optional true-3D refinement ----------
    if refine_3d:
        from scipy.optimize import minimize_scalar

        def dist2(t):
            psi = psi0 + turn * t / R
            return (
                (C[0] + R * np.cos(psi) - ref[0]) ** 2
                + (C[1] + R * np.sin(psi) - ref[1]) ** 2
                + (v[2] + t * tan_lambda - ref[2]) ** 2
            )

        half_turn = np.pi * R
        res = minimize_scalar(
            dist2, bracket=(sT - 0.25 * half_turn, sT, sT + 0.25 * half_turn)
        )
        sT = float(res.x)
        psi = psi0 + turn * sT / R
        poca_xy = C + R * np.array([np.cos(psi), np.sin(psi)])
        z = v[2] + sT * tan_lambda

    # ---------- perigee parameters ----------
    phi = phi0 + turn * sT / R  # azimuth of p at the POCA
    phi = (phi + np.pi) % (2.0 * np.pi) - np.pi

    dx, dy = poca_xy[0] - ref[0], poca_xy[1] - ref[1]
    d0 = -dx * np.sin(phi) + dy * np.cos(phi)

    return dict(
        poca=np.array([poca_xy[0], poca_xy[1], z]),
        d0=d0,
        z0=z - ref[2],
        phi=phi,
        theta=theta,
        qOverP=q / pmag,
        s=sT * pmag / pT,
        radius=R,
        centre=C,
    )


# DRAWING STUFF, not relevant for POCA approximation
def helix_point(vertex, momentum, q, Bz, sT, units_per_metre=1000.0):
    """Position after signed transverse arc length sT (used by the self-test)."""
    v = np.asarray(vertex, float)
    p = np.asarray(momentum, float)
    pT = np.hypot(p[0], p[1])
    phi0 = np.arctan2(p[1], p[0])
    rho = pT / (KAPPA * q * Bz) * units_per_metre
    R, turn = abs(rho), -np.sign(rho)
    C = np.array([v[0] + rho * np.sin(phi0), v[1] - rho * np.cos(phi0)])
    psi0 = np.arctan2(v[1] - C[1], v[0] - C[0])
    psi = psi0 + turn * np.asarray(sT) / R
    return np.stack(
        [
            C[0] + R * np.cos(psi),
            C[1] + R * np.sin(psi),
            v[2] + np.asarray(sT) * p[2] / pT,
        ],
        axis=-1,
    )


def readExampleRootData():
    particles_path = "/home/taleiko/Documents/CERN/Doktorsstudier/Program/acts/ml_data/training/mega_data_0/particles.root"
    particles_tree_name = "particles"
    summary_path = "/home/taleiko/Documents/CERN/Doktorsstudier/Program/acts/ml_data/training/mega_data_0/tracksummary.root"
    summary_tree_name = "tracksummary"
    summary_tree = ur.open(summary_path)[summary_tree_name]
    particles_tree = ur.open(particles_path)[particles_tree_name]
    # print(summary_tree.keys())
    # print(particles_tree.keys())
    # print(summary_tree["t_d0"].array())

    summary = summary_tree.arrays(
        ["t_d0", "t_z0", "t_phi", "t_theta", "t_pT"], library="ak"
    )

    summary_fields_to_check = [
        "t_d0",
        "t_z0",
        "t_phi",
        "t_theta",
        "t_pT",
    ]
    masks = []
    for field in summary_fields_to_check:
        nonempty = ak.to_numpy(ak.num(summary[field]) > 0)

        firsts = ak.firsts(summary[field])
        arr = ak.to_numpy(firsts)
        arr = np.asarray(arr).squeeze()
        # finite: numeric finite values (will be False for NaN/Inf)
        finite = np.isfinite(arr)
        combined = np.logical_and(nonempty, finite)
        masks.append(combined)

    particles = particles_tree.arrays(
        ["vx", "vy", "vz", "px", "py", "pz", "q"], library="ak"
    )
    particle_fields_to_check = [
        "vx",
        "vy",
        "vz",
        "px",
        "py",
        "pz",
        "q",
    ]
    for field in particle_fields_to_check:
        nonempty = ak.to_numpy(ak.num(particles[field]) > 0)

        firsts = ak.firsts(particles[field])
        arr = ak.to_numpy(firsts)
        arr = np.asarray(arr).squeeze()
        # finite: numeric finite values (will be False for NaN/Inf)
        finite = np.isfinite(arr)
        combined = np.logical_and(nonempty, finite)
        masks.append(combined)

    combined_mask = np.logical_and.reduce(masks)

    v = {
        "d0": ak.to_numpy(summary["t_d0"][combined_mask][:, 0]),
        "z0": ak.to_numpy(summary["t_z0"][combined_mask][:, 0]),
        "phi": ak.to_numpy(summary["t_phi"][combined_mask][:, 0]),
        "theta": ak.to_numpy(summary["t_theta"][combined_mask][:, 0]),
        "pT": ak.to_numpy(summary["t_pT"][combined_mask][:, 0]),
        "vx": ak.to_numpy(particles["vx"][combined_mask][:, 0]),
        "vy": ak.to_numpy(particles["vy"][combined_mask][:, 0]),
        "vz": ak.to_numpy(particles["vz"][combined_mask][:, 0]),
        "px": ak.to_numpy(particles["px"][combined_mask][:, 0]),
        "py": ak.to_numpy(particles["py"][combined_mask][:, 0]),
        "pz": ak.to_numpy(particles["pz"][combined_mask][:, 0]),
        "q": ak.to_numpy(particles["q"][combined_mask][:, 0]),
    }
    return v


if __name__ == "__main__":
    # v = readExampleRootData()
    # print(v)

    # # SINGLE EVENTS
    # rng = np.random.default_rng(7)
    # vtx = np.array([v["vx"][0], v["vy"][0], v["vz"][0]])
    # pT_ = v["pT"][0]
    # mom = np.array([v["px"][0], v["py"][0], v["pz"][0]])
    # q = v["q"]
    # Bz = 2.0
    ref = np.array([0, 0, 0])

    # out = helix_poca(vtx, mom, q, Bz, ref, refine_3d=False)
    # print(out)
    # R = out["radius"]
    # sys.exit(0)

    # # Claude's test
    # rng = np.random.default_rng(7)
    # worst_xy = worst_z = worst_d0 = worst_dir = 0.0

    # for _ in range(3000):
    #     # vtx = rng.normal(0, 30, 3)
    #     vtx = np.array([v["vx"], v["vy"], v["vz"]])
    #     pT_ = 10 ** rng.uniform(-0.7, 1.5)            # 0.2 .. 30 GeV
    #     ph = rng.uniform(-np.pi, np.pi)
    #     eta = rng.uniform(-3, 3)
    #     # mom = np.array([pT_ * np.cos(ph), pT_ * np.sin(ph), pT_ * np.sinh(eta)])
    #     mom = np.array([v["px"], v["py"], v["pz"]])
    #     # q = rng.choice([-1.0, 1.0])
    #     q = -1.0
    #     # Bz = rng.choice([2.0, -2.0, 4.0])
    #     Bz = -2.0
    #     # ref = rng.normal(0, 20, 3)
    #     ref = np.array([0, 0, 0])

    #     out = helix_poca(vtx, mom, q, Bz, ref, refine_3d=True)
    #     R = out["radius"]

    #     # (1) brute-force scan of the transverse distance over +-1 full turn
    #     coarse = np.linspace(-np.pi * R, np.pi * R, 4001)
    #     pc = helix_point(vtx, mom, q, Bz, coarse)
    #     t0 = coarse[np.argmin(np.hypot(pc[:, 0] - ref[0], pc[:, 1] - ref[1]))]
    #     fine = np.linspace(t0 - 1e-3 * R, t0 + 1e-3 * R, 4001)
    #     pf = helix_point(vtx, mom, q, Bz, fine)
    #     best = pf[np.argmin(np.hypot(pf[:, 0] - ref[0], pf[:, 1] - ref[1]))]

    #     worst_xy = max(worst_xy, np.hypot(*(out["poca"][:2] - best[:2])))
    #     worst_z = max(worst_z, abs(out["poca"][2] - best[2]))

    #     # (2) |d0| must equal the transverse POCA-reference distance
    #     worst_d0 = max(worst_d0, abs(abs(out["d0"]) - np.hypot(out["poca"][0] - ref[0],
    #                                                           out["poca"][1] - ref[1])))

    #     # (3) phi at the POCA must be perpendicular to the radial (POCA-centre) vector
    #     radial = out["poca"][:2] - out["centre"]
    #     tangent = np.array([np.cos(out["phi"]), np.sin(out["phi"])])
    #     worst_dir = max(worst_dir, abs(np.dot(radial / R, tangent)))

    # print(f"max |d_xy| vs brute force  : {worst_xy:.3e} mm")
    # print(f"max |dz|   vs brute force  : {worst_z:.3e} mm")
    # print(f"max |d0| consistency       : {worst_d0:.3e} mm")
    # print(f"max |radial . tangent|     : {worst_dir:.3e}")

    # ex = helix_poca((1.0, -2.0, 5.0), (1.2, 0.9, 1.4), q=-1.0, Bz=2.0,
    #                 reference=(0.0, 0.0, 0.0), refine_3d=True)
    # print("\nexample: 2.05 GeV electron, B = 2 T, reference = origin")
    # for k in ("poca", "d0", "z0", "phi", "theta", "qOverP", "s", "radius"):
    #     print(f"  {k:7s} = {ex[k]}")

    # print("REAL VALUES FROM ACTS:")
    # print("d0:", v["d0"])
    # print("z0:", v["z0"])
    # print("phi:", v["phi"])
    # print("theta:", v["theta"])

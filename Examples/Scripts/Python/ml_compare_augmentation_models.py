"""Compares MLPs trained with ml_start_mlp_training_data_augmentation.py in the
different training modes (aug, noaug, wide) by overlaying their residual
histograms on the same test data.

Each model is evaluated with its own input/output scalers and with the beamspot
at (0, 0, 0), like evaluatePOCAParameterResiduals() in the training script, so
the residuals are w.r.t. the truth parameters in the tracksummary.

The models to compare are given by their .pt file paths, one argument per
mode (at least one). The MLP layer sizes are read from the model files.

The per-track Kalman filter residuals of the same test data (res_*_fit in
tracksummary_kf.root, also fit - truth) are drawn as a reference.

Example:
    python ml_compare_augmentation_models.py --particle muon \
        --aug_model runs/<config>_aug/seed_<seed>/<model>.pt \
        --noaug_model runs/<config>_noaug/seed_<seed>/<model>.pt \
        --wide_model runs/<config>_wide/seed_<seed>/<model>.pt \
        --test_data narrow wide
"""

import os
import argparse

import numpy as np
import torch
import uproot
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import constants as co
from ml_utilities import DataHandler, Evaluator, MlDataset
from ml_trainer import getAugmentedMlpOutputs, NOISE
from ml_regressor_models import MLP

MODES = ["aug", "noaug", "wide"]
MODE_LABELS = {
    "aug": "Beamspot augmentation",
    "noaug": "No augmentation",
    "wide": "Wide vertex data",
}
MODE_COLORS = {"aug": "cornflowerblue", "noaug": "darkorange", "wide": "forestgreen"}

KF_FILE = "tracksummary_kf.root"
KF_TREE = "tracksummary"
# Per-track residuals (fit - truth) of d0, z0, phi, theta, q/p
KF_BRANCHES = [
    "res_eLOC0_fit",
    "res_eLOC1_fit",
    "res_ePHI_fit",
    "res_eTHETA_fit",
    "res_eQOP_fit",
]
KF_LABEL = "Kalman filter"
KF_COLOR = "black"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--particle", type=str, default="muon", choices=["electron", "muon"]
)
# Particle whose training scalers are loaded. Default: same as --particle. Set
# e.g. --particle electron --scaler_particle muon to evaluate muon-trained
# models on electron data with the muon scalers.
parser.add_argument(
    "--scaler_particle", type=str, default=None, choices=["electron", "muon"]
)
# Path to the .pt model file of each mode. Modes without a path are left out
# of the comparison.
for mode in MODES:
    parser.add_argument(
        "--{}_model".format(mode),
        type=str,
        default=None,
        help="model file of the '{}' mode ({})".format(mode, MODE_LABELS[mode]),
    )
# Test data to evaluate on (one figure each): the narrow or the wide vertex
# version of the test directory (mega_data_<test_dir_number>)
parser.add_argument(
    "--test_data",
    type=str,
    nargs="+",
    default=["narrow"],
    choices=["narrow", "wide"],
)
parser.add_argument("--test_dir_number", type=int, default=9)
# Draw the Kalman filter residuals from tracksummary_kf.root as a reference
parser.add_argument("--kalman", type=int, default=1)
# Number of bins: one value for all histograms, or five values, one per
# parameter in the order d0 z0 phi theta q/p
parser.add_argument("--bins", type=int, nargs="+", default=[100])
# Plotted range per parameter: this quantile of |residual|, taken over all
# models, times 1.2
parser.add_argument("--range_quantile", type=float, default=0.99)
# Fixed half-width of the plotted range (-h, h) per parameter, overriding
# --range_quantile: one value for all or five values (d0 z0 phi theta q/p).
# A value <= 0 keeps the automatic range for that parameter.
parser.add_argument("--half_range", type=float, nargs="+", default=None)
parser.add_argument("--log_y", type=int, default=0)
parser.add_argument("--show", type=int, default=0)
# Resolution of the saved figure (pixels = figsize in inches * dpi)
parser.add_argument("--dpi", type=int, default=300)
# Directory for the figures. Default: runs/model_comparison
parser.add_argument("--output_dir", type=str, default=None)
args = parser.parse_args()


def perParameter(values, name):
    """Expand a one- or five-valued option to one value per parameter."""
    if len(values) == 1:
        return list(values) * 5
    if len(values) == 5:
        return list(values)
    parser.error("--{} takes 1 or 5 values, got {}".format(name, len(values)))


BINS = perParameter(args.bins, "bins")
HALF_RANGES = (
    perParameter(args.half_range, "half_range")
    if args.half_range is not None
    else [0.0] * 5
)

MODEL_PATHS = {
    mode: os.path.abspath(os.path.expanduser(getattr(args, "{}_model".format(mode))))
    for mode in MODES
    if getattr(args, "{}_model".format(mode)) is not None
}
if not MODEL_PATHS:
    parser.error(
        "give at least one model path with "
        + ", ".join("--{}_model".format(mode) for mode in MODES)
    )
for mode, path in MODEL_PATHS.items():
    if not os.path.isfile(path):
        parser.error("--{}_model: file {} not found".format(mode, path))

PARTICLE = args.particle
SCALER_PARTICLE = args.scaler_particle or PARTICLE
SIMULATION = "geant4"
DATA_PREFIX = "mega"
ACTS_DIR = co.ACTS_DIR
RUN_DIR = os.path.join(ACTS_DIR, "runs")
COMPARISON_DIR = (
    os.path.abspath(os.path.expanduser(args.output_dir))
    if args.output_dir
    else os.path.join(RUN_DIR, "model_comparison")
)

VARIABLE_NAMES = [r"$d_0$", r"$z_0$", r"$\phi$", r"$\theta$", r"$q/p$"]
LABELS = [
    r"$d_0^{\text{fit}}-d_0^{\text{true}}$ [mm]",
    r"$z_0^{\text{fit}}-z_0^{\text{true}}$ [mm]",
    r"$\phi^{\text{fit}}-\phi^{\text{true}}$ [rad]",
    r"$\theta^{\text{fit}}-\theta^{\text{true}}$ [rad]",
    r"$q/p^{\text{fit}}-q/p^{\text{true}}$ [$\text{GeV}^{-1}$]",
]


def testDataDir(vertex_data):
    return os.path.join(
        ACTS_DIR,
        "ml_data/mega_data_{}/{}/{}/{}/0.0_0.0_0.0/train_100000".format(
            args.test_dir_number, PARTICLE, SIMULATION, vertex_data
        ),
    )


def scalerPaths(mode):
    """Same scaler paths as in the training script for the given mode."""
    prefix = (
        DATA_PREFIX
        if SCALER_PARTICLE == "electron"
        else DATA_PREFIX + "_" + SCALER_PARTICLE
    )
    if mode == "wide":
        prefix += "_wide"
    output_ending = (
        "_output_scaler_augmented.joblib"
        if mode == "aug"
        else "_output_scaler_noaug.joblib"
    )
    return (
        os.path.join(RUN_DIR, prefix + "_input_scaler.joblib"),
        os.path.join(RUN_DIR, prefix + output_ending),
    )


def mlpArchitecture(state_dict):
    """Input size, hidden layer size and number of hidden layers of an MLP from
    ml_regressor_models from its state dict: the Linear layers are the 2D
    weights in net.<index>.weight, the first and last being the input and
    output layers."""
    weights = [
        tensor
        for key, tensor in sorted(
            state_dict.items(), key=lambda item: int(item[0].split(".")[1])
        )
        if key.endswith(".weight") and tensor.dim() == 2
    ]
    input_dim = weights[0].shape[1]
    hidden_dim = weights[0].shape[0]
    n_hidden_layers = len(weights) - 2
    return input_dim, hidden_dim, n_hidden_layers


def createEvaluator(mode):
    input_scaler_path, output_scaler_path = scalerPaths(mode)
    # DataHandler creates and saves new scalers if loading fails, which with no
    # training directories would overwrite the training scalers. Stop instead.
    for path in (input_scaler_path, output_scaler_path):
        if not os.path.exists(path):
            raise SystemExit("ERROR: scaler {} not found".format(path))
    dh = DataHandler(
        [],
        input_scaler_path=input_scaler_path,
        output_scaler_path=output_scaler_path,
        # Must match the semi-axes stored in the output scaler
        augmentation_semi_axes=NOISE if mode == "aug" else np.zeros(3),
    )
    ev = Evaluator(
        dh.outlier_roof,
        dh.getInputScaler(),
        dh.getOutputScaler(),
        dh,
        getAugmentedOutputs=getAugmentedMlpOutputs,
        beamspot=(0.0, 0.0, 0.0),
    )
    return dh, ev


def loadModel(mode, max_seq_len):
    path = MODEL_PATHS[mode]
    print("Loading {} model from {}".format(mode, path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(path, map_location=device)
    input_dim, hidden_dim, n_hidden_layers = mlpArchitecture(state_dict)
    # Coordinates and 3 beamspot coordinates
    expected_input_dim = max_seq_len * 3 + 3
    if input_dim != expected_input_dim:
        raise SystemExit(
            "ERROR: {} has input size {}, expected {} (hits and beamspot)".format(
                path, input_dim, expected_input_dim
            )
        )
    print("  MLP with {} hidden layers of size {}".format(n_hidden_layers, hidden_dim))
    model = MLP(
        input_dim=input_dim,
        output_dim=5,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def computeResiduals(mode, test_data_dir):
    """Residuals (prediction - truth, like evaluatePOCAParameterResiduals())
    in physical units of the given mode's model on the test data."""
    dh, ev = createEvaluator(mode)
    model = loadModel(mode, dh.outlier_roof)
    X_test = dh.readX([test_data_dir])
    y_test = dh.readY([test_data_dir])
    return ev.residualCalculations(
        model, MlDataset(X_test, y_test), is_transformer=False
    )


def loadKalmanResiduals(test_data_dir):
    """Per-track Kalman filter residuals (fit - truth) of the test directory as
    an (n_tracks, 5) array, or None if the file is missing. Tracks without
    fitted parameters are left out."""
    path = os.path.join(test_data_dir, KF_FILE)
    if not os.path.isfile(path):
        print("WARNING: {} not found, no Kalman filter residuals".format(path))
        return None
    with uproot.open(path) as f:
        arrays = f[KF_TREE].arrays(["hasFittedParams"] + KF_BRANCHES, library="np")
    # One entry per event, each holding the vector of tracks of the event
    fitted = np.concatenate(arrays["hasFittedParams"]).astype(bool)
    residuals = np.stack(
        [np.concatenate(arrays[branch]) for branch in KF_BRANCHES], axis=1
    ).astype(np.float64)
    print(
        "Kalman filter: {} of {} tracks with fitted parameters".format(
            fitted.sum(), len(fitted)
        )
    )
    return residuals[fitted]


def printStatistics(residuals_by_label):
    names = ["d0", "z0", "phi", "theta", "q/p"]
    print("\nResidual statistics (mean, RMS, half width of the central 68%):")
    for i, name in enumerate(names):
        print(name)
        for label, residuals in residuals_by_label.items():
            res = residuals[:, i]
            res = res[np.isfinite(res)]
            q16, q84 = np.quantile(res, [0.16, 0.84])
            print(
                "  {:6s} mean {: .4e}  RMS {:.4e}  68% half width {:.4e}".format(
                    label, np.mean(res), np.sqrt(np.mean(res**2)), 0.5 * (q84 - q16)
                )
            )


def comparisonHistograms(residuals_by_mode, title, fig_path, kf_residuals=None):
    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(2, 6, figure=fig)
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]
    # Each entry: residuals, label, histogram style
    series = [
        (
            residuals,
            MODE_LABELS[mode],
            dict(color=MODE_COLORS[mode], alpha=0.7),
        )
        for mode, residuals in residuals_by_mode.items()
    ]
    if kf_residuals is not None:
        series.append((kf_residuals, KF_LABEL, dict(color=KF_COLOR, linestyle="--")))
    for i, ax in enumerate(axes):
        # Common symmetric range for all models and the Kalman filter
        if HALF_RANGES[i] > 0:
            half = HALF_RANGES[i]
        else:
            half = 1.2 * max(
                np.quantile(np.abs(res[np.isfinite(res[:, i]), i]), args.range_quantile)
                for res, _, _ in series
            )
            half = half if half > 0 else 1e-6
        bin_edges = np.linspace(-half, half, BINS[i] + 1)
        # for residuals, label, style in [series[1], series[3]]:
        for residuals, label, style in series:
            res = residuals[:, i]
            res = res[np.isfinite(res)]
            rms = np.sqrt(np.mean(res**2))
            ax.hist(
                res,
                bins=bin_edges,
                histtype="step",
                linewidth=1.3,
                label="{} (RMS {:.3g})".format(label, rms),
                **style,
            )
        ax.set_title(VARIABLE_NAMES[i])
        ax.set_xlabel(LABELS[i])
        ax.set_ylabel("Frequency")
        ax.set_xlim(-half, half)
        # ax.set_xlim(-0.2, 0.2)
        if args.log_y:
            ax.set_yscale("log")
        ax.grid(True)
        ax.legend(fontsize=7)
    axes[0].set_xlim(-0.12, 0.12)
    # axes[0].set_xlim(-0.06, 0.06)
    axes[1].set_xlim(-20, 20)
    # axes[2].set_xlim(-0.2, 0.2)
    axes[3].set_xlim(-0.1, 0.1)
    axes[4].set_xlim(-0.03, 0.03)
    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=args.dpi)
    print("Saved comparison figure to", fig_path)
    if args.show:
        plt.show()
    plt.close(fig)


def main():
    for vertex_data in args.test_data:
        test_data_dir = testDataDir(vertex_data)
        print("\nEvaluating on", test_data_dir)
        residuals_by_mode = {
            mode: computeResiduals(mode, test_data_dir) for mode in MODEL_PATHS
        }
        n_tracks = {len(res) for res in residuals_by_mode.values()}
        if len(n_tracks) != 1:
            print("WARNING: the models were evaluated on different numbers of tracks")
        kf_residuals = loadKalmanResiduals(test_data_dir) if args.kalman else None
        printStatistics(
            residuals_by_mode
            if kf_residuals is None
            else {**residuals_by_mode, "kf": kf_residuals}
        )
        fig_path = os.path.join(
            COMPARISON_DIR,
            "residual_comparison_{}{}_{}_test_{}.png".format(
                PARTICLE,
                (
                    ""
                    if SCALER_PARTICLE == PARTICLE
                    else "_{}_scalers".format(SCALER_PARTICLE)
                ),
                "_".join(MODEL_PATHS),
                vertex_data,
            ),
        )
        comparisonHistograms(
            residuals_by_mode,
            "MLP residuals on {} vertex test data".format(vertex_data),
            fig_path,
            kf_residuals,
        )


if __name__ == "__main__":
    main()

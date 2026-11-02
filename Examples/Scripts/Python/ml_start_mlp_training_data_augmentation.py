import os
import argparse
import constants as co
import time
import wandb

# import data_tools as dt
from ml_utilities import DataHandler, Evaluator
from ml_trainer import (
    ArbitraryBeamspotTrainer,
    getAugmentedMlpOutputs,
    NOISE,
)
from ml_regressor_models import MLP, printModelSummary
import torch
import random
from sklearn.model_selection import train_test_split
import torch.multiprocessing as mp

# Use 'spawn' start method to avoid CUDA context issues with forking
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # if the start method has already been set, ignore
    print("Didn't set mp start method. It has already been set(?)")
    # pass

# For debugging
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000)
# parser.add_argument("--data_reading_batch_size", type=int, default=100000)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--hidden", type=int, default=2)
parser.add_argument("--neurons", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--load_model", type=int, default=0)
parser.add_argument("--mini_data", type=int, default=0)
args = parser.parse_args()

MODEL_TYPE = "MLP"
# NOTE: USERS: Set these variables:
PARTICLE = "electron"
SIMULATION = "geant4"
LEARNING_RATE = args.lr
# Given by ChatGPT for training
# EVENT_BATCH_SIZE = 32
# Given by ChatGPT for inference
# EVENT_BATCH_SIZE = 512
# DATA_READING_BATCH_SIZE = args.data_reading_batch_size
BATCH_SIZE = args.batch_size
N_EPOCHS = args.n_epochs
N_HIDDEN_LAYERS = args.hidden
HIDDEN_DIM = args.neurons
LOAD_PREV_MODEL = bool(args.load_model)
MINI_DATA = bool(args.mini_data)

BASE_DIR = co.BASE_DIR
RUN_DIR = os.path.join(BASE_DIR, "runs")
# This script is assumed to be in acts/Examples/Scripts/Python, with the
# data directories in the ACTS root directory
ACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

MINI_ML_TRAIN_DATA_DIRS = [
    os.path.join(
        ACTS_DIR,
        "mini_data/mini_data_{}/{}/{}/train_1000".format(
            str(num), PARTICLE, SIMULATION
        ),
    )
    for num in range(10)
]
MEGA_ML_TRAIN_DATA_DIRS = [
    os.path.join(
        ACTS_DIR,
        "mega_data/mega_data_{}/{}/{}/train_100000".format(
            str(num), PARTICLE, SIMULATION
        ),
    )
    for num in range(10)
]
MEGA_ML_TEST_DATA_DIR = os.path.join(
    ACTS_DIR, "mega_data/mega_data_test/{}/{}/test".format(PARTICLE, SIMULATION)
)
EXTRA_DATA_DIR = os.path.join(
    ACTS_DIR, "mega_data/mega_data_38/electron/geant4/train_100000"
)
EXTRA_DATA_DIR2 = os.path.join(
    ACTS_DIR, "mega_data/mega_data_49/electron/geant4/train_100000"
)
EXTRA_DATA_DIR3 = os.path.join(
    ACTS_DIR, "mega_data/mega_data_96/electron/geant4/train_100000"
)
EXTRA_DATA_DIR4 = os.path.join(
    ACTS_DIR, "mega_data/mega_data_97/electron/geant4/train_100000"
)
CONFIG_FILE_NAME_EXTENSION = "_{}e_{}h_{}n_{}lr_{}b".format(
    N_EPOCHS, N_HIDDEN_LAYERS, HIDDEN_DIM, LEARNING_RATE, BATCH_SIZE
)
MODEL_FILE_NAME_ENDING = CONFIG_FILE_NAME_EXTENSION + ".pt"

if MINI_DATA:
    FIGURES_DIR = os.path.join(RUN_DIR, "mini_mlp" + CONFIG_FILE_NAME_EXTENSION)
    MODEL_SAVE_PATH = os.path.join(
        RUN_DIR,
        os.path.join(
            "mini_mlp" + CONFIG_FILE_NAME_EXTENSION, "mini_mlp" + MODEL_FILE_NAME_ENDING
        ),
    )
    WANDB_PROJECT_NAME = "Mini MLP SHIPPING"
else:
    FIGURES_DIR = os.path.join(RUN_DIR, "mega_mlp" + CONFIG_FILE_NAME_EXTENSION)
    MODEL_SAVE_PATH = os.path.join(
        RUN_DIR,
        os.path.join(
            "mega_mlp" + CONFIG_FILE_NAME_EXTENSION, "mega_mlp" + MODEL_FILE_NAME_ENDING
        ),
    )
    WANDB_PROJECT_NAME = "Mega MLP SHIPPING"
print(MODEL_SAVE_PATH)
# Shared between runs on the same dataset, since fitting the augmented output
# scaler requires propagating the training tracks. The output scaler is kept
# separate from the non-augmented one, as d0 is much wider with augmentation.
DATA_PREFIX = "mini" if MINI_DATA else "mega"
INPUT_SCALER_PATH = os.path.join(RUN_DIR, DATA_PREFIX + "_input_scaler.joblib")
OUTPUT_SCALER_PATH = os.path.join(
    RUN_DIR, DATA_PREFIX + "_output_scaler_augmented.joblib"
)
# Number of tracks per training directory propagated for the output scaler fit
AUGMENTED_SCALER_TRACKS_PER_DIR = 20000
RES_FIG_TITLE = "MLP: {} hidden layers with size {}".format(
    str(N_HIDDEN_LAYERS), str(HIDDEN_DIM)
)

TRUTH_DISTR_FIG_PATH = os.path.join(FIGURES_DIR, "truth_distributions.png")
FILTER_RES_FIG_PATH = os.path.join(FIGURES_DIR, "filter_residuals.png")
ML_RES_FIG_PATH = os.path.join(
    FIGURES_DIR,
    "prediction_residuals_mlp_{}_{}_{}e_{}h_{}n.png".format(
        PARTICLE, SIMULATION, str(N_EPOCHS), str(N_HIDDEN_LAYERS), str(HIDDEN_DIM)
    ),
)
DISTR_FIG_BASE_PATH = os.path.join(FIGURES_DIR, CONFIG_FILE_NAME_EXTENSION)

START_TIME = time.time()


def trainMlpModel(
    train_data_dirs=MEGA_ML_TRAIN_DATA_DIRS[:8],
    val_data_dir=MEGA_ML_TRAIN_DATA_DIRS[8],
    test_data_dir=MEGA_ML_TRAIN_DATA_DIRS[9],
    model_save_path=MODEL_SAVE_PATH,
):
    figures_dir = FIGURES_DIR

    print("Running MLP training with POCA parameter data")
    print("Using data from directories", train_data_dirs)
    print("MLP details:")
    print("Hidden layers: {}".format(str(N_HIDDEN_LAYERS)))
    print("Hidden layer size: {}".format(str(HIDDEN_DIM)))

    print("Initializing wandb project")
    project_name = "Debugging" if co.TEST_ENV else WANDB_PROJECT_NAME
    # run_name = "MLP_{}h_{}n_{}lr".format(str(N_HIDDEN_LAYERS), str(HIDDEN_DIM), str(LEARNING_RATE))
    run_name = "MLP{}".format(CONFIG_FILE_NAME_EXTENSION)
    # run = wandb.init(project=MEGA_WANDB_PROJECT_NAME, name=run_name, config={
    # run = wandb.init(project="Debugging", name="Debug", config={
    run = wandb.init(
        project=project_name,
        name=run_name,
        config={
            "epochs": N_EPOCHS,
            # "batch_size": 64,
            "learning_rate": LEARNING_RATE,
            "hidden_size": HIDDEN_DIM,
        },
    )
    # NOTE: Skipping finding of longest sequence length and limiting to well behaved sequences
    # max_seq_len = dt.iterativeFindLongestSequence(train_data_dirs)
    # print("INITIAL max_seq_len:", max_seq_len)
    # mlp_output_dim = 5
    # if dt.OUTLIER_ROOF < max_seq_len:
    #     print("Decreasing max seq len from {} to {}".format(max_seq_len, dt.OUTLIER_ROOF))
    #     max_seq_len = dt.OUTLIER_ROOF

    dh = DataHandler(
        train_data_dirs,
        input_scaler_path=INPUT_SCALER_PATH,
        output_scaler_path=OUTPUT_SCALER_PATH,
        augmentation_semi_axes=NOISE,
        augmentation_max_tracks_per_dir=AUGMENTED_SCALER_TRACKS_PER_DIR,
    )
    max_seq_len = dh.outlier_roof
    output_dim = 5

    model = MLP(
        # Coordinates and 3 beamspot coordinates
        input_dim=max_seq_len * 3 + 3,
        output_dim=output_dim,
        hidden_dim=HIDDEN_DIM,
        n_hidden_layers=N_HIDDEN_LAYERS,
    )
    printModelSummary(model)

    trainer = ArbitraryBeamspotTrainer(
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        getModelOutputs=getAugmentedMlpOutputs,
        model_save_path=MODEL_SAVE_PATH,
        max_seq_len=max_seq_len,
        input_scaler=dh.getInputScaler(),
        output_scaler=dh.getOutputScaler(),
    )
    if LOAD_PREV_MODEL:
        print("LOADING SAVED MODEL WITH THE SAME CONFIGURATION AND SKIPPING TRAINING")
        d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_save_path, map_location=d))
        model.eval()
    else:
        # SPLIT OPTION 1: Read all data and split it
        # X_all = dh.readX()
        # y_all = dh.readY()
        # train_val_split = random.randint(0, 9999999)
        # val_test_split = random.randint(0, 9999999)
        # print("Train, val split random_state:", train_val_split)
        # print("Val, test split random_state:", val_test_split)
        # # Using the training data for splits
        # print("Splitting training data into training and validation data")
        # X_train, X_val, y_train, y_val = train_test_split(
        #     X_all, y_all, test_size=0.2, random_state=train_val_split
        # )
        # print("Splitting validation data into validation and test data")
        # X_val, X_test, y_val, y_test = train_test_split(
        #     X_val, y_val, test_size=0.5, random_state=val_test_split
        # )

        # SPLIT OPTION 2: Predetermined directories for different purposes
        print("Train data directories:", train_data_dirs)
        print("Validation data directory:", val_data_dir)
        print("Test data directory:", test_data_dir)
        X_train = dh.readX(train_data_dirs)
        train_particle_truth = dh.readParticleTruth(train_data_dirs)
        X_val = dh.readX([val_data_dir])
        val_particle_truth = dh.readParticleTruth([val_data_dir])
        model = trainer.train(
            model, X_train, train_particle_truth, X_val, val_particle_truth
        )

        model.eval()
        X_test = dh.readX([test_data_dir])
        test_particle_truth = dh.readParticleTruth([test_data_dir])
        trainer.test(model, X_test, test_particle_truth)

    # TODO: REMOVE LATER
    # for dir in [test_data_dir, EXTRA_DATA_DIR2, EXTRA_DATA_DIR2, EXTRA_DATA_DIR3]:
    #     print(dir)
    #     X_test = dh.readX([dir])
    #     y_test = dh.readY([dir])
    #     trainer.test(model, X_test, y_test)

    # print(EXTRA_DATA_DIR2)
    # X_test2 = dh.readX([EXTRA_DATA_DIR2])
    # y_test2 = dh.readY([EXTRA_DATA_DIR2])
    # trainer.test(model, X_test2, y_test2)
    # sys.exit(0)

    ev = Evaluator(max_seq_len, trainer.input_scaler, trainer.output_scaler, dh)

    ev.plotDistributions(
        test_data_dir, model, is_transformer=False, distr_base_path=DISTR_FIG_BASE_PATH
    )
    # ev.correlationPlots(test_data_dir, model, is_transformer=False)

    ev.evaluatePOCAParameterResiduals(
        model, test_data_dir, is_transformer=False, res_fig_path=ML_RES_FIG_PATH
    )

    ev.residualCorrelationHeatmaps(
        model, test_data_dir, is_transformer=False, fig_dir=figures_dir
    )

    file_name_ending = "".join(
        ["_{}_vs_filter".format(MODEL_TYPE), CONFIG_FILE_NAME_EXTENSION, ".png"]
    )
    ev.pocaModelToGSFComparison(
        model,
        test_data_dir,
        is_transformer=False,
        figures_dir=figures_dir,
        file_name_ending=file_name_ending,
    )


if __name__ == "__main__":
    try:
        os.makedirs(FIGURES_DIR)
    except:
        print("Mega figures directory already exists")

    # debug_dirs = [os.path.join(ACTS_DIR, dir) for dir in ["debug_data/debug_data_0/electron/geant4/train_100", "debug_data/debug_data_1/electron/geant4/train_100"]]
    # debug_dirs = [os.path.join(ACTS_DIR, dir) for dir in ["debug_data/debug_data_2/electron/geant4/train_100", "debug_data/debug_data_3/electron/geant4/train_100"]]
    debug_dirs = [
        os.path.join(ACTS_DIR, dir)
        for dir in [
            "debug_data/debug_data_0/electron/geant4/train_200",
            "debug_data/debug_data_2/electron/geant4/train_200",
        ]
    ]
    ten_thousant_dir = "/home/taleiko/Documents/CERN/Technical_Student/Program/acts/ml_output/electron/geant4/train"
    if MINI_DATA:
        data_dirs = MINI_ML_TRAIN_DATA_DIRS
    else:
        data_dirs = MEGA_ML_TRAIN_DATA_DIRS
    if co.TEST_ENV:
        print("Test environment mode. Using a small dataset")
        # trainMegaMlpModel(train_data_dirs=debug_dirs)
        # trainMegaMlpModel(train_data_dirs=[MEGA_ML_TRAIN_DATA_DIRS[0]])
        # trainMegaMlpModel(train_data_dirs=[ten_thousant_dir])
        # trainMlpModel()
        # trainMlpModel(
        #     train_data_dirs=data_dirs[:8],
        #     val_data_dir=data_dirs[8],
        #     test_data_dir=data_dirs[9]
        # )
        trainMlpModel(
            train_data_dirs=debug_dirs,
            val_data_dir=debug_dirs[1],
            test_data_dir=debug_dirs[1],
        )
        # trainMlpModel(
        #     train_data_dirs=data_dirs[:8],
        #     val_data_dir=data_dirs[8],
        #     test_data_dir=data_dirs[9]
        #     # test_data_dir=EXTRA_DATA_DIR
        # )
    else:
        print("Using the real dataset")
        # trainMlpModel()
        trainMlpModel(
            train_data_dirs=data_dirs[:8],
            val_data_dir=data_dirs[8],
            test_data_dir=data_dirs[9],
            # test_data_dir=EXTRA_DATA_DIR
        )
    end_time = time.time()
    elapsed_time = end_time - START_TIME
    mins, secs = divmod(elapsed_time, 60)
    print(f"Total runtime: {int(mins)} min {secs:.2f} sec")

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import simulation_geometry as sg
import uproot as ur
import awkward as ak
import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class MlDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class EarlyStopping:
    def __init__(self, model_save_path, patience=3000, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = model_save_path

        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            # if self.verbose:
            #     print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print("Validation loss decreased, saving model...")
        self.best_model_state = model.state_dict()
        if self.path is not None:
            torch.save(self.best_model_state, self.path)
            # wandb.save(self.path)

    def load_best_model(self, model):
        """Load the best model state into the given model."""
        model.load_state_dict(self.best_model_state)

class DataHandler():
    def __init__(
        self,
        train_data_dirs:   list,
        load_data_scalers: bool = False
    ):
        self.outlier_floor = 8
        # Setting 
        self.outlier_roof = 20
        print("DataHandler outlier floor: {}".format(self.outlier_floor))
        print("DataHandler outlier roof: {}".format(self.outlier_roof))
        self.train_data_dirs = train_data_dirs
        self.poca_columns = ["d0", "z0", "phi", "theta", "q_over_p"]
        if load_data_scalers:
            input_scaler_path = "/home/taleiko/Documents/CERN/Technical_Student/Program/ml_model/input_scaler.pkl"
            output_scaler_path = "/home/taleiko/Documents/CERN/Technical_Student/Program/ml_model/output_scaler.pkl"
            try:
                print("Trying to load data scalers from previous runs...")
                self.input_scaler = self.loadScaler(input_scaler_path)
                self.output_scaler = self.loadScaler(output_scaler_path)
                print("Successfully loaded input and output data scalers.")
            except:
                print("Couldn't load data scalers. Creating new ones instead and saving them.")
                self.input_scaler = self.createInputScaler()
                self.output_scaler = self.createParameterOutputScaler()
                self.saveScaler(self.input_scaler, input_scaler_path)
                self.saveScaler(self.output_scaler, output_scaler_path)
        else:
            self.input_scaler = self.createInputScaler()
            self.output_scaler = self.createParameterOutputScaler()

        


    def loadScaler(self, scaler_path):
        return joblib.load(scaler_path)

    def saveScaler(self, scaler, file_path):
        joblib.dump(scaler, file_path)

    def readX(self, data_dirs=None):
        if data_dirs is None:
            data_dirs = self.train_data_dirs
        print("Reading measurement dfs")
        # m_dfs = dt.createAllMeasurementDfs(data_dirs, no_outliers=True, poca=True)
        m_dfs = [self.createMeasurementDfPOCA(dir) for dir in data_dirs]
        X_all = np.concatenate([self.preprocessX(measurement_df) for measurement_df in m_dfs])
        return X_all

    def readRootTree(self, file_path, tree_name):
        tree = ur.open(file_path)[tree_name]
        return tree

    def createInputOutlierMask(self, measurements_tree):
        event_ids = ak.to_numpy(measurements_tree["event_nr"].array()).squeeze()
        unique_ids, counts = np.unique(event_ids, return_counts=True)
        valid_event_ids = unique_ids[(counts >= self.outlier_floor) & (counts <= self.outlier_roof)]
        non_outlier_mask = np.isin(event_ids, valid_event_ids)
        n_outlier_filtered = event_ids.shape[0] - np.sum(non_outlier_mask)
        print("Number of sequence length outliers in data fits:", n_outlier_filtered)
        # print("Number of removed measurement rows:", event_ids.shape[0] - np.sum(non_outlier_mask), "out of", event_ids.shape[0])
        return non_outlier_mask
    
    def createSuccessfulParticlePOCAIds(self, data_dir=None, tracksummary_tree=None):
        if data_dir is None and tracksummary_tree is None:
            raise ValueError("DataHandler.createSuccessfulD0Z0Ids: data_dir or tracksummary_tree must be provided")
        if tracksummary_tree is None:
            file_path = os.path.join(data_dir, "tracksummary.root")
            tracksummary_tree = self.readRootTree(file_path, "tracksummary")

        # Event ids
        event_ids = ak.to_numpy(tracksummary_tree["event_nr"].array()).squeeze()

        # Default set of truth fields to require as finite values. Include
        # fields used downstream (t_d0, t_z0, t_phi, t_theta, t_p, t_charge).
        # Require both true POCA parameters and the filter/GSF fit outputs to be finite
        fields_to_check = [
            "t_d0", "t_z0", "t_phi", "t_theta", "t_p", "t_charge",
            "eLOC0_fit", "eLOC1_fit", "ePHI_fit", "eTHETA_fit", "eQOP_fit"
        ]
        masks = []
        for field in fields_to_check:
            # Some trees might not contain all fields; if missing, skip the field
            if field not in tracksummary_tree.keys():
                continue
            arr_ak = tracksummary_tree[field].array()
            # nonempty: entries with at least one element (handles awkward missing/empty lists)
            nonempty = np.asarray(ak.to_numpy(ak.num(arr_ak) > 0)).squeeze()
            # Extract a single scalar per entry (first element) to get a regular 1D array
            # This avoids converting jagged subarrays directly to a RegularArray
            firsts = ak.firsts(arr_ak)
            arr = ak.to_numpy(firsts)
            arr = np.asarray(arr).squeeze()
            # finite: numeric finite values (will be False for NaN/Inf)
            finite = np.isfinite(arr)
            # combine both criteria: present and numeric
            combined = np.logical_and(nonempty, finite)
            masks.append(combined)

        # Copilot's fallback code
        # if len(masks) == 0:
        #     # Fallback: if no numeric fields found use presence of eLOC0_fit as before
        #     if "eLOC0_fit" in tracksummary_tree.keys():
        #         eLOC0 = tracksummary_tree["eLOC0_fit"].array()
        #         empty_filter = np.array([bool(x) for x in ak.num(eLOC0)])
        #         successful_ids = ak.to_numpy(tracksummary_tree["event_nr"].array()[empty_filter]).squeeze()
        #         return successful_ids
        #     else:
        #         return event_ids

        combined_mask = np.logical_and.reduce(masks)
        # print("UNSUCCESSFUL POCA RUNS:")
        # print(len(event_ids))
        # print(combined_mask.sum())
        # print(len(event_ids) - combined_mask.sum())
        # sys.exit(0)
        successful_ids = event_ids[combined_mask]
        return successful_ids

    def createNonNanMeasMask(self, measurements_tree, data_dir):
        successful_d0_z0_ids = self.createSuccessfulParticlePOCAIds(data_dir)
        # print(successful_d0_z0_ids)
        id_set = set(successful_d0_z0_ids)
        # print(id_set)
        meas_event_ids = ak.to_numpy(measurements_tree["event_nr"].array()).squeeze()
        non_nan_mask = np.array([x in id_set for x in meas_event_ids])
        # print(list(set(meas_event_ids[non_nan_mask])))
        return non_nan_mask

    def createMeasurementDfPOCA(self, data_dir):
        print("Creating measurement df")
        file_path = os.path.join(data_dir, "root/measurements.root")
        measurements_tree = self.readRootTree(file_path, "measurements")

        non_outlier_mask = self.createInputOutlierMask(measurements_tree)
        non_nan_mask = self.createNonNanMeasMask(measurements_tree, data_dir)
        indices = np.array([non_outlier and non_nan for non_outlier, non_nan in zip(non_outlier_mask, non_nan_mask)])

        global_coords = sg.localToGlobalCoordinateConversion(measurements_tree)
        measurements_df = pd.DataFrame({
            # NOTE: Changing the event nr column name for consistency
            "event_id":  ak.to_numpy(measurements_tree["event_nr"].array()[indices]).squeeze(),
            "volume_id": ak.to_numpy(measurements_tree["volume_id"].array()[indices]).squeeze(),
            "rec_loc0":  ak.to_numpy(measurements_tree["rec_loc0"].array()[indices]).squeeze(),
            "rec_loc1":  ak.to_numpy(measurements_tree["rec_loc1"].array()[indices]).squeeze(),
            "global_x":  global_coords[:,0][indices],
            "global_y":  global_coords[:,1][indices],
            "global_z":  global_coords[:,2][indices]
        })
        # For debugging: comparison with particle_df
        # orig_ids = set(ak.to_numpy(measurements_tree["event_nr"].array()[indices]).squeeze())
        # incl_ids = set(measurements_df.event_id)
        # filtered_out_ids = orig_ids - incl_ids
        # print("Filtered out MEASUREMENT IDs:\n", list(filtered_out_ids))
        return measurements_df

    def findNonOutlierEventIds(self, data_dir):
        file_path = os.path.join(data_dir, "root/measurements.root")
        measurements_tree = self.readRootTree(file_path, "measurements")
        event_ids = ak.to_numpy(measurements_tree["event_nr"].array()).squeeze()
        unique_ids, counts = np.unique(event_ids, return_counts=True)
        valid_event_ids = unique_ids[(counts >= self.outlier_floor) & (counts <= self.outlier_roof)]
        return valid_event_ids
    
    def prepareTrackSummaryDf(self, data_dir):
        file_path = os.path.join(data_dir, "tracksummary.root")
        tracksummary_tree = self.readRootTree(file_path, "tracksummary")

        event_id_array = ak.to_numpy(tracksummary_tree["event_nr"].array()).squeeze()
        non_outlier_event_ids = set(self.findNonOutlierEventIds(data_dir))
        non_outlier_mask = np.array([x in non_outlier_event_ids for x in event_id_array])

        successful_train_d0_z0_ids = self.createSuccessfulParticlePOCAIds(data_dir)
        id_set = set(successful_train_d0_z0_ids)
        non_nan_mask = np.array([x in id_set for x in event_id_array])

        combined_filter = np.array([non_outlier and non_nan for non_outlier, non_nan in zip(non_outlier_mask, non_nan_mask)])

        # print("SEQUENCE OUTLIERS:")
        # print(len(non_outlier_mask))
        # print(len(non_outlier_mask) - non_outlier_mask.sum())
        # print("FAILED GSF RUNS:")
        # print(len(non_nan_mask))
        # print(len(non_nan_mask) - non_nan_mask.sum())
        # print("COMBINED FILTER:")
        # print(len(combined_filter))
        # print(len(combined_filter) - combined_filter.sum())
        # sys.exit(0)

        return (tracksummary_tree, combined_filter)

    def createPOCAParameterDf(self, data_dir):
        tracksummary_tree, combined_filter = self.prepareTrackSummaryDf(data_dir)
        t_q = ak.to_numpy(tracksummary_tree["t_charge"].array()[combined_filter]).squeeze()
        t_p = ak.to_numpy(tracksummary_tree["t_p"].array()[combined_filter]).squeeze()
        tracksummary_df = pd.DataFrame({
            "event_id": ak.to_numpy(tracksummary_tree["event_nr"].array()[combined_filter]).squeeze(),
            # TODO: These are not the vertex coordinates, right?
            "d0":       ak.to_numpy(tracksummary_tree["t_d0"].array()[combined_filter]).squeeze(),
            "z0":       ak.to_numpy(tracksummary_tree["t_z0"].array()[combined_filter]).squeeze(),
            "phi":      ak.to_numpy(tracksummary_tree["t_phi"].array()[combined_filter]).squeeze(),
            "theta":    ak.to_numpy(tracksummary_tree["t_theta"].array()[combined_filter]).squeeze(),
            "q_over_p": t_q / t_p,
        })
        # For debugging: comparison with measurement_df
        # orig_ids = set(ak.to_numpy(tracksummary_tree["event_nr"].array()).squeeze())
        # incl_ids = set(tracksummary_df["event_id"])
        # filtered_out_ids = orig_ids - incl_ids
        # print("Filtered out PARTICLE IDs:\n", list(filtered_out_ids))
        # Copilot before empty and NaN fix in createSuccessfulParticlePOCAIds()
        # Ensure numeric dtype and drop any rows with NaN / Inf in the POCA columns.
        # tracksummary_df[self.poca_columns] = tracksummary_df[self.poca_columns].apply(pd.to_numeric, errors='coerce')
        # tracksummary_df = tracksummary_df.replace([np.inf, -np.inf], np.nan)
        # n_before = len(tracksummary_df)
        # tracksummary_df = tracksummary_df.dropna(subset=self.poca_columns)
        # n_after = len(tracksummary_df)
        
        return tracksummary_df

    def createFilterTrackSummaryDf(self, data_dir):
        tracksummary_tree, combined_filter = self.prepareTrackSummaryDf(data_dir)
        tracksummary_df = pd.DataFrame({
            "event_id":     ak.to_numpy(tracksummary_tree["event_nr"].array())[combined_filter],
            "est_d0":       ak.to_numpy(tracksummary_tree["eLOC0_fit"].array()[combined_filter]).squeeze(),
            "est_z0":       ak.to_numpy(tracksummary_tree["eLOC1_fit"].array()[combined_filter]).squeeze(),
            "est_phi":      ak.to_numpy(tracksummary_tree["ePHI_fit"].array()[combined_filter]).squeeze(),
            "est_theta":    ak.to_numpy(tracksummary_tree["eTHETA_fit"].array()[combined_filter]).squeeze(),
            "est_q_over_p": ak.to_numpy(tracksummary_tree["eQOP_fit"].array()[combined_filter]).squeeze(),
        })
        return tracksummary_df

    def createFilterResidualDf(self, data_dir):
        tracksummary_tree, combined_filter = self.prepareTrackSummaryDf(data_dir)
        tracksummary_df = pd.DataFrame({
            "event_id":         ak.to_numpy(tracksummary_tree["event_nr"].array())[combined_filter],
            "res_gsf_d0":       ak.to_numpy(tracksummary_tree["res_eLOC0_fit"].array()[combined_filter]).squeeze(),
            "res_gsf_z0":       ak.to_numpy(tracksummary_tree["res_eLOC1_fit"].array()[combined_filter]).squeeze(),
            "res_gsf_phi":      ak.to_numpy(tracksummary_tree["res_ePHI_fit"].array()[combined_filter]).squeeze(),
            "res_gsf_theta":    ak.to_numpy(tracksummary_tree["res_eTHETA_fit"].array()[combined_filter]).squeeze(),
            "res_gsf_q_over_p": ak.to_numpy(tracksummary_tree["res_eQOP_fit"].array()[combined_filter]).squeeze(),
        })
        return tracksummary_df

    def readY(self, data_dirs=None):
        if data_dirs is None:
            data_dirs = self.train_data_dirs
        print("Reading parameter dfs")
        # p_dfs = dt.createAllParticleDfs(data_dirs, no_outliers=True, poca=True, parameters=True)
        p_dfs = [self.createPOCAParameterDf(dir) for dir in data_dirs]
        y_all = np.concatenate([self.preprocessY(particle_df) for particle_df in p_dfs])
        return y_all

    def preprocessX(self, measurement_df):
        print("Preprocessing started")
        measurement_df = measurement_df[["event_id", "global_x", "global_y", "global_z"]]
        input_columns_to_normalize = ["global_x", "global_y", "global_z"]
        # measurement_df.loc[:, input_columns_to_normalize] = self.input_scaler.transform(
        #     measurement_df[input_columns_to_normalize]
        # )
        coords = measurement_df[input_columns_to_normalize]
        coords_scaled = self.input_scaler.transform(coords)
        meas_df_scaled = measurement_df.copy()
        meas_df_scaled[input_columns_to_normalize] = coords_scaled
        grouped = meas_df_scaled.groupby("event_id")
        X = np.stack([self.processXInward(group, self.outlier_roof) for event_id, group in grouped])
        return X

    def processXInward(self, group, max_seq_len):
        coords = group[["global_x", "global_y", "global_z"]].values
        # Change the order of the coordinate to go inward instead of outward
        coords = np.flip(coords, axis=0)
        pad_len = max_seq_len - len(coords)
        # Add zeroes after the data
        coords = np.pad(coords, ((0, pad_len), (0, 0)), mode='constant')
        coords = coords.flatten()
        return coords

    def preprocessY(self, particle_df):
        y = particle_df[self.poca_columns]
        y = self.output_scaler.transform(y)
        return y


    def createInputScaler(self):
        print("Creating input scaler")
        input_columns_to_normalize = ["global_x", "global_y", "global_z"]
        input_scaler = StandardScaler()
        for data_dir in self.train_data_dirs:
            measurement_df = self.createMeasurementDfPOCA(data_dir)
            measurement_df = measurement_df[input_columns_to_normalize]
            input_scaler.partial_fit(measurement_df)
        print("Columns used for output scaling:", input_columns_to_normalize)
        return input_scaler

    def createParameterOutputScaler(self):
        print("Creating POCA output scaler")
        output_columns_to_normalize = self.poca_columns
        output_scaler = StandardScaler()
        for particle_df in [self.createPOCAParameterDf(data_dir) for data_dir in self.train_data_dirs]:
            output_scaler.partial_fit(particle_df[output_columns_to_normalize])
        print("Columns used for output scaling:", output_columns_to_normalize)
        return output_scaler
    
    def getInputScaler(self):
        return self.input_scaler
    
    def getOutputScaler(self):
        return self.output_scaler


class Evaluator():
    def __init__(
        self,
        max_seq_len:   int,
        input_scaler:  StandardScaler,
        output_scaler: StandardScaler,
        data_handler:  DataHandler,
    ):
        self.max_seq_len = max_seq_len
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.data_handler = data_handler
        self.poca_columns = ["d0", "z0", "phi", "theta", "q_over_p"]
        self.gsf_columns = ["est_d0", "est_z0", "est_phi", "est_theta", "est_q_over_p"]
        self.titles = ["d₀", "z₀", "φ", "θ", "q/p"]
        # self.labels = ["mm", "mm", "rad", "rad", "1/(GeV/c)"]
        self.labels = ["mm", "mm", "rad", "rad", r"$\text{GeV}^{-1}$"]


    ##################################################
    # Distribution plotting
    ##################################################
    def plotDistributions(self, data_dir, model, is_transformer, distr_base_path):
        # self.plotTruthDistributions(data_dir, distr_base_path)
        # self.plotGsfDistributions(data_dir, distr_base_path)
        self.plotModelOutputDistributions(model, data_dir, is_transformer, distr_base_path)

    def plotTruthDistributions(self, data_dir, distr_base_path):
        poca_df = self.data_handler.createPOCAParameterDf(data_dir)
        # fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(5 * 5, 4))
        fig = plt.figure(figsize=(12, 7))
        # fig = plt.figure(figsize=(15, 9))
        gs = GridSpec(2, 6, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax3 = fig.add_subplot(gs[0, 4:6])
        ax4 = fig.add_subplot(gs[1, 1:3])
        ax5 = fig.add_subplot(gs[1, 3:5])
        axes = [ax1, ax2, ax3, ax4, ax5]
        for i in range(len(self.titles)):
            data = poca_df[self.poca_columns[i]].to_numpy()
            axes[i].hist(data, bins=50, color='skyblue', edgecolor='black')
            axes[i].set_title(self.titles[i])
            axes[i].set_xlabel(self.labels[i])
            axes[i].set_ylabel('Frequency')
            if i == 0:
                axes[i].set_xlim([-0.07, 0.07])
            axes[i].grid(True)
        fig.suptitle("Truth distributions")
        plt.tight_layout()
        plt.savefig(distr_base_path + "_truth_distr.png")
        plt.show()

    def plotGsfDistributions(self, data_dir, distr_base_path):
        filter_df = self.data_handler.createFilterTrackSummaryDf(data_dir)
        # fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(5 * 5, 4))
        fig = plt.figure(figsize=(12, 7))
        gs = GridSpec(2, 6, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax3 = fig.add_subplot(gs[0, 4:6])
        ax4 = fig.add_subplot(gs[1, 1:3])
        ax5 = fig.add_subplot(gs[1, 3:5])
        axes = [ax1, ax2, ax3, ax4, ax5]
        for i in range(1, len(self.titles)):
            data = filter_df[self.gsf_columns[i]].to_numpy()
            axes[i].hist(data, bins=50, color='skyblue', edgecolor='black')
            axes[i].set_title(self.titles[i])
            axes[i].set_xlabel(self.labels[i])
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True)
        data = filter_df[self.gsf_columns[0]].to_numpy()
        axes[0].hist(data, bins=5000, color='skyblue', edgecolor='black')
        axes[0].set_title(self.titles[0])
        axes[0].set_xlabel([self.labels[0]])
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True)
        axes[0].set_xlim([-0.5, 0.5])
        fig.suptitle("GSF output distributions")
        plt.tight_layout()
        plt.savefig(distr_base_path + "_GSF_distr.png")
        plt.show()

    def plotModelOutputDistributions(self, model, test_data_dir, is_transformer, distr_base_path):
        evaluation_dataset = self.createTestDataset(test_data_dir)
        preds = self.makePredictions(
            model,
            evaluation_dataset,
            is_transformer
        )
        # fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(5 * 5, 4))
        fig = plt.figure(figsize=(12, 7))
        gs = GridSpec(2, 6, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax3 = fig.add_subplot(gs[0, 4:6])
        ax4 = fig.add_subplot(gs[1, 1:3])
        ax5 = fig.add_subplot(gs[1, 3:5])
        axes = [ax1, ax2, ax3, ax4, ax5]
        for i in range(len(self.titles)):
            # axes[i].hist(preds[:, i], bins=50, color='skyblue', edgecolor='black')
            # axes[i].hist(preds[:, i], bins=50, color='cornflowerblue', edgecolor='black', alpha=1.0)
            axes[i].hist(preds[:, i], bins=50, color='darkorange', edgecolor='black', alpha=1.0)
            axes[i].set_title(self.titles[i])
            axes[i].set_xlabel(self.labels[i])
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True)
            # axes[i].set_xlim(-0.1, 0.1)
        fig.suptitle("Model output distributions")
        plt.tight_layout()
        if is_transformer:
            plt.savefig(distr_base_path + "_transformer_distr.png")
        else:
            plt.savefig(distr_base_path + "_mlp_distr.png")
        plt.show()

    def createTestDataset(self, test_data_dir):
        test_measurements_df = self.data_handler.createMeasurementDfPOCA(test_data_dir)
        print("Reading targets")
        test_particles_df = self.data_handler.createPOCAParameterDf(test_data_dir)
        X_test = self.data_handler.preprocessX(test_measurements_df)
        y_test = self.data_handler.preprocessY(test_particles_df)
        test_dataset = MlDataset(X_test, y_test)
        return test_dataset

    ##################################################
    # Model residuals
    ##################################################
    def evaluatePOCAParameterResiduals(
        self,
        model,
        test_data_dir,
        is_transformer,
        res_fig_path
    ):
        # print("Reading inputs")
        # test_measurements_df = self.data_handler.createMeasurementDfPOCA(test_data_dir)
        # print("Reading targets")
        # test_particles_df = self.data_handler.createPOCAParameterDf(test_data_dir)
        # X_test = self.data_handler.preprocessX(test_measurements_df)
        # y_test = self.data_handler.preprocessY(test_particles_df)
        print("Reading inputs")
        X_test = self.data_handler.readX([test_data_dir])
        print("Reading targets")
        y_test = self.data_handler.readY([test_data_dir])
        test_dataset = MlDataset(X_test, y_test)
        residuals = self.residualCalculations(model, test_dataset, is_transformer)
        output_dim = residuals.shape[1]
        variable_names = [r"$d_0$", r"$z_0$", r"$\phi$", r"$\theta$", r"$q/p$"]
        labels = [
            r"$d_0^{\text{true}}-d_0^{\text{fit}}$ [mm]",
            r"$z_0^{\text{true}}-z_0^{\text{fit}}$ [mm]",
            r"$\phi^{\text{true}}-\phi^{\text{fit}}$ [rad]",
            r"$\theta^{\text{true}}-\theta^{\text{fit}}$ [rad]",
            r"$q/p^{\text{true}}-q/p^{\text{fit}}$ [$\text{GeV}^{-1}$]"
        ]
        print(labels)
        print("Making histograms")
        if is_transformer:
            res_fig_title = "Residuals with the transformer model"
        else:
            res_fig_title = "Residuals with the MLP model"
        self.residualHistogram1Row(output_dim, residuals, variable_names, is_transformer, labels, res_fig_path, res_fig_title)

    def residualHistogram1Row(
        self,
        n_residuals,
        residuals,
        variable_names,
        is_transformer,
        labels,
        res_fig_path,
        res_fig_title=None,
        res_distr_bins=50
    ):
        # fig, axes = plt.subplots(nrows=1, ncols=n_residuals, figsize=(5 * n_residuals, 4))
        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(2, 6, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax3 = fig.add_subplot(gs[0, 4:6])
        ax4 = fig.add_subplot(gs[1, 1:3])
        ax5 = fig.add_subplot(gs[1, 3:5])
        axes = [ax1, ax2, ax3, ax4, ax5]
        if n_residuals == 1:
            axes = [axes]
        # TRANSFORMER RANGES
        if is_transformer:
            # ranges = [[-0.03, 0.03], [-25, 25], [-0.35, 0.35], [-0.25, 0.25], [-0.05, 0.05]]
            ranges = [[-0.02, 0.02], [-25, 25], [-0.4, 0.4], [-0.2, 0.2], [-0.03, 0.03]]
            # ranges = [[-0.02, 0.02], [-25, 25], [-0.4, 0.4], [-0.2, 0.2], [-0.03, 0.03]]
            # bins = [100, 70, 50, 50, 75]
            bins = [100, 70, 50, 50, 75]
            ml_color = "darkorange"
        # MLP RANGES
        else:
            # ranges = [[-0.005, 0.005], [-23, 23], [-0.35, 0.35], [-0.25, 0.25], [-0.03, 0.03]]
            ranges = [[-0.002, 0.002], [-25, 25], [-0.4, 0.4], [-0.2, 0.2], [-0.03, 0.03]]
            # ranges = [[-0.02, 0.02], [-25, 25], [-0.4, 0.4], [-0.2, 0.2], [-0.03, 0.03]]
            # bins = [100, 70, 50, 50, 75]
            # bins = [100, 70, 50, 50, 75]
            bins = [400, 70, 50, 50, 75]
            ml_color = "cornflowerblue"
        for i in range(n_residuals):
            # axes[i].hist(residuals[:, i], bins=bins[i], color='skyblue', edgecolor='black')
            axes[i].hist(residuals[:, i], bins=bins[i], color=ml_color, edgecolor='black')
            # axes[i].set_title(f'Residuals for {variable_names[i]}')
            axes[i].set_title(f'{variable_names[i]}')
            axes[i].set_xlabel(f"{labels[i]}")
            axes[i].set_ylabel('Frequency')
            axes[i].set_xlim(ranges[i])
            axes[i].grid(True)
            # axes[i].set_xlim(-0.1, 0.1)
        if res_fig_title:
            fig.suptitle(res_fig_title)
        plt.tight_layout()
        # plt.savefig(res_fig_path, dpi=300)
        plt.savefig(res_fig_path)
        plt.show()
        print("Saved residual figures to", res_fig_path)

    def residualCalculations(self, model, evaluation_dataset, is_transformer):
        preds, targets = self.makePredictions(
            model,
            evaluation_dataset,
            is_transformer,
            return_targets=True
        )
        print(preds[:5])
        print(targets[:5])
        residuals = preds - targets
        return residuals

    def makeTransformerResidualPredictions(
        self,
        transformer_model,
        X_batch,
        y_batch,
        device
    ):
        X_batch = X_batch.reshape(y_batch.shape[0], self.max_seq_len, 3)
        mask = (X_batch == 0).all(dim=2)
        mask_gpu = mask.to(device)
        X_batch = X_batch.to(device)
        preds = transformer_model(X_batch, mask=mask_gpu)
        preds.cpu()
        return preds

    def makeMlpResidualPredictions(self, mlp_model, X_batch, device):
        X_batch = X_batch.to(device)
        preds = mlp_model(X_batch)
        return preds
    
    ##################################################
    # Correlation
    ##################################################
    def residualCorrelationHeatmaps(self, model, test_data_dir, is_transformer, fig_dir):
        test_measurements_df = self.data_handler.createMeasurementDfPOCA(test_data_dir)
        test_particles_df = self.data_handler.createPOCAParameterDf(test_data_dir)
        X_test = self.data_handler.preprocessX(test_measurements_df)
        y_test = self.data_handler.preprocessY(test_particles_df)
        test_dataset = MlDataset(X_test, y_test)
        predictions = self.makePredictions(model, test_dataset, is_transformer)

        pred_d0 = predictions[:,0]
        pred_z0 = predictions[:,1]
        pred_phi = predictions[:,2]
        pred_theta = predictions[:,3]
        pred_q_over_p = predictions[:,4]

        d0_truth =       test_particles_df["d0"].to_numpy()
        z0_truth =       test_particles_df["z0"].to_numpy()
        phi_truth =      test_particles_df["phi"].to_numpy()
        theta_truth =    test_particles_df["theta"].to_numpy()
        q_over_p_truth = test_particles_df["q_over_p"].to_numpy()
        
        filter_df = self.data_handler.createFilterTrackSummaryDf(test_data_dir)
        d0_filter =       filter_df["est_d0"].to_numpy()
        z0_filter =       filter_df["est_z0"].to_numpy()
        phi_filter =      filter_df["est_phi"].to_numpy()
        theta_filter =    filter_df["est_theta"].to_numpy()
        q_over_p_filter = filter_df["est_q_over_p"].to_numpy()

        residuals_ml = {
            "d₀": d0_truth - pred_d0,
            "z₀": z0_truth - pred_z0,
            "φ": phi_truth - pred_phi,
            "θ": theta_truth - pred_theta,
            "q/p": q_over_p_truth - pred_q_over_p
        }
        residuals_filter = {
            "d₀": d0_truth - d0_filter,
            "z₀": z0_truth - z0_filter,
            "φ": phi_truth - phi_filter,
            "θ": theta_truth - theta_filter,
            "q/p": q_over_p_truth - q_over_p_filter
        }
        self.correlationHeatmap(residuals_filter, "GSF", fig_dir)
        self.correlationHeatmap(residuals_ml, "Transformer", fig_dir)

    def correlationHeatmap(self, data_dict, vis_name, fig_dir):
        df = pd.DataFrame(data_dict)
        corr = df.corr(method="pearson")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr.values, vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns)
        ax.set_yticklabels(corr.columns)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Pearson correlation")
        ax.set_title(f"Residual correlation matrix for {vis_name}")
        plt.tight_layout()
        plt.savefig("".join([fig_dir, "/", vis_name, ".png"]))
        plt.show()



    ##################################################
    # Model vs. GSF
    ##################################################
    def pocaModelToGSFComparison(
        self,
        model,
        test_data_dir,
        is_transformer,
        figures_dir,
        file_name_ending
    ):
        test_measurements_df = self.data_handler.createMeasurementDfPOCA(test_data_dir)
        test_particles_df = self.data_handler.createPOCAParameterDf(test_data_dir)
        X_test = self.data_handler.preprocessX(test_measurements_df)
        y_test = self.data_handler.preprocessY(test_particles_df)
        test_dataset = MlDataset(X_test, y_test)
        predictions = self.makePredictions(model, test_dataset, is_transformer)

        pred_d0 = predictions[:,0]
        pred_z0 = predictions[:,1]
        pred_phi = predictions[:,2]
        pred_theta = predictions[:,3]
        pred_q_over_p = predictions[:,4]

        d0_truth =       test_particles_df["d0"].to_numpy()
        z0_truth =       test_particles_df["z0"].to_numpy()
        phi_truth =      test_particles_df["phi"].to_numpy()
        theta_truth =    test_particles_df["theta"].to_numpy()
        q_over_p_truth = test_particles_df["q_over_p"].to_numpy()
        
        filter_df = self.data_handler.createFilterTrackSummaryDf(test_data_dir)
        d0_filter =       filter_df["est_d0"].to_numpy()
        z0_filter =       filter_df["est_z0"].to_numpy()
        phi_filter =      filter_df["est_phi"].to_numpy()
        theta_filter =    filter_df["est_theta"].to_numpy()
        q_over_p_filter = filter_df["est_q_over_p"].to_numpy()

        residuals_ml = {
            "d₀": d0_truth - pred_d0,
            "z₀": z0_truth - pred_z0,
            "φ": phi_truth - pred_phi,
            "θ": theta_truth - pred_theta,
            "q/p": q_over_p_truth - pred_q_over_p
        }
        residuals_filter = {
            "d₀": d0_truth - d0_filter,
            "z₀": z0_truth - z0_filter,
            "φ": phi_truth - phi_filter,
            "θ": theta_truth - theta_filter,
            "q/p": q_over_p_truth - q_over_p_filter
        }
        self.mlVsModelPlotting(
            residuals_ml,
            residuals_filter,
            is_transformer,
            figures_dir,
            file_name_ending
        )

    def mlVsModelPlotting(
        self,
        residuals_ml,
        residuals_filter,
        is_transformer,
        figures_dir,
        file_name_ending
    ):
        # plt.figure(figsize=(15, 10))
        alpha = 0.6
        units = np.array(["mm", "mm", "radians", "radians", "C / (GeV c⁻¹)"])

        if is_transformer:
            ranges = [[-0.06, 0.06], [-10, 10], [-0.175, 0.175], [-0.1, 0.1], [-0.03, 0.03]]
            n_bins_list = [70000, 300, 1200, 80, 3000]
            # ranges = [[-2.5, 2.5], [-5, 5], [-0.025, 0.025], [-0.0025, 0.0025], [-0.2, 0.2]]
        else:
            ranges = [[-0.05, 0.05], [-6, 6], [-0.175, 0.175], [-0.08, 0.08], [-0.03, 0.03]]
            # n_bins_list = [70000, 500, 1200, 80, 3000]
            n_bins_list = [110000, 800, 1200, 120, 3000]
        # fig = plt.figure(figsize=(10, 6))
        fig = plt.figure(figsize=(15, 9))
        gs = GridSpec(2, 6, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax3 = fig.add_subplot(gs[0, 4:6])
        ax4 = fig.add_subplot(gs[1, 1:3])
        ax5 = fig.add_subplot(gs[1, 3:5])
        axes = [ax1, ax2, ax3, ax4, ax5]
        for i, key in enumerate(residuals_ml.keys(), start=1):
            # plt.subplot(2, 3, i)  # 2 rows × 3 columns grid

            # Combined bins calculation for combined image
            n_bins = n_bins_list[i-1]
            combined = np.hstack((residuals_ml[key], residuals_filter[key]))
            min_val, max_val = np.nanmin(combined), np.nanmax(combined)
            # bins = 200  # adjust as needed
            bins = np.linspace(min_val, max_val, n_bins + 1)
            # bins = 200
            # # For the filter
            # ranges = [[-0.25, 0.25], [-0.5, 0.5], [-0.01, 0.01], [-0.00075, 0.00075], [-0.03, 0.03]]
            # n_bins_list = [10000, 5000, 5000, 800, 2000]
            # # For separate filter visualization
            # plt.hist(residuals_filter[key], bins=n_bins_list[i], color='skyblue', edgecolor='black', alpha=0.5, label='Filter', density=False)
            # plt.hist(residuals_filter[key], bins, color='skyblue', edgecolor='black', alpha=0.5, label='Filter', density=False)

            if is_transformer:
                c = "darkorange"
                l = "Transformer"
            else:
                c = "royalblue"
                l = "MLP"
            
            ax = axes[i-1]
            ax.hist(residuals_ml[key], bins=bins, color=c, edgecolor='black', alpha=alpha, label=l, density=False)
            ax.hist(residuals_filter[key], bins=bins, color='red', edgecolor='black', alpha=alpha, label='GSF', density=False)
            ax.axvline(0, color='black', linestyle='--', linewidth=1)
            ax.grid(True)
            ax.set_xlim(ranges[i-1])
            ax.set_xlabel(f"Residual ({units[i-1]})")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Residuals for {key}")
            ax.legend()
            
            # plt.hist(residuals_ml[key], bins=n_bins_list[i-1], color='blue', edgecolor='black', alpha=alpha, label='MLP', density=False)
            # plt.hist(residuals_ml[key], bins=50, color='blue', edgecolor='black', alpha=alpha, label='MLP', density=False)
            # ax = plt.gca()
            # plt.xlabel("Residual ({})".format(units[i-1]))
            # plt.ylabel("Frequency")
            # plt.title(f"Residuals for {key}")
            # plt.legend()
            

        # ax = plt.gca()
        # ax[0].set_xlim([-0.025, 0.025])
        # ax[1].set_xlim([-10, 10])
        # ax[2].set_xlim([-0.025, 0.025])
        # ax[3].set_xlim([-0.15, 0.15])
        # ax[4].set_xlim([-0.04, 0.04])
        fig.suptitle(f"{l} residuals compared to GSF residuals")
        plt.tight_layout()
        res_fig_path = os.path.join(figures_dir, "MEGA_model_to_filter_comparison" + file_name_ending)
        plt.savefig(res_fig_path, dpi=300)
        # plt.show()
        print("Saved model VS. filter z_0 comparison to", res_fig_path)


    # Copypaste of the two functions above for combining the MLP and transformer figures
    # into one
    def pocaModelToGSFComparisonCombined(
        self,
        mlp_model,
        tr_model,
        test_data_dir,
        figures_dir,
        file_name_ending
    ):
        test_measurements_df = self.data_handler.createMeasurementDfPOCA(test_data_dir)
        test_particles_df = self.data_handler.createPOCAParameterDf(test_data_dir)
        X_test = self.data_handler.preprocessX(test_measurements_df)
        y_test = self.data_handler.preprocessY(test_particles_df)
        test_dataset = MlDataset(X_test, y_test)
        mlp_predictions = self.makePredictions(mlp_model, test_dataset, is_transformer=False)
        tr_predictions = self.makePredictions(tr_model, test_dataset, is_transformer=True)

        mlp_pred_d0 = mlp_predictions[:,0]
        mlp_pred_z0 = mlp_predictions[:,1]
        mlp_pred_phi = mlp_predictions[:,2]
        mlp_pred_theta = mlp_predictions[:,3]
        mlp_pred_q_over_p = mlp_predictions[:,4]

        tr_pred_d0 = tr_predictions[:,0]
        tr_pred_z0 = tr_predictions[:,1]
        tr_pred_phi = tr_predictions[:,2]
        tr_pred_theta = tr_predictions[:,3]
        tr_pred_q_over_p = tr_predictions[:,4]

        d0_truth =       test_particles_df["d0"].to_numpy()
        z0_truth =       test_particles_df["z0"].to_numpy()
        phi_truth =      test_particles_df["phi"].to_numpy()
        theta_truth =    test_particles_df["theta"].to_numpy()
        q_over_p_truth = test_particles_df["q_over_p"].to_numpy()
        
        filter_df = self.data_handler.createFilterTrackSummaryDf(test_data_dir)
        d0_filter =       filter_df["est_d0"].to_numpy()
        z0_filter =       filter_df["est_z0"].to_numpy()
        phi_filter =      filter_df["est_phi"].to_numpy()
        theta_filter =    filter_df["est_theta"].to_numpy()
        q_over_p_filter = filter_df["est_q_over_p"].to_numpy()

        residuals_mlp = {
            r"$d_0$": d0_truth - mlp_pred_d0,
            r"$z_0$": z0_truth - mlp_pred_z0,
            r"$\phi$": phi_truth - mlp_pred_phi,
            r"$\theta$": theta_truth - mlp_pred_theta,
            r"$q/p$": q_over_p_truth - mlp_pred_q_over_p
        }
        residuals_tr = {
            r"$d_0$": d0_truth - tr_pred_d0,
            r"$z_0$": z0_truth - tr_pred_z0,
            r"$\phi$": phi_truth - tr_pred_phi,
            r"$\theta$": theta_truth - tr_pred_theta,
            r"$q/p$": q_over_p_truth - tr_pred_q_over_p
        }
        residuals_filter = {
            r"$d_0$": d0_truth - d0_filter,
            r"$z_0$": z0_truth - z0_filter,
            r"$\phi$": phi_truth - phi_filter,
            r"$\theta$": theta_truth - theta_filter,
            r"$q/p$": q_over_p_truth - q_over_p_filter
        }
        self.mlVsModelPlottingCombined(
            residuals_mlp,
            residuals_tr,
            residuals_filter,
            figures_dir,
            file_name_ending
        )

    def mlVsModelPlottingCombined(
        self,
        residuals_mlp,
        residuals_tr,
        residuals_filter,
        figures_dir,
        file_name_ending
    ):
        # plt.figure(figsize=(15, 10))
        alpha = 0.6
        # units = np.array(["mm", "mm", "radians", "radians", "C / (GeV c⁻¹)"])
        units = np.array(["mm", "mm", "rad", "rad", r"$\text{GeV}^{-1}$"])
        labels = [
            r"$d_0^{\text{true}}-d_0^{\text{fit}$ [mm]",
            r"$z_0^{\text{true}}-z_0^{\text{fit}$ [mm]",
            r"$\phi{\text{true}}-\phi^{\text{fit}}$ [rad]",
            r"$\theta^{\text{true}}-\theta^{\text{fit}}$ [rad]",
            r"$q/p^{\text{true}}-q/p^{\text{fit}}$ [$\text{GeV}^{-1}$]"
        ]
        non_latex_names = ["d₀", "z₀", "φ", "θ", "q/p"]

        # Separate transformer
        # ranges = [[-0.06, 0.06], [-10, 10], [-0.175, 0.175], [-0.1, 0.1], [-0.03, 0.03]]
        # n_bins_list = [70000, 300, 1200, 80, 3000]
        # Separate MLP
        # ranges = [[-0.05, 0.05], [-6, 6], [-0.175, 0.175], [-0.08, 0.08], [-0.03, 0.03]]
        # n_bins_list = [110000, 800, 1200, 120, 3000]

        ranges = [[-0.06, 0.06], [-10, 10], [-0.175, 0.175], [-0.1, 0.1], [-0.03, 0.03]]
        # n_bins_list = [100000, 300, 1200, 80, 3000]
        n_bins_list = [100000, 400, 1300, 85, 4000]

        # fig = plt.figure(figsize=(10, 6))
        # fig = plt.figure(figsize=(15, 9))
        fig = plt.figure(figsize=(12, 7))
        gs = GridSpec(2, 6, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax3 = fig.add_subplot(gs[0, 4:6])
        ax4 = fig.add_subplot(gs[1, 1:3])
        ax5 = fig.add_subplot(gs[1, 3:5])
        axes = [ax1, ax2, ax3, ax4, ax5]
        for i, key in enumerate(residuals_mlp.keys(), start=1):
            # plt.subplot(2, 3, i)  # 2 rows × 3 columns grid

            # Combined bins calculation for combined image
            n_bins = n_bins_list[i-1]
            print("Stacking values...")
            combined = np.hstack((residuals_mlp[key], residuals_tr[key], residuals_filter[key]))
            min_val, max_val = np.nanmin(combined), np.nanmax(combined)
            # bins = 200  # adjust as needed
            print("Creating linspace")
            bins = np.linspace(min_val, max_val, n_bins + 1)
            # bins = 200
            # # For the filter
            # ranges = [[-0.25, 0.25], [-0.5, 0.5], [-0.01, 0.01], [-0.00075, 0.00075], [-0.03, 0.03]]
            # n_bins_list = [10000, 5000, 5000, 800, 2000]
            # # For separate filter visualization
            # plt.hist(residuals_filter[key], bins=n_bins_list[i], color='skyblue', edgecolor='black', alpha=0.5, label='Filter', density=False)
            # plt.hist(residuals_filter[key], bins, color='skyblue', edgecolor='black', alpha=0.5, label='Filter', density=False)

            print(f"Plotting {key}")
            ax = axes[i-1]
            ax.hist(residuals_mlp[key], bins=bins, color="royalblue", edgecolor='black', alpha=alpha, label="MLP", density=False)
            ax.hist(residuals_tr[key], bins=bins, color="darkorange", edgecolor='black', alpha=alpha, label="Transformer", density=False)
            ax.hist(residuals_filter[key], bins=bins, color='red', edgecolor='black', alpha=alpha, label='GSF', density=False)
            ax.axvline(0, color='black', linestyle='--', linewidth=1)
            ax.grid(True)
            ax.set_xlim(ranges[i-1])
            # ax.set_xlabel(f"Residual [{units[i-1]}]")
            ax.set_xlabel(f"{labels[i-1]}")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{key}")
            ax.legend()

            # plt.hist(residuals_ml[key], bins=n_bins_list[i-1], color='blue', edgecolor='black', alpha=alpha, label='MLP', density=False)
            # plt.hist(residuals_ml[key], bins=50, color='blue', edgecolor='black', alpha=alpha, label='MLP', density=False)
            # ax = plt.gca()
            # plt.xlabel("Residual ({})".format(units[i-1]))
            # plt.ylabel("Frequency")
            # plt.title(f"Residuals for {key}")
            # plt.legend()
            

        # ax = plt.gca()
        # ax[0].set_xlim([-0.025, 0.025])
        # ax[1].set_xlim([-10, 10])
        # ax[2].set_xlim([-0.025, 0.025])
        # ax[3].set_xlim([-0.15, 0.15])
        # ax[4].set_xlim([-0.04, 0.04])
        print("Postprocessing")
        fig.suptitle(f"ML residuals compared to GSF residuals")
        plt.tight_layout()
        res_fig_path = os.path.join(figures_dir, "MEGA_model_to_filter_comparison" + file_name_ending)
        plt.savefig(res_fig_path, dpi=300)
        # plt.show()
        print("Saved model VS. filter z_0 comparison to", res_fig_path)



    def makePredictions(
        self,
        model,
        evaluation_dataset,
        is_transformer, # for makeTransformerResidualPredictions or makeMlpResidualPredictions
        return_targets=False,
        event_batch_size=1024
    ):
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # Important that the loader does not shuffle in test cases where the order is assumed to be
        # the same in comparisons with e.g. the truth or the GSF
        dataloader = DataLoader(evaluation_dataset, batch_size=event_batch_size, shuffle=False)
        all_preds = []
        if return_targets:
            all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                if is_transformer:
                    preds = self.makeTransformerResidualPredictions(model, X_batch, y_batch, device)
                else:
                    preds = self.makeMlpResidualPredictions(model, X_batch, device)
                all_preds.append(preds)
                if return_targets:
                    all_targets.append(y_batch)
        all_preds = torch.cat(all_preds)
        all_preds = all_preds.cpu().numpy()
        all_preds = self.output_scaler.inverse_transform(all_preds)
        if return_targets:
            all_targets = torch.cat(all_targets)
            all_targets = all_targets.cpu().numpy()
            all_targets = self.output_scaler.inverse_transform(all_targets)
            return (all_preds, all_targets)
        return all_preds
    

    # ##################################################
    # # Correlation plotting
    # ##################################################
    # def correlationPlots(self, data_dir, model, is_transformer):
    #     self.truthCorrelationPlots(data_dir)
    #     self.gsfCorrelationPlots(data_dir)
    #     self.modelCorrelationPlots(data_dir, model, is_transformer)

    # def truthCorrelationPlots(self, data_dir):
    #     poca_df = self.data_handler.createPOCAParameterDf(data_dir)
    #     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5 * 2, 4))

    #     d0 =    poca_df["d0"]
    #     phi =   poca_df["phi"]
    #     z0 =    poca_df["z0"]
    #     theta = poca_df["theta"]

    #     axes[0].scatter(d0, phi)
    #     axes[0].set_title("d₀ vs φ")
    #     axes[0].set_xlabel("d₀")
    #     axes[0].set_ylabel("φ")
    #     axes[0].grid(True)

    #     axes[1].scatter(z0, theta)
    #     axes[1].set_title("z₀ vs. θ")
    #     axes[1].set_xlabel("z₀")
    #     axes[1].set_ylabel("θ")
    #     axes[1].grid(True)

    #     fig.suptitle("Correlations (Truth)")
    #     plt.tight_layout()
    #     plt.show()

    # def gsfCorrelationPlots(self, data_dir):
    #     poca_df = self.data_handler.createFilterTrackSummaryDf(data_dir)
    #     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5 * 2, 4))

    #     d0 =    poca_df["est_d0"]
    #     phi =   poca_df["est_phi"]
    #     z0 =    poca_df["est_z0"]
    #     theta = poca_df["est_theta"]

    #     axes[0].scatter(d0, phi)
    #     axes[0].set_title("d₀ vs φ")
    #     axes[0].set_xlabel("d₀")
    #     axes[0].set_ylabel("φ")
    #     axes[0].grid(True)

    #     axes[1].scatter(z0, theta)
    #     axes[1].set_title("z₀ vs. θ")
    #     axes[1].set_xlabel("z₀")
    #     axes[1].set_ylabel("θ")
    #     axes[1].grid(True)

    #     fig.suptitle("Correlations (GSF)")
    #     plt.tight_layout()
    #     plt.show()

    # def modelCorrelationPlots(self, data_dir, model, is_transformer):
    #     evaluation_dataset = self.createTestDataset(data_dir)
    #     preds = self.makePredictions(
    #         model,
    #         evaluation_dataset,
    #         is_transformer
    #     )
    #     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5 * 2, 4))

    #     d0 =    preds[:,0]
    #     phi =   preds[:,1]
    #     z0 =    preds[:,2]
    #     theta = preds[:,3]

    #     axes[0].scatter(d0, phi)
    #     axes[0].set_title("d₀ vs φ")
    #     axes[0].set_xlabel("d₀")
    #     axes[0].set_ylabel("φ")
    #     axes[0].grid(True)

    #     axes[1].scatter(z0, theta)
    #     axes[1].set_title("z₀ vs. θ")
    #     axes[1].set_xlabel("z₀")
    #     axes[1].set_ylabel("θ")
    #     axes[1].grid(True)

    #     fig.suptitle("Correlations (GSF)")
    #     plt.tight_layout()
    #     plt.show()

from collections.abc import Callable
from torch.nn import Module

import sys
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import wandb
from ml_utilities import (
    MlDataset,
    EarlyStopping,
    createNewBeamspotsXy,
    createRandomBeamspotAndTrackParameters,
)

LOWER = -29
UPPER = 29
N_POINTS_IN_XY = 40
DISTRIBUTION_RADIUS = 25
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def getMlpOutputs(model, X_gpu, max_seq_len=None, device=None):
    return model(X_gpu)  # .squeeze()


def getTransformerOutputs(model, X_gpu, max_seq_len, device):
    batch_size = X_gpu.shape[0]
    X_gpu = X_gpu.reshape(batch_size, max_seq_len, 3)
    mask = (X_gpu == 0).all(dim=2)
    train_mask_gpu = mask.to(device)
    outputs = model(X_gpu, mask=train_mask_gpu)
    return outputs


def createCosineTailSchedulerWithWarmup(self, optimizer, total_steps, warmup_steps):
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer,
                start_factor=1e-4,
                end_factor=1.0,
                total_iters=warmup_steps,
            ),
            CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps),
        ],
        milestones=[warmup_steps],
    )
    return scheduler


class MlTrainer:
    def __init__(
        self,
        n_epochs: int,
        learning_rate: float,
        batch_size: int,
        getModelOutputs: Callable,
        model_save_path: str,
        max_seq_len: int,
        input_scaler: StandardScaler,
        output_scaler: StandardScaler,
        loss_function: Module = nn.MSELoss(),
    ):
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.getModelOutputs = getModelOutputs
        self.model_save_path = model_save_path
        self.max_seq_len = max_seq_len
        self.train_losses = np.full(self.n_epochs, -1, dtype=np.float32)
        self.train_losses_unscaled = np.full(self.n_epochs, -1, dtype=np.float32)
        self.train_losses_ind = np.full((self.n_epochs, 5), -1, dtype=np.float32)
        self.train_losses_unscaled_ind = np.full(
            (self.n_epochs, 5), -1, dtype=np.float32
        )
        self.val_losses = np.full(self.n_epochs, -1, dtype=np.float32)
        self.val_losses_unscaled = np.full(self.n_epochs, -1, dtype=np.float32)
        self.val_losses_ind = np.full((self.n_epochs, 5), -1, dtype=np.float32)
        self.val_losses_unscaled_ind = np.full((self.n_epochs, 5), -1, dtype=np.float32)
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = loss_function
        self.target_names = ["d0", "z0", "phi", "theta", "q_over_p"]

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_save_path)

    def load_model(self, model):
        model.load_state_dict(
            torch.load(self.model_save_path, map_location=self.device)
        )

    def getUnscaledLoss(self, criterion, outputs, y_batch):
        with torch.no_grad():
            # Moving to CPU and using converted numpy copies
            # pred_unscaled = self.output_scaler.inverse_transform(outputs.cpu().numpy())
            # y_unscaled = self.output_scaler.inverse_transform(y_batch.cpu().numpy())
            # pred_unscaled = torch.from_numpy(pred_unscaled).to(self.device)
            # y_unscaled = torch.from_numpy(y_unscaled).to(self.device)

            # Staying on the same device for faster equivalent computation
            outputs_det = outputs.detach().to(self.device)
            y_tensor = (
                y_batch.detach().to(self.device)
                if isinstance(y_batch, torch.Tensor)
                else torch.tensor(y_batch, device=self.device)
            )
            scale = torch.as_tensor(
                self.output_scaler.scale_, device=self.device, dtype=outputs_det.dtype
            )
            mean = torch.as_tensor(
                self.output_scaler.mean_, device=self.device, dtype=outputs_det.dtype
            )
            pred_unscaled = outputs_det * scale + mean
            y_unscaled = y_tensor * scale + mean

            loss_item = criterion(pred_unscaled, y_unscaled).item()
        return loss_item

    def getIndividualScaledLosses(self, outputs, y_gpu):
        criterion_none = torch.nn.MSELoss(reduction="none")
        with torch.no_grad():
            per_sample_per_output = criterion_none(outputs, y_gpu)
            # shape: (batch_size, n_outputs)
            per_output_mse = per_sample_per_output.mean(dim=0)
            per_output_mse = per_output_mse.detach().cpu().numpy()
        return per_output_mse

    def getIndividualUnscaledLosses(self, outputs, y_gpu):
        criterion_none = torch.nn.MSELoss(reduction="none")
        with torch.no_grad():
            # Moving to CPU and using converted numpy copies
            # pred_unscaled = self.output_scaler.inverse_transform(outputs.cpu().numpy())
            # y_unscaled = self.output_scaler.inverse_transform(y_gpu.cpu().numpy())
            # pred_unscaled = torch.from_numpy(pred_unscaled).to(self.device)
            # y_unscaled = torch.from_numpy(y_unscaled).to(self.device)

            # Staying on the same device for faster equivalent computation
            outputs_det = outputs.detach().to(self.device)
            y_tensor = y_gpu.to(self.device)
            scale = torch.as_tensor(
                self.output_scaler.scale_, device=self.device, dtype=outputs_det.dtype
            )
            mean = torch.as_tensor(
                self.output_scaler.mean_, device=self.device, dtype=outputs_det.dtype
            )
            pred_unscaled = outputs_det * scale + mean
            y_unscaled = y_tensor * scale + mean

            per_sample_per_output = criterion_none(pred_unscaled, y_unscaled)
            per_output_mse = per_sample_per_output.mean(dim=0)
            per_output_mse = per_output_mse.detach().cpu().numpy()
        return per_output_mse

    def _unscale_tensors(self, outputs, y_tensor):
        """Return (pred_unscaled, y_unscaled) tensors on trainer device."""
        outputs_det = outputs.detach().to(self.device)
        y_tensor = (
            y_tensor.detach().to(self.device)
            if isinstance(y_tensor, torch.Tensor)
            else torch.tensor(y_tensor, device=self.device)
        )
        scale = torch.as_tensor(
            self.output_scaler.scale_, device=self.device, dtype=outputs_det.dtype
        )
        mean = torch.as_tensor(
            self.output_scaler.mean_, device=self.device, dtype=outputs_det.dtype
        )
        pred_unscaled = outputs_det * scale + mean
        y_unscaled = y_tensor * scale + mean
        return pred_unscaled, y_unscaled

    def _variance_ratios(self, pred_unscaled, y_unscaled):
        """Compute per-target and overall prediction and residual variance ratios.

        Returns a dict with:
          - pred_ratio_overall, resid_ratio_overall (floats)
          - pred_ratio_per_target, resid_ratio_per_target (numpy arrays length n_targets)
        """
        # pred_unscaled, y_unscaled: tensors shape (N, n_targets)
        if pred_unscaled.numel() == 0:
            # empty -> return nans
            n_targets = pred_unscaled.shape[1] if pred_unscaled.dim() > 1 else 1
            nan_arr = np.full(n_targets, np.nan, dtype=np.float32)
            return {
                "pred_ratio_overall": float("nan"),
                "resid_ratio_overall": float("nan"),
                "pred_ratio_per_target": nan_arr,
                "resid_ratio_per_target": nan_arr,
            }

        # per-target variances
        with torch.no_grad():
            target_var = y_unscaled.var(dim=0, unbiased=False)
            pred_var = pred_unscaled.var(dim=0, unbiased=False)
            resid_var = (y_unscaled - pred_unscaled).var(dim=0, unbiased=False)

            # avoid division by zero: produce nan where target_var == 0
            safe_target_var = torch.where(
                target_var == 0,
                torch.tensor(float("nan"), device=self.device, dtype=target_var.dtype),
                target_var,
            )
            pred_ratio_per = (pred_var / safe_target_var).detach().cpu().numpy()
            resid_ratio_per = (resid_var / safe_target_var).detach().cpu().numpy()

            # overall (flattened) variances
            target_var_all = y_unscaled.view(-1).var(unbiased=False)
            pred_var_all = pred_unscaled.view(-1).var(unbiased=False)
            resid_var_all = (y_unscaled - pred_unscaled).view(-1).var(unbiased=False)
            pred_ratio_overall = (
                float(pred_var_all / target_var_all)
                if target_var_all != 0
                else float("nan")
            )
            resid_ratio_overall = (
                float(resid_var_all / target_var_all)
                if target_var_all != 0
                else float("nan")
            )

            # expose raw variances as numpy arrays / floats as well
            pred_var_per_np = pred_var.detach().cpu().numpy()
            resid_var_per_np = resid_var.detach().cpu().numpy()
            target_var_per_np = target_var.detach().cpu().numpy()
            pred_var_all_f = float(pred_var_all)
            resid_var_all_f = float(resid_var_all)
            target_var_all_f = float(target_var_all)
            # residual means and stds
            resid_mean_per_np = (
                (y_unscaled - pred_unscaled).mean(dim=0).detach().cpu().numpy()
            )
            resid_mean_all_f = float((y_unscaled - pred_unscaled).view(-1).mean())
            # std is sqrt of variance
            resid_std_per_np = np.sqrt(resid_var_per_np)
            try:
                resid_std_all_f = float(resid_var_all**0.5)
            except Exception:
                resid_std_all_f = float("nan")

        return {
            "pred_ratio_overall": pred_ratio_overall,
            "resid_ratio_overall": resid_ratio_overall,
            "pred_ratio_per_target": pred_ratio_per,
            "resid_ratio_per_target": resid_ratio_per,
            "pred_var_per_target": pred_var_per_np,
            "resid_var_per_target": resid_var_per_np,
            "target_var_per_target": target_var_per_np,
            "pred_var_overall": pred_var_all_f,
            "resid_var_overall": resid_var_all_f,
            "target_var_overall": target_var_all_f,
            "resid_mean_per_target": resid_mean_per_np,
            "resid_mean_overall": resid_mean_all_f,
            "resid_std_per_target": resid_std_per_np,
            "resid_std_overall": resid_std_all_f,
        }

    def _computeVarianceStats(self, preds_list, targets_list):
        """Concatenate collected per-batch (pred, target) tensors and compute
        variance-ratio stats via _variance_ratios. Returns the same all-NaN
        dict shape (covering every key _variance_ratios can produce) when
        nothing was collected, so callers never need a separate fallback."""
        if not preds_list:
            n_targets = len(self.target_names)
            nan_arr = np.full(n_targets, np.nan, dtype=np.float32)
            nan_keys = [
                "pred_ratio_overall",
                "resid_ratio_overall",
                "resid_var_overall",
                "resid_mean_overall",
                "resid_std_overall",
            ]
            stats = {k: float("nan") for k in nan_keys}
            for k in [
                "pred_ratio_per_target",
                "resid_ratio_per_target",
                "resid_var_per_target",
                "resid_mean_per_target",
                "resid_std_per_target",
            ]:
                stats[k] = nan_arr
            return stats

        preds = torch.cat(preds_list, dim=0).to(self.device)
        targets = torch.cat(targets_list, dim=0).to(self.device)
        return self._variance_ratios(preds, targets)

    def _flattenVarianceStats(self, prefix, split_label, stats):
        """Flatten a _computeVarianceStats() dict into wandb-loggable scalar
        entries, named to match the existing dashboard conventions
        (e.g. "8. Train pred var ratio overall")."""
        log_dict = {
            f"{prefix}. {split_label} pred var ratio overall": stats[
                "pred_ratio_overall"
            ],
            f"{prefix}. {split_label} resid var ratio overall": stats[
                "resid_ratio_overall"
            ],
            f"{prefix}. {split_label} resid var overall (abs)": float(
                stats["resid_var_overall"]
            ),
            f"{prefix}. {split_label} resid mean overall (abs)": float(
                stats["resid_mean_overall"]
            ),
            f"{prefix}. {split_label} resid std overall (abs)": float(
                stats["resid_std_overall"]
            ),
        }
        for i, name in enumerate(self.target_names):
            log_dict[f"{prefix}. {split_label} pred var ratio {name}"] = float(
                stats["pred_ratio_per_target"][i]
            )
            log_dict[f"{prefix}. {split_label} resid var ratio {name}"] = float(
                stats["resid_ratio_per_target"][i]
            )
            log_dict[f"{prefix}. {split_label} resid var {name} (abs)"] = float(
                stats["resid_var_per_target"][i]
            )
            log_dict[f"{prefix}. {split_label} resid mean {name} (abs)"] = float(
                stats["resid_mean_per_target"][i]
            )
            log_dict[f"{prefix}. {split_label} resid std {name} (abs)"] = float(
                stats["resid_std_per_target"][i]
            )
        return log_dict

    def wandbLogging(self, epoch):
        # Return a dict of epoch-level metrics so they can be merged and logged
        # together with other metrics in the training loop (avoids double
        # wandb.log() calls per epoch).
        return {
            "1. Epoch": epoch + 1,
            "1. Scaled training loss (MSE)": self.train_losses[epoch],
            "1. Scaled validation loss (MSE)": self.val_losses[epoch],
            "1. Unscaled training loss (MSE)": self.train_losses_unscaled[epoch],
            "1. Unscaled validation loss (MSE)": self.val_losses_unscaled[epoch],
            "1. Learning rate": self.last_lr,
            "2. d0_train loss unscaled": self.train_losses_unscaled_ind[epoch][0],
            "2. z0_train loss unscaled": self.train_losses_unscaled_ind[epoch][1],
            "2. phi_train loss unscaled": self.train_losses_unscaled_ind[epoch][2],
            "2. theta_train loss unscaled": self.train_losses_unscaled_ind[epoch][3],
            "2. q_over_p_train loss unscaled": self.train_losses_unscaled_ind[epoch][4],
            "3. d0_val loss unscaled": self.val_losses_unscaled_ind[epoch][0],
            "3. z0_val loss unscaled": self.val_losses_unscaled_ind[epoch][1],
            "3. phi_val loss unscaled": self.val_losses_unscaled_ind[epoch][2],
            "3. theta_val loss unscaled": self.val_losses_unscaled_ind[epoch][3],
            "3. q_over_p_val loss unscaled": self.val_losses_unscaled_ind[epoch][4],
            "4. d0_train loss scaled": self.train_losses_ind[epoch][0],
            "4. z0_train loss scaled": self.train_losses_ind[epoch][1],
            "4. phi_train loss scaled": self.train_losses_ind[epoch][2],
            "4. theta_train loss scaled": self.train_losses_ind[epoch][3],
            "4. q_over_p_train loss scaled": self.train_losses_ind[epoch][4],
            "5. d0_val loss scaled": self.val_losses_ind[epoch][0],
            "5. z0_val loss scaled": self.val_losses_ind[epoch][1],
            "5. phi_val loss scaled": self.val_losses_ind[epoch][2],
            "5. theta_val loss scaled": self.val_losses_ind[epoch][3],
            "5. q_over_p_val loss scaled": self.val_losses_ind[epoch][4],
        }


class OrigoBeamspotTrainer(MlTrainer):
    def train(self, model, X_train, y_train, X_val, y_val):
        train_dataset = MlDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_dataset = MlDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        early_stopper = EarlyStopping(self.model_save_path, patience=3000, verbose=True)

        model.to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)

        tot_steps = self.n_epochs * len(train_loader)
        scheduler = createCosineTailSchedulerWithWarmup(
            optimizer, total_steps=tot_steps, warmup_steps=int(0.02 * tot_steps)
        )

        print("Starting training loop")
        for epoch in range(self.n_epochs):
            model.train()
            tot_train_loss = 0.0
            tot_train_loss_unscaled = 0.0
            tot_ind_train_loss = np.full(5, 0.0, dtype=np.float32)
            tot_ind_train_loss_unscaled = np.full(5, 0.0, dtype=np.float32)
            tot_ind_val_loss = np.full(5, 0.0, dtype=np.float32)
            tot_ind_val_loss_unscaled = np.full(5, 0.0, dtype=np.float32)
            # collectors for variance ratio computation (unscaled)
            train_preds_unscaled_list = []
            train_targets_unscaled_list = []
            for X_batch, y_batch in train_loader:
                X_train_gpu = X_batch.to(self.device)
                y_train_gpu = y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.getModelOutputs(
                    model, X_train_gpu, self.max_seq_len, self.device
                )
                loss = self.criterion(outputs, y_train_gpu)
                # collect unscaled predictions and targets for variance ratio
                pred_unscaled_batch, y_unscaled_batch = self._unscale_tensors(
                    outputs, y_train_gpu
                )
                train_preds_unscaled_list.append(pred_unscaled_batch.detach().cpu())
                train_targets_unscaled_list.append(y_unscaled_batch.detach().cpu())
                # Individual losses
                aver_ind_output_losses = self.getIndividualScaledLosses(
                    outputs, y_train_gpu
                )
                aver_ind_unscaled_output_losses = self.getIndividualUnscaledLosses(
                    outputs, y_train_gpu
                )
                tot_ind_train_loss += aver_ind_output_losses * X_batch.shape[0]
                tot_ind_train_loss_unscaled += (
                    aver_ind_unscaled_output_losses * X_batch.shape[0]
                )
                loss.backward()
                optimizer.step()
                # Loss multiplied with the batch size in case the sizes of the batches are not the same
                # so it can be divided by the length of the dataset (n samples) later
                tot_train_loss += loss.item() * X_batch.shape[0]
                aver_unscaled_loss = self.getUnscaledLoss(
                    self.criterion, outputs, y_batch
                )
                tot_train_loss_unscaled += aver_unscaled_loss * X_batch.shape[0]

                # capture the LR actually used for this batch before the scheduler
                # advances it, so the epoch's logged LR isn't off by one step
                self.last_lr = optimizer.param_groups[0]["lr"]
                scheduler.step()

            # NOTE: divided by the length of the dataset in case the batch lengths are not the same
            self.train_losses[epoch] = tot_train_loss / len(train_dataset)
            self.train_losses_unscaled[epoch] = tot_train_loss_unscaled / len(
                train_dataset
            )
            self.train_losses_ind[epoch] = tot_ind_train_loss / len(train_dataset)
            self.train_losses_unscaled_ind[epoch] = tot_ind_train_loss_unscaled / len(
                train_dataset
            )

            model.eval()
            tot_val_loss = 0.0
            tot_val_loss_unscaled = 0.0
            with torch.no_grad():
                val_preds_unscaled_list = []
                val_targets_unscaled_list = []
                for val_X, val_y in val_loader:
                    X_val_gpu = val_X.to(self.device)
                    y_val_gpu = val_y.to(self.device)
                    val_outputs = self.getModelOutputs(
                        model, X_val_gpu, self.max_seq_len, self.device
                    )
                    val_loss = self.criterion(val_outputs, y_val_gpu).item()
                    tot_val_loss += val_loss * val_X.shape[0]

                    aver_unscaled_val_loss = self.getUnscaledLoss(
                        self.criterion, val_outputs, val_y
                    )
                    tot_val_loss_unscaled += aver_unscaled_val_loss * val_X.shape[0]

                    aver_ind_val_losses = self.getIndividualScaledLosses(
                        val_outputs, y_val_gpu
                    )
                    aver_ind_unscaled_val_losses = self.getIndividualUnscaledLosses(
                        val_outputs, y_val_gpu
                    )
                    tot_ind_val_loss += aver_ind_val_losses * val_X.shape[0]
                    tot_ind_val_loss_unscaled += (
                        aver_ind_unscaled_val_losses * val_X.shape[0]
                    )

                    # collect unscaled predictions and targets for variance ratio
                    pred_unscaled_val, y_unscaled_val = self._unscale_tensors(
                        val_outputs, y_val_gpu
                    )
                    val_preds_unscaled_list.append(pred_unscaled_val.detach().cpu())
                    val_targets_unscaled_list.append(y_unscaled_val.detach().cpu())

            self.val_losses[epoch] = tot_val_loss / len(val_dataset)
            self.val_losses_unscaled[epoch] = tot_val_loss_unscaled / len(val_dataset)
            self.val_losses_ind[epoch] = tot_ind_val_loss / len(val_dataset)
            self.val_losses_unscaled_ind[epoch] = tot_ind_val_loss_unscaled / len(
                val_dataset
            )

            print(
                f"Epoch {epoch+1}, Scaled training loss: {self.train_losses[epoch]:.4f}, Unscaled training loss: {self.train_losses_unscaled[epoch]:.4f}, Scaled validation loss: {self.val_losses[epoch]:.4f}, Unscaled validation loss {self.val_losses_unscaled[epoch]:.4f}"
            )
            # Core epoch metrics always get logged, even if the supplementary
            # variance-ratio metrics below fail for some reason.
            epoch_log = self.wandbLogging(epoch)

            variance_log = {}
            try:
                train_stats = self._computeVarianceStats(
                    train_preds_unscaled_list, train_targets_unscaled_list
                )
                val_stats = self._computeVarianceStats(
                    val_preds_unscaled_list, val_targets_unscaled_list
                )
                variance_log.update(
                    self._flattenVarianceStats("8", "Train", train_stats)
                )
                variance_log.update(self._flattenVarianceStats("9", "Val", val_stats))
            except Exception as exc:
                # Supplementary metrics only - never let a failure here drop
                # the core loss curves from this epoch's log.
                print(f"Epoch {epoch+1}: skipping variance-ratio metrics ({exc!r})")

            # single wandb.log() call per epoch, so the step axis stays aligned
            wandb.log({**epoch_log, **variance_log})

            early_stopper(self.val_losses[epoch], model)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break

        early_stopper.load_best_model(model)
        return model

    def test(self, model, X_test, y_test):
        model.eval()
        test_dataset = MlDataset(X_test, y_test)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        tot_test_loss = 0.0
        tot_test_loss_unscaled = 0.0
        tot_ind_test_loss = np.full(5, 0.0, dtype=np.float32)
        tot_ind_test_loss_unscaled = np.full(5, 0.0, dtype=np.float32)

        test_preds_unscaled_list = []
        test_targets_unscaled_list = []

        with torch.no_grad():
            for X_test_batch, y_test in test_loader:
                X_test_gpu = X_test_batch.to(self.device)
                y_test_gpu = y_test.to(self.device)
                test_outputs = self.getModelOutputs(
                    model, X_test_gpu, self.max_seq_len, self.device
                )
                tot_test_loss += (
                    self.criterion(test_outputs, y_test_gpu).item()
                    * X_test_batch.shape[0]
                )
                aver_unscaled_test_loss = self.getUnscaledLoss(
                    self.criterion, test_outputs, y_test_gpu
                )
                tot_test_loss_unscaled += (
                    aver_unscaled_test_loss * X_test_batch.shape[0]
                )

                aver_ind_output_losses = self.getIndividualScaledLosses(
                    test_outputs, y_test_gpu
                )
                aver_ind_unscaled_output_losses = self.getIndividualUnscaledLosses(
                    test_outputs, y_test_gpu
                )
                tot_ind_test_loss += aver_ind_output_losses * X_test_batch.shape[0]
                tot_ind_test_loss_unscaled += (
                    aver_ind_unscaled_output_losses * X_test_batch.shape[0]
                )

                # collect unscaled preds/targets for variance ratios
                pred_unscaled_t, y_unscaled_t = self._unscale_tensors(
                    test_outputs, y_test_gpu
                )
                test_preds_unscaled_list.append(pred_unscaled_t.detach().cpu())
                test_targets_unscaled_list.append(y_unscaled_t.detach().cpu())

        test_loss = tot_test_loss / len(test_dataset)
        test_loss_unscaled = tot_test_loss_unscaled / len(test_dataset)
        test_losses_ind = tot_ind_test_loss / len(test_dataset)
        test_losses_unscaled_ind = tot_ind_test_loss_unscaled / len(test_dataset)

        print(
            f"Scaled test loss: {test_loss:.4f}, Unscaled test loss: {test_loss_unscaled:.4f}"
        )  # , Individual scaled test losses: {test_losses_ind:.4f}, Individual unscaled test losses: {test_losses_unscaled_ind:.4f}")

        # Core test metrics always get logged, even if the supplementary
        # variance-ratio metrics below fail for some reason.
        test_log = {
            "Scaled test loss (MSE)": test_loss,
            "Unscaled test loss (MSE)": test_loss_unscaled,
            "6. Test loss d0 scaled": test_losses_ind[0],
            "6. Test loss z0 scaled": test_losses_ind[1],
            "6. Test loss phi scaled": test_losses_ind[2],
            "6. Test loss theta scaled": test_losses_ind[3],
            "6. Test loss q_over_p scaled": test_losses_ind[4],
            "7. Test loss d0 unscaled": test_losses_unscaled_ind[0],
            "7. Test loss z0 unscaled": test_losses_unscaled_ind[1],
            "7. Test loss phi unscaled": test_losses_unscaled_ind[2],
            "7. Test loss theta unscaled": test_losses_unscaled_ind[3],
            "7. Test loss q_over_p unscaled": test_losses_unscaled_ind[4],
        }

        try:
            test_stats = self._computeVarianceStats(
                test_preds_unscaled_list, test_targets_unscaled_list
            )
            test_log.update(self._flattenVarianceStats("10", "Test", test_stats))
        except Exception as exc:
            # TODO: ACTS logging?
            print(f"Test evaluation: skipping variance-ratio metrics ({exc!r})")

        wandb.log(test_log)


class ArbitraryBeamspotTrainer(MlTrainer):
    def train(self, model, X_train, train_params, X_val, val_params):
        train_dataset = MlDataset(X_train, train_params)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_dataset = MlDataset(X_val, val_params)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        early_stopper = EarlyStopping(self.model_save_path, patience=3000, verbose=True)

        model.to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)

        tot_steps = self.n_epochs * len(train_loader)
        scheduler = createCosineTailSchedulerWithWarmup(
            optimizer, total_steps=tot_steps, warmup_steps=int(0.02 * tot_steps)
        )

        beamspots = createNewBeamspotsXy(
            LOWER, UPPER, N_POINTS_IN_XY, DISTRIBUTION_RADIUS
        )
        distance_between_beamspots = (UPPER - LOWER) / N_POINTS_IN_XY

        print("Starting training loop")
        for epoch in range(self.n_epochs):
            model.train()
            tot_train_loss = 0.0
            tot_train_loss_unscaled = 0.0
            tot_ind_train_loss = np.full(5, 0.0, dtype=np.float32)
            tot_ind_train_loss_unscaled = np.full(5, 0.0, dtype=np.float32)
            tot_ind_val_loss = np.full(5, 0.0, dtype=np.float32)
            tot_ind_val_loss_unscaled = np.full(5, 0.0, dtype=np.float32)
            # collectors for variance ratio computation (unscaled)
            train_preds_unscaled_list = []
            train_targets_unscaled_list = []
            for X_batch, params_train_batch in train_loader:
                y_batch = createRandomBeamspotAndTrackParameters(
                    beamspots,
                    params_train_batch,
                    noise_scale=distance_between_beamspots / 2,
                )
                X_train_gpu = X_batch.to(self.device)
                y_train_gpu = y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.getModelOutputs(
                    model, X_train_gpu, self.max_seq_len, self.device
                )
                loss = self.criterion(outputs, y_train_gpu)
                # collect unscaled predictions and targets for variance ratio
                pred_unscaled_batch, y_unscaled_batch = self._unscale_tensors(
                    outputs, y_train_gpu
                )
                train_preds_unscaled_list.append(pred_unscaled_batch.detach().cpu())
                train_targets_unscaled_list.append(y_unscaled_batch.detach().cpu())
                # Individual losses
                aver_ind_output_losses = self.getIndividualScaledLosses(
                    outputs, y_train_gpu
                )
                aver_ind_unscaled_output_losses = self.getIndividualUnscaledLosses(
                    outputs, y_train_gpu
                )
                tot_ind_train_loss += aver_ind_output_losses * X_batch.shape[0]
                tot_ind_train_loss_unscaled += (
                    aver_ind_unscaled_output_losses * X_batch.shape[0]
                )
                loss.backward()
                optimizer.step()
                # Loss multiplied with the batch size in case the sizes of the batches are not the same
                # so it can be divided by the length of the dataset (n samples) later
                tot_train_loss += loss.item() * X_batch.shape[0]
                aver_unscaled_loss = self.getUnscaledLoss(
                    self.criterion, outputs, y_batch
                )
                tot_train_loss_unscaled += aver_unscaled_loss * X_batch.shape[0]

                # capture the LR actually used for this batch before the scheduler
                # advances it, so the epoch's logged LR isn't off by one step
                self.last_lr = optimizer.param_groups[0]["lr"]
                scheduler.step()

            # NOTE: divided by the length of the dataset in case the batch lengths are not the same
            self.train_losses[epoch] = tot_train_loss / len(train_dataset)
            self.train_losses_unscaled[epoch] = tot_train_loss_unscaled / len(
                train_dataset
            )
            self.train_losses_ind[epoch] = tot_ind_train_loss / len(train_dataset)
            self.train_losses_unscaled_ind[epoch] = tot_ind_train_loss_unscaled / len(
                train_dataset
            )

            model.eval()
            tot_val_loss = 0.0
            tot_val_loss_unscaled = 0.0
            with torch.no_grad():
                val_preds_unscaled_list = []
                val_targets_unscaled_list = []
                for val_X, params_val_batch in val_loader:
                    val_y = createRandomBeamspotAndTrackParameters(
                        beamspots, params_val_batch
                    )
                    X_val_gpu = val_X.to(self.device)
                    y_val_gpu = val_y.to(self.device)
                    val_outputs = self.getModelOutputs(
                        model, X_val_gpu, self.max_seq_len, self.device
                    )
                    val_loss = self.criterion(val_outputs, y_val_gpu).item()
                    tot_val_loss += val_loss * val_X.shape[0]

                    aver_unscaled_val_loss = self.getUnscaledLoss(
                        self.criterion, val_outputs, val_y
                    )
                    tot_val_loss_unscaled += aver_unscaled_val_loss * val_X.shape[0]

                    aver_ind_val_losses = self.getIndividualScaledLosses(
                        val_outputs, y_val_gpu
                    )
                    aver_ind_unscaled_val_losses = self.getIndividualUnscaledLosses(
                        val_outputs, y_val_gpu
                    )
                    tot_ind_val_loss += aver_ind_val_losses * val_X.shape[0]
                    tot_ind_val_loss_unscaled += (
                        aver_ind_unscaled_val_losses * val_X.shape[0]
                    )

                    # collect unscaled predictions and targets for variance ratio
                    pred_unscaled_val, y_unscaled_val = self._unscale_tensors(
                        val_outputs, y_val_gpu
                    )
                    val_preds_unscaled_list.append(pred_unscaled_val.detach().cpu())
                    val_targets_unscaled_list.append(y_unscaled_val.detach().cpu())

            self.val_losses[epoch] = tot_val_loss / len(val_dataset)
            self.val_losses_unscaled[epoch] = tot_val_loss_unscaled / len(val_dataset)
            self.val_losses_ind[epoch] = tot_ind_val_loss / len(val_dataset)
            self.val_losses_unscaled_ind[epoch] = tot_ind_val_loss_unscaled / len(
                val_dataset
            )

            print(
                f"Epoch {epoch+1}, Scaled training loss: {self.train_losses[epoch]:.4f}, Unscaled training loss: {self.train_losses_unscaled[epoch]:.4f}, Scaled validation loss: {self.val_losses[epoch]:.4f}, Unscaled validation loss {self.val_losses_unscaled[epoch]:.4f}"
            )
            # Core epoch metrics always get logged, even if the supplementary
            # variance-ratio metrics below fail for some reason.
            epoch_log = self.wandbLogging(epoch)

            variance_log = {}
            try:
                train_stats = self._computeVarianceStats(
                    train_preds_unscaled_list, train_targets_unscaled_list
                )
                val_stats = self._computeVarianceStats(
                    val_preds_unscaled_list, val_targets_unscaled_list
                )
                variance_log.update(
                    self._flattenVarianceStats("8", "Train", train_stats)
                )
                variance_log.update(self._flattenVarianceStats("9", "Val", val_stats))
            except Exception as exc:
                # Supplementary metrics only - never let a failure here drop
                # the core loss curves from this epoch's log.
                print(f"Epoch {epoch+1}: skipping variance-ratio metrics ({exc!r})")

            # single wandb.log() call per epoch, so the step axis stays aligned
            wandb.log({**epoch_log, **variance_log})

            early_stopper(self.val_losses[epoch], model)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break

        early_stopper.load_best_model(model)
        return model

    def test(self, model, X_test, test_params):
        model.eval()
        test_dataset = MlDataset(X_test, test_params)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        tot_test_loss = 0.0
        tot_test_loss_unscaled = 0.0
        tot_ind_test_loss = np.full(5, 0.0, dtype=np.float32)
        tot_ind_test_loss_unscaled = np.full(5, 0.0, dtype=np.float32)

        test_preds_unscaled_list = []
        test_targets_unscaled_list = []

        beamspots = createNewBeamspotsXy(
            LOWER, UPPER, N_POINTS_IN_XY, DISTRIBUTION_RADIUS
        )
        distance_between_beamspots = (UPPER - LOWER) / N_POINTS_IN_XY

        with torch.no_grad():
            for X_test_batch, params_test_batch in test_loader:
                y_batch = createRandomBeamspotAndTrackParameters(
                    beamspots,
                    params_test_batch,
                    noise_scale=distance_between_beamspots / 2,
                )
                X_test_gpu = X_test_batch.to(self.device)
                y_test_gpu = y_batch.to(self.device)
                test_outputs = self.getModelOutputs(
                    model, X_test_gpu, self.max_seq_len, self.device
                )
                tot_test_loss += (
                    self.criterion(test_outputs, y_test_gpu).item()
                    * X_test_batch.shape[0]
                )
                aver_unscaled_test_loss = self.getUnscaledLoss(
                    self.criterion, test_outputs, y_test_gpu
                )
                tot_test_loss_unscaled += (
                    aver_unscaled_test_loss * X_test_batch.shape[0]
                )

                aver_ind_output_losses = self.getIndividualScaledLosses(
                    test_outputs, y_test_gpu
                )
                aver_ind_unscaled_output_losses = self.getIndividualUnscaledLosses(
                    test_outputs, y_test_gpu
                )
                tot_ind_test_loss += aver_ind_output_losses * X_test_batch.shape[0]
                tot_ind_test_loss_unscaled += (
                    aver_ind_unscaled_output_losses * X_test_batch.shape[0]
                )

                # collect unscaled preds/targets for variance ratios
                pred_unscaled_t, y_unscaled_t = self._unscale_tensors(
                    test_outputs, y_test_gpu
                )
                test_preds_unscaled_list.append(pred_unscaled_t.detach().cpu())
                test_targets_unscaled_list.append(y_unscaled_t.detach().cpu())

        test_loss = tot_test_loss / len(test_dataset)
        test_loss_unscaled = tot_test_loss_unscaled / len(test_dataset)
        test_losses_ind = tot_ind_test_loss / len(test_dataset)
        test_losses_unscaled_ind = tot_ind_test_loss_unscaled / len(test_dataset)

        print(
            f"Scaled test loss: {test_loss:.4f}, Unscaled test loss: {test_loss_unscaled:.4f}"
        )  # , Individual scaled test losses: {test_losses_ind:.4f}, Individual unscaled test losses: {test_losses_unscaled_ind:.4f}")

        # Core test metrics always get logged, even if the supplementary
        # variance-ratio metrics below fail for some reason.
        test_log = {
            "Scaled test loss (MSE)": test_loss,
            "Unscaled test loss (MSE)": test_loss_unscaled,
            "6. Test loss d0 scaled": test_losses_ind[0],
            "6. Test loss z0 scaled": test_losses_ind[1],
            "6. Test loss phi scaled": test_losses_ind[2],
            "6. Test loss theta scaled": test_losses_ind[3],
            "6. Test loss q_over_p scaled": test_losses_ind[4],
            "7. Test loss d0 unscaled": test_losses_unscaled_ind[0],
            "7. Test loss z0 unscaled": test_losses_unscaled_ind[1],
            "7. Test loss phi unscaled": test_losses_unscaled_ind[2],
            "7. Test loss theta unscaled": test_losses_unscaled_ind[3],
            "7. Test loss q_over_p unscaled": test_losses_unscaled_ind[4],
        }

        try:
            test_stats = self._computeVarianceStats(
                test_preds_unscaled_list, test_targets_unscaled_list
            )
            test_log.update(self._flattenVarianceStats("10", "Test", test_stats))
        except Exception as exc:
            # TODO: ACTS logging?
            print(f"Test evaluation: skipping variance-ratio metrics ({exc!r})")

        wandb.log(test_log)

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from scipy.stats import norm
from config_noiseAnalysis import configurations

##### Global variables ######
num_of_pixel_rows = 16  # each EVENT has 16 pixel-rows stored as 16 columns

# Loads yprofiles into shape (N_events, num_of_pixel_rows).
def load_yprofiles(path: str):
    if path.endswith(".npy"):
        arr = np.load(path)
    elif path.endswith(".csv"):
        arr = pd.read_csv(path, header=None).values
    else:
        raise ValueError(f"Unsupported file type: {path}")

    arr = np.asarray(arr)
    if arr.ndim == 1:
        if arr.size != num_of_pixel_rows:
            raise ValueError(f"1D yprofile must have length {num_of_pixel_rows}, got {arr.size}")
        arr = arr.reshape(1, num_of_pixel_rows)

    if arr.ndim > 2:
        arr = np.squeeze(arr)

    if arr.ndim != 2 or arr.shape[1] != num_of_pixel_rows:
        raise ValueError(f"Expected shape (N_events,{num_of_pixel_rows}), got {arr.shape}")

    return arr.astype(float)

# collect noise values from rows where no charges are injected
def collect_noise_where_truth_zero(y_true: np.ndarray, y_exp: np.ndarray):
    if (y_true.shape[1] != num_of_pixel_rows) or (y_exp.shape[1] != num_of_pixel_rows):
        raise ValueError(f"Second dimension of yprofile array different from expected ({num_of_pixel_rows}).")
    
    per_pixel_row_noise: list[np.ndarray] = []
    projected_parts: list[np.ndarray] = []

    for r in range(num_of_pixel_rows):
        truth_zero_mask = (y_true[:, r] == 0)     # mask over EVENTS for this pixel-row r
        noise_vals = y_exp[truth_zero_mask, r]    # experimental values in those events
        per_pixel_row_noise.append(noise_vals)
        projected_parts.append(noise_vals)

    projected_noise = np.concatenate(projected_parts, axis=0) if projected_parts else np.array([])
    return per_pixel_row_noise, projected_noise


# collect all noise values, even from rows where charges are injected
def collect_all_noise(y_true: np.ndarray, y_exp: np.ndarray, injection_data=None):
    if (y_true.shape[1] != num_of_pixel_rows) or (y_exp.shape[1] != num_of_pixel_rows):
        raise ValueError(f"Second dimension of yprofile array different from expected ({num_of_pixel_rows}).")

    per_pixel_row_noise: list[np.ndarray] = []
    projected_parts: list[np.ndarray] = []

    for r in range(num_of_pixel_rows):
        noise_vals = y_exp[:, r] - y_true[:, r]  # subtract true yprofile from experimental charge values to extract noise values 
        per_pixel_row_noise.append(noise_vals)
        projected_parts.append(noise_vals)

    projected_noise = np.concatenate(projected_parts, axis=0) if projected_parts else np.array([])
    
    if injection_data is not None:
        per_pixel_row_noise_noInjection: list[np.ndarray] = []
        projected_parts_noInjection: list[np.ndarray] = []
        per_pixel_row_noise_withInjection: list[np.ndarray] = []
        projected_parts_withInjection: list[np.ndarray] = []
        mask_zero = (injection_data == 0)
        
        for r in range(num_of_pixel_rows):
            truth_zero_mask = (injection_data[:, r] == 0)  # mask over EVENTS for this pixel-row r
            noise_vals_no_injection = y_exp[truth_zero_mask, r]  # experimental values in those events
            per_pixel_row_noise_noInjection.append(noise_vals_no_injection)
            projected_parts_noInjection.append(noise_vals_no_injection)

            truth_non_zero_mask = (injection_data[:, r] != 0)  # mask for non-zero injection data
            noise_vals_with_injection = y_exp[truth_non_zero_mask, r]  # experimental values in those events
            per_pixel_row_noise_withInjection.append(noise_vals_with_injection)
            projected_parts_withInjection.append(noise_vals_with_injection)
        projected_noise_noInjection = np.concatenate(projected_parts_noInjection, axis=0) if projected_parts_noInjection else np.array([])
        projected_noise_withInjection = np.concatenate(projected_parts_withInjection, axis=0) if projected_parts_withInjection else np.array([])
        return per_pixel_row_noise, projected_noise, per_pixel_row_noise_noInjection, projected_noise_noInjection, per_pixel_row_noise_withInjection, projected_noise_withInjection
    else:
        return per_pixel_row_noise, projected_noise


def plot_histogram(values: np.ndarray, title: str, outpath: str, bins: int = 0, logy: bool = False, fit_gaussian: bool = False):
    assert np.all(np.equal(np.mod(values, 1), 0)), "Not all noise values are integers!"
    plt.figure(figsize=(7.5, 4.5))
    # Try to make bin edges along integers
    if bins == 0:
        if np.all(np.equal(np.mod(values, 1), 0)):
            bins = np.arange(values.min(), values.max() + 2)
    counts, bin_edges, _ = plt.hist(values, bins=bins, histtype="stepfilled", alpha=0.75, label="Data")
    # plt.hist(values, bins=bins, histtype="stepfilled", alpha=0.75)
    plt.title(title)
    plt.xlabel("Column-summed noise values (quantized)")
    plt.ylabel("Counts")
    plt.ylim(bottom=0.1)
    if logy:
        plt.yscale("log")
    
    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    stats_box_text = ''
    # Fit Gaussian
    if fit_gaussian:
        fit_min, fit_max = mu - 3 * sigma, mu + 3 * sigma
        fit_values = values[(values >= fit_min) & (values <= fit_max)]
        if fit_values.size > 1:
            mu_fit, sigma_fit = norm.fit(fit_values)
        else:
            print("\n\nNOTE: Not enough data points for fitting, using sample mean and std as fit parameters.\n\n")
            mu_fit, sigma_fit = mu, sigma

        x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
        y_fit = norm.pdf(x_fit, mu_fit, sigma_fit) * values.size * (bin_edges[1] - bin_edges[0])
        plt.plot(x_fit, y_fit, 'r--', label=fr'Gaussian fit\n$\mu$={mu_fit:.2f}, $\sigma$={sigma_fit:.2f}')

        # Chi2/ndf calculation
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        expected = norm.pdf(bin_centers, mu_fit, sigma_fit) * values.size * (bin_edges[1] - bin_edges[0])
        mask = counts > 0
        chi2 = np.sum(((counts[mask] - expected[mask]) ** 2) / expected[mask])
        ndf = np.sum(mask) - 2  # 2 fit parameters
        chi2_ndf = chi2 / ndf if ndf > 0 else np.nan
        stats_box_text = r'Fit $\chi^2$/ndf={chi2_ndf:.2e}'

    # Basic stats for text box
    zero_count = np.count_nonzero(values == 0)
    stats_text = (f'n={values.size}\nmean={mu:.3g}\nstd={sigma:.3g}\ncounts@0={zero_count}\nrate(0)={zero_count/values.size:.3%}\n' + stats_box_text)
    plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    plt.tight_layout()
    plt.grid()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_histograms_multiple(values_list: list[np.array], labels_list: list[str], title: str, outpath: str, bins: int = 0, logy: bool = False):
    plt.figure(figsize=(7.5, 4.5))
    for values, hist_label in zip(values_list, labels_list):
        print(values)
        assert np.all(np.equal(np.mod(values, 1), 0)), "Not all noise values are integers!"
        # Try to make bin edges along integers
        if isinstance(bins, int) and bins == 0:
            bins = np.arange(values.min(), values.max() + 2)
        plt.hist(values, bins=bins, histtype="step", linewidth=2, label=f'{hist_label}')
    
    plt.title(title)
    plt.xlabel("Column-summed noise values (quantized)")
    plt.ylabel("Counts")
    if logy:
        plt.yscale("log")
    
    # Display stats for each histogram separately
    n_stats = len(values_list)
    spacing = 0.9 / n_stats  # Spread boxes over 90% of the vertical axis
    for i, values in enumerate(values_list):
        mu = float(np.mean(values))
        sigma = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        zero_count = np.count_nonzero(values == 0)
        stats_text = (
            f'{labels_list[i]}: n={values.size}\nmean={mu:.3g}\nstd={sigma:.3g}\n'
            f'counts@0={zero_count}\nrate(0)={zero_count/values.size:.3%}'
        )
        y_pos = 0.95 - i * spacing
        plt.gca().text(
            0.95, y_pos, stats_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
        )
    
    plt.legend()
    plt.tight_layout()
    plt.grid()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()   

def plot_per_row_distributions(per_row_noise: list[np.ndarray], saveNamePrefix: str, outdir: str, bins: int = 0):
    for r, vals in enumerate(per_row_noise):
        if vals.size == 0:
            # Save empty plot when there is no data for that row
            plt.figure(figsize=(7.5, 2.8))
            plt.title(f"Row {r}: no samples where truth==0")
            plt.xlabel("Column-summed noise values (quantized)")
            plt.ylabel("Counts")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"noise_truth0_row{r:02d}.png"), dpi=150)
            plt.close()
            continue
        if "truth" in saveNamePrefix:
            title = f"Noise distribution (truth==0) — row {r}"
        else:
            title = f"Noise distribution (all events) — row {r}"
        plot_histogram(vals, title=title, outpath=os.path.join(outdir, f"{saveNamePrefix}_row{r:02d}.png"), bins=bins, logy=False)


def covariance_matrix_manual(x: np.ndarray):
    """
    Compute covariance matrix using the exact formula:
        cov[i,j] = 1/(N-1) * sum_k (x[k,i]-mean_i)*(x[k,j]-mean_j)
    Here:
      x shape: (N_events, num_of_pixel_rows)
      i,j index pixel-row (column index)
      k indexes events (rows)
    """
    if x.ndim != 2 or x.shape[1] != num_of_pixel_rows:
        raise ValueError(f"Expected x shape (N_events,{num_of_pixel_rows}), got {x.shape}")
    n = x.shape[0]
    print("Calculating covariance matrix: N_events =", n)
    if n < 2:
        raise ValueError("Need at least 2 events to compute covariance (N>=2).")

    mu = np.mean(x, axis=0, keepdims=True)     # (1,16)
    xc = x - mu                               # (N,16)
    cov = (xc.T @ xc) / (n - 1)               # (16,16)
    return cov

def plot_covariance_matrix(cov: np.ndarray, title: str, outpath: str, cmap: str = "viridis"):
    if cov.shape != (num_of_pixel_rows, num_of_pixel_rows):
        raise ValueError(f"Expected cov shape ({num_of_pixel_rows},{num_of_pixel_rows}), got {cov.shape}")

    plt.figure(figsize=(6.5, 5.5))
    im = plt.imshow(cov, cmap=cmap, origin="lower")
    plt.title(title)
    plt.xlabel("Pixel-row j")
    plt.ylabel("Pixel-row i")
    plt.xticks(np.arange(num_of_pixel_rows))
    plt.yticks(np.arange(num_of_pixel_rows))
    plt.colorbar(im, label="Covariance")
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, dpi=170)
    plt.close()

    
def correlation_matrix_manual(x: np.ndarray):
        """
        Compute a set of Pearson correlation coefficients using the exact formula:
                pcc[i,j] = sum_k((x[k,i]-mean_i)*(x[k,j]-mean_j)) /
                                     ((N-1) * std_i * std_j)
        Here:
            x shape: (N_events, num_of_pixel_rows)
            i,j index pixel-row (column index)
            k indexes events (rows)
        Returns:
            corr: (num_of_pixel_rows, num_of_pixel_rows) Pearson correlation matrix
        """
        if x.ndim != 2:
                raise ValueError(f"Expected 2D array, got {x.shape}")
        n, num_of_pixel_rows = x.shape
        if n < 2:
                raise ValueError("Need at least 2 events to compute correlation (N>=2).")

        mu = np.mean(x, axis=0, keepdims=True)     # (1, num_of_pixel_rows)
        xc = x - mu                               # (N_events, num_of_pixel_rows)
        cov = (xc.T @ xc) / (n - 1)               # (num_of_pixel_rows, num_of_pixel_rows)
        std = np.sqrt(np.diag(cov))               # (num_of_pixel_rows,)
        std[std == 0] = 1e-12                     # Avoid division by zero
        corr = cov / np.outer(std, std)           # (num_of_pixel_rows, num_of_pixel_rows)
        return corr


def plot_correlation_matrix(cov: np.ndarray, title: str, outpath: str, cmap: str = "viridis"):
    if cov.shape != (num_of_pixel_rows, num_of_pixel_rows):
        raise ValueError(f"Expected cov shape ({num_of_pixel_rows},{num_of_pixel_rows}), got {cov.shape}")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cov, cmap=cmap, origin="lower", vmin=-1, vmax=1)
    ax.set_title(title, pad=20)  # Increase pad to move title up
    ax.set_xlabel("Pixel-row j")
    ax.set_ylabel("Pixel-row i")
    ax.set_xticks(np.arange(num_of_pixel_rows))
    ax.set_yticks(np.arange(num_of_pixel_rows))
    cbar = fig.colorbar(im, ax=ax, label="Pearson correlation coefficient", fraction=0.046, pad=0.04)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for title
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, dpi=170)
    plt.close()

def find_yprofiles_path(noise_threshold: int, hlev: float):
    for config in configurations:
        nt, hl = config["noise_threshold_hlev"]
        if nt == noise_threshold and abs(hl - hlev) < 1e-6:
            return (config["yprofiles_path"], config["true_yprofiles_path"])
    return None


def main():
    parser = argparse.ArgumentParser(description="Noise Statistics Analysis")
    parser.add_argument('-n', '--noise-yprofile-analysis', action='store_true', help='Analyze noise statistics from yprofiles when was set to HLEV=0')
    parser.add_argument('-s', '--standalone-plots', nargs=2,type=float, metavar=('NOISE_THRESHOLD', 'HLEV'), help='Analyze noise distributions from a single noise-threshold and single HLEV value that was used during data-taking.')
    parser.add_argument('-c', '--combine-plots', action='store_true', help='Analyze noise distributions from different noise-thresholds and compare hists&fits in the same plot')
    args = parser.parse_args()
    args.n = args.noise_yprofile_analysis if args.noise_yprofile_analysis else None
    
    OUTDIR = "./noise_analysis_outputs_temp"
    os.makedirs(OUTDIR, exist_ok=True)
    # HLEV=0.6?
    # yprofile_at400e = '/mnt/local/CMSPIX28/data/ChipVersion1_ChipID23_SuperPix1/2026.01.29_07.16.26_DNN_vth0-0.012_vth1-0.082_vth2-0.129/yprofiles.csv'
    # yprofile_at700e = '/mnt/local/CMSPIX28/data/ChipVersion1_ChipID23_SuperPix1/2026.01.13_07.06.18_DNN_vth0-0.030_vth1-0.082_vth2-0.129/yprofiles.csv'
    # yprofile_at1000e = '/mnt/local/CMSPIX28/data/ChipVersion1_ChipID23_SuperPix1/2026.01.09_22.52.47_DNN_vth0-0.047_vth1-0.082_vth2-0.129/yprofiles.csv'
    # original_at400e = './tmp_16x16_centeredIncidence/400_1600_2400/yprofiles.npy'
    # original_at700e = './tmp_16x16_centeredIncidence/700_1600_2400/yprofiles.npy'
    # original_at1000e = './tmp_16x16_centeredIncidence/1000_1600_2400/yprofiles.npy'
    # HLEV=0?
    # yprofile_at400e = '/mnt/local/CMSPIX28/data/ChipVersion1_ChipID23_SuperPix1/2026.04.20_20.21.57_DNN_vth0-0.012_vth1-0.082_vth2-0.129/yprofiles.csv'
    # yprofile_at700e = '/mnt/local/CMSPIX28/data/ChipVersion1_ChipID23_SuperPix1/2026.04.22_12.47.56_DNN_vth0-0.030_vth1-0.082_vth2-0.129/yprofiles.csv'
    # yprofile_at1000e = '/mnt/local/CMSPIX28/data/ChipVersion1_ChipID23_SuperPix1/2026.04.21_22.01.46_DNN_vth0-0.047_vth1-0.082_vth2-0.129/yprofiles.csv'
    # original_at400e = './tmp_16x16_centeredIncidence/400_1600_2400/yprofiles.npy'
    # original_at700e = './tmp_16x16_centeredIncidence/700_1600_2400/yprofiles.npy'
    # original_at1000e = './tmp_16x16_centeredIncidence/1000_1600_2400/yprofiles.npy'
    # HLEV = 0.22
    yprofiles_fromASIC = []
    original_yprofiles = []
    thresholds = []

    if args.standalone_plots:
        noise_threshold, hlev = args.standalone_plots
        (yprofile_exp, yprofile_orig) = find_yprofiles_path(noise_threshold, hlev)
        if yprofile_exp:
            print(f"Found yprofiles_path: {yprofile_exp}")
        else:
            print("No matching configuration found.")
        yprofiles_fromASIC.append(yprofile_exp)
        original_yprofiles.append(yprofile_orig)
        thresholds.append(int(noise_threshold))
    if args.combine_plots:
        thresholds_tmp = [400, 700]
        for thresh in thresholds_tmp:
            for config in configurations:
                nt, _ = config["noise_threshold_hlev"]
                if nt == thresh and config.get("optimized", True):
                    yprofiles_fromASIC.append(config["yprofiles_path"])
                    original_yprofiles.append(config["true_yprofiles_path"])
                    thresholds.append(int(config["noise_threshold_hlev"][0]))
    # original_at400e = '/asic/projects/C/CMS_PIX_28/benjamin/verilog/workarea/cms28_smartpix_verification/PnR_cms28_smartpix_verification_D/tb/dnn/csv/l6/input_1.csv'
    for asicYprofile_path, truthYprofile_path, thresh in zip(yprofiles_fromASIC, original_yprofiles, thresholds):
        NO_INJECTION_DIR = os.path.join(OUTDIR, "truth0_results_" + str(thresh) + "e")
        os.makedirs(NO_INJECTION_DIR, exist_ok=True)
        INJECTION_DIR = os.path.join(OUTDIR, "all_events_results_" + str(thresh) + "e")
        os.makedirs(INJECTION_DIR, exist_ok=True)

        y_true = load_yprofiles(truthYprofile_path)
        y_exp = load_yprofiles(asicYprofile_path)
        if y_true.shape != y_exp.shape:
            print(f"Shape mismatch: y_true {y_true.shape} vs y_exp {y_exp.shape}")
            y_true = y_true[:y_exp.shape[0]]  # Select the same number of rows as y_exp
        if args.n:
            # NOTE: REMOVE WHEN NOT LOOKING AT PURE NOISE DATA
            y_true = np.zeros_like(y_true)
            injection_data = load_yprofiles(truthYprofile_path)[:y_exp.shape[0]]  # Load the original yprofiles as injection data for the all-noise analysis
        
        # Commenting out for now because we've recently not been looking at these plots, and that they need to be modified when args.c is passed.
        # # ===========================================================
        # # Analysis on noise, in pixel rows where no charge is injected
        # # ===========================================================
        # # 1) Per-row distributions
        # per_row_noise, projected_noise = collect_noise_where_truth_zero(y_true, y_exp)
        # # 2) Projected distribution (all rows combined)
        # plot_per_row_distributions(per_row_noise, saveNamePrefix = "noise_truth0", outdir=NO_INJECTION_DIR)
        # if projected_noise.size == 0:
        #     raise ValueError("No samples found where truth==0 (projected). Nothing to plot.")
        # plot_histogram(projected_noise, title=f"Projected noise distribution (truth==0) — all rows", outpath=os.path.join(NO_INJECTION_DIR, "noise_truth0_projected.png"), logy=False)
        # plot_histogram(projected_noise, title=f"Projected noise distribution (truth==0) — all rows (log y)", outpath=os.path.join(NO_INJECTION_DIR, "noise_truth0_projected_logy.png"), logy=True)
        # # 3 ) Covariance matrix
        # # We only consider y_exp values for pixel-rows where y_true == 0.
        # mask = (y_true == 0)  # shape (N_events, 16)
        # x_cov = np.where(mask, y_exp, 0)  # shape (N_events, 16)        
        # cov = covariance_matrix_manual(x_cov)
        # plot_covariance_matrix(cov, title=f"Covariance of y_exp for truth==0 events, threshold={thresh}e", outpath=os.path.join(NO_INJECTION_DIR, f"cov_truth0_{thresh}e.png"), cmap="viridis")
        # # np.save(os.path.join(NO_INJECTION_DIR, f"cov_truth0_only_{thresh}e.npy"), cov)

        # ===========================================================
        # Analysis on noise, in all pixel rows. 
        # ===========================================================
        # In rows where charges are injected, noise = y_exp - y_true. In rows where no charge is injected, this reduces to y_exp (same as previous section).
        # 1) Per-row distributions
        if args.n:
            per_row_noise_all, projected_noise_all, per_row_noise_noInjection, projected_parts_noInjection, per_row_noise_withInjection, projected_parts_withInjection = collect_all_noise(y_true, y_exp, injection_data)
        else:
            per_row_noise_all, projected_noise_all = collect_all_noise(y_true, y_exp)
        # 2) Projected distribution (all rows combined)
        plot_per_row_distributions(per_row_noise_all, saveNamePrefix = "noise_all", outdir=INJECTION_DIR)
        if projected_noise_all.size == 0:
            raise ValueError("No samples found (projected). Nothing to plot.")
        fit_gaussian = False if args.standalone_plots is not None else True
        plot_histogram(projected_noise_all, title=f"Projected noise distribution — all rows", outpath=os.path.join(INJECTION_DIR, "noise_all_projected.png"), logy=False, fit_gaussian=fit_gaussian)
        plot_histogram(projected_noise_all, title=f"Projected noise distribution — all rows (log y)", outpath=os.path.join(INJECTION_DIR, "noise_all_projected_logy.png"), logy=True, fit_gaussian=fit_gaussian)
        if args.n:
            plot_histograms_multiple([projected_noise_all, projected_parts_noInjection, projected_parts_withInjection], labels_list=["All rows", "Rows with no injection", "Rows with injection"], title=f"Projected noise distribution — all rows", outpath=os.path.join(INJECTION_DIR, "noise_all_projected_basedOnInjection.png"))
            plot_histograms_multiple([projected_noise_all, projected_parts_noInjection, projected_parts_withInjection], labels_list=["All rows", "Rows with no injection", "Rows with injection"], title=f"Projected noise distribution — all rows", outpath=os.path.join(INJECTION_DIR, "noise_all_projected_basedOnInjection_logy.png"), logy=True)
        # 3 ) Covariance matrix
        residuals = y_exp - y_true  # shape (N_events, 16)
        cov = covariance_matrix_manual(residuals)
        plot_covariance_matrix(cov, title=f"Covariance of residuals (y_exp - y_true), threshold={thresh}e", outpath=os.path.join(INJECTION_DIR, f"cov_residuals_{thresh}e.png"), cmap="viridis")
        corr_pearson = correlation_matrix_manual(residuals)
        plot_correlation_matrix(corr_pearson, title=f"Pearson correlation of residuals (y_exp - y_true), threshold={thresh}e", outpath=os.path.join(INJECTION_DIR, f"corr_residuals_{thresh}e.png"), cmap="viridis")
        # save the raw noise samples for downstream fits (Gaussian, Landau, etc.)
        # np.save(os.path.join(OUTDIR, "noise_all_projected.npy"), projected_noise_all)
        # for r, vals in enumerate(per_row_noise_all):
        #     np.save(os.path.join(OUTDIR, f"noise_all_row{r:02d}.npy"), vals)
        # np.save(os.path.join(INJECTION_DIR, f"cov_residuals_{thresh}e.npy"), cov)

        print(f"Wrote outputs to: {OUTDIR}")


if __name__ == "__main__":
    main()
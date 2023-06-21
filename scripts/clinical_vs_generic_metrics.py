import argparse
from pathlib import Path
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from XrayTo3DShape import MODEL_NAMES, filter_wandb_run, get_run_from_model_name

plt.style.use(["science", "no-latex"])
# Increase the resolution of all the plots below
plt.rcParams.update({"figure.dpi": 150})


def plot_ci_bootstrap(xs, ys, resid, nboot=500, ax=None):
    """Return an axes of confidence bands using a bootstrap approach.

    Notes
    -----
    The bootstrap approach iteratively resampling residuals.
    It plots `nboot` number of straight lines and outlines the shape of a band.
    The density of overlapping lines indicates improved confidence.

    Returns
    -------
    ax : axes
        - Cluster of lines
        - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers

    References
    ----------
    .. [1] J. Stults. "Visualizing Confidence Intervals", Various Consequences.
       http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html

    """
    if ax is None:
        ax = plt.gca()

    bootindex = sp.random.randint

    for _ in range(nboot):
        resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
        # Make coeffs of for polys
        pc = np.polyfit(xs, ys + resamp_resid, 1)
        # Plot bootstrap cluster
        ax.plot(xs, np.polyval(pc, xs), "b-", linewidth=2, alpha=3.0 / float(nboot))

    return ax


hip_COLUMNS = ["ASIS_L", "ASIS_R", "PT_L", "PT_R", "IS_L", "IS_R", "PSIS_L", "PSIS_R"]
vertebra_COLUMNS = ["spl", "spa", "avbh", "pvbh", "svbl", "ivbl", "vcl"]
femur_COLUMNS = [
    "FHR",
    "FHC",
    "NSA",
    "FNA_x",
    "FNA_y",
    "FNA_z",
    "FDA_x",
    "FDA_y",
    "FDA_z",
]
anatomy_wise_details = {
    "hip": {
        "columns": hip_COLUMNS,
        "clinical_log_name": "hip_landmark_error.csv",
        "subject_id_post_fix": "_hip_msk",
    },
    "vertebra": {
        "columns": vertebra_COLUMNS,
        "clinical_log_name": "vertebra_morphometry_error.csv",
        "subject_id_post_fix": "-seg-vert-msk",
    },
    "femur": {
        "columns": femur_COLUMNS,
        "clinical_log_name": "femur_morphometry_error.csv",
        "subject_id_post_fix": "s0174_femur_righ",  # this is a prefix
    },
}
ROWS = [
    "DSC",
    "NSD",
    "ASD",
    "HD95",
]


generalized_metrics_template = "runs/2d-3d-benchmark/{run_id}/evaluation/metric-log.csv"
clinical_parameters_template = (
    "runs/2d-3d-benchmark/{run_id}/evaluation/{clinical_log_name}"
)

parser = argparse.ArgumentParser()
parser.add_argument("anatomy", choices=["hip", "vertebra", "femur", "rib"])
args = parser.parse_args()


ANATOMY = args.anatomy
runs = filter_wandb_run(anatomy=ANATOMY, tags=["model-compare", "dropout"])

# save model wise
for model in MODEL_NAMES:
    run = get_run_from_model_name(model, runs)
    clinical_csv = pd.read_csv(
        clinical_parameters_template.format(
            run_id=run.id,
            clinical_log_name=anatomy_wise_details[ANATOMY]["clinical_log_name"],
        )
    )
    generalized_metric_csv = pd.read_csv(
        generalized_metrics_template.format(
            run_id=run.id,
        )
    )

    post_fix = anatomy_wise_details[ANATOMY]["subject_id_post_fix"]

    if ANATOMY == "femur":
        post_fix_length = len(post_fix)  # this is actually a prefix so no negative sign
    else:
        post_fix_length = -len(post_fix)
    generalized_metric_csv["id"] = generalized_metric_csv["subject-id"].str[
        :post_fix_length
    ]
    merged_csv = pd.merge(clinical_csv, generalized_metric_csv, on="id", how="left")

    subplot_sz = 5
    rows = 1
    cols = len(anatomy_wise_details[ANATOMY]["columns"])
    fig, ax = plt.subplots(rows, cols, figsize=(cols * subplot_sz, rows * subplot_sz))
    rw = "DSC"
    for clm_idx, clm in enumerate(anatomy_wise_details[ANATOMY]["columns"]):
        threshold = merged_csv[clm].quantile(0.90)
        # merged_filtered_csv = merged_csv
        merged_filtered_csv = merged_csv[merged_csv[clm] < threshold].dropna()

        dsc = merged_filtered_csv[rw]
        y = merged_filtered_csv[clm]

        regressor = LinearRegression().fit(dsc.values.reshape(-1, 1), y.values)
        y_pred = regressor.predict(dsc.values.reshape(-1, 1))
        r2 = r2_score(y, y_pred)
        rho = np.corrcoef(dsc, y)[0, 1]
        residual = y - y_pred
        plot_ci_bootstrap(
            np.asarray(dsc.values.tolist()),
            np.asarray(y.values.tolist()),
            np.asarray(residual.values.tolist()),
            ax=ax[clm_idx],
        )
        ax[clm_idx].plot(dsc, y_pred)
        ax[clm_idx].scatter(dsc, y, s=subplot_sz * 5)

        ax[clm_idx].set_title(r"$R^2={:.2f},\rho={:.2f}$".format(r2, rho), fontsize=25)
        ax[clm_idx].set_xlabel(rw, fontsize=25)
        ax[clm_idx].set_ylabel(clm, fontsize=25)
        ax[clm_idx].xaxis.set_tick_params(labelsize=25)
        ax[clm_idx].yaxis.set_tick_params(labelsize=25)
    plt.tight_layout()
    out_file = f"scripts/clinical_vs_general_metrics/{ANATOMY}/{model}_{ANATOMY}_{rw}_clinical_relationship.pdf"
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)

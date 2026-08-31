# June 2026
# validation european databse, deep learning for disturbance detection
# @aviana @csenf

# Packages 
import numpy as np
import pandas as pd

reference_database = pd.read_csv("/.../disturbance_database.csv")

predictions_unet = pd.read_csv("/.../validation_samples_predictions_Unet_w5")
predictions_tempcnn = pd.read_csv("/.../validation_samples_predictions_tempcnn_w5")
predictions_rfw5 = pd.read_csv("/.../validation_samples_predictions_RF_w5")
predictions_efdav2 = pd.read_csv("/.../validation_samples_efdav21")

# Parameters ----
niter = 1000
n_samples_country = 400

# Sampling weights ----
weights = pd.read_csv("/.../sample_weights.csv",  sep=",", low_memory=False)

weights = (
    weights[
        ["stratum", "count", "weight", "country"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

weights_forest = (
    weights.groupby("country", as_index=False)
    .agg(count=("count", "sum")))

weights_forest["weight"] = (weights_forest["count"] / weights_forest["count"].sum())



### ** Spatial validation** ###
print("***************************************************")
def compute_spatial_accuracy(
    reference_database,
    predictions,
    weights,
    weights_forest,
    model_name = "Model",
    niter=niter,
    n_samples_country=n_samples_country,
):

    # validation_dat_spatial

    predictions_small = predictions[["unique_id", "year", "prediction"]].copy()

    reference_database["unique_id"] = reference_database["unique_id"].astype(str)
    predictions_small["unique_id"] = predictions_small["unique_id"].astype(str)
    
    validation_dat_spatial = (
        reference_database
        .merge(predictions_small, on=["unique_id", "year"], how="left")
        .loc[lambda x: x["prediction"].notna()]
        .groupby(["unique_id", "country", "stratum"], as_index=False)
        .agg(
            reference=("disturbance", lambda x: int(x.sum() > 0)),
            prediction=("prediction", lambda x: int(x.sum() > 0))
        )
    )

    # bootstrap iterations

    acc_overall_collector = []

    weights_join = (
        weights
        .assign(
            stratum=lambda x: np.where(
                x["stratum"] == "stable",
                0,
                1
            )
        )
        .drop(columns=["count"], errors="ignore")
    )

    forest_weights = (
        weights_forest
        .rename(columns={"weight": "weight_forest"})
        .drop(columns=["count"], errors="ignore")
    )

    for k in range(niter):

        # sample within country

        validation_dat_random = (validation_dat_spatial.merge(weights_join, on=["country", "stratum"], how="left"))

        sampled_countries = []

        for country, df_country in validation_dat_random.groupby("country"):

            sampled = df_country.sample(n=n_samples_country,replace=True, weights=df_country["weight"], random_state=None)

            sampled_countries.append(sampled)

        validation_dat_random = pd.concat(sampled_countries, ignore_index=True)

        # sample by forest area weights
        validation_dat_random = (validation_dat_random.merge(forest_weights,on="country",how="left"))

        validation_dat_random = validation_dat_random.sample(
            n=len(validation_dat_random),
            replace=True,
            weights=validation_dat_random["weight_forest"],
        )

        # confusion matrix counts
        tp = ((validation_dat_random["reference"] == 1) & (validation_dat_random["prediction"] == 1)).sum()
        tn = ((validation_dat_random["reference"] == 0) & (validation_dat_random["prediction"] == 0)).sum()
        fp = ((validation_dat_random["reference"] == 0) & (validation_dat_random["prediction"] == 1)).sum()
        fn = ((validation_dat_random["reference"] == 1) & (validation_dat_random["prediction"] == 0)).sum()

        total = tp + tn + fp + fn
        
        oa = (tp + tn) / total

        precision_d = tp/(tp+fp) if (tp+fp)>0 else np.nan
        recall_d    = tp/(tp+fn) if (tp+fn)>0 else np.nan
        
        precision_s = tn/(tn+fn) if (tn+fn)>0 else np.nan
        recall_s    = tn/(tn+fp) if (tn+fp)>0 else np.nan
        
        f1_disturbance = (
            2*precision_d*recall_d/(precision_d+recall_d)
            if (precision_d+recall_d)>0 else np.nan
        )
        
        f1_stable = (
            2*precision_s*recall_s/(precision_s+recall_s)
            if (precision_s+recall_s)>0 else np.nan
        )
        f1_macro = (f1_disturbance + f1_stable) / 2

        acc_overall_collector.append(
            {
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "sum": total,
                "oo": 1 - (tp + tn) / total,
                "ce_disturbance": 1 - tp / (tp + fp)
                if (tp + fp) > 0 else np.nan,
                "ce_stable": 1 - tn / (tn + fn)
                if (tn + fn) > 0 else np.nan,
                "oe_disturbance": 1 - tp / (tp + fn)
                if (tp + fn) > 0 else np.nan,
                "oe_stable": 1 - tn / (tn + fp)
                if (tn + fp) > 0 else np.nan,
                "oa": oa,
                "f1_disturbance": f1_disturbance,
                "f1_stable": f1_stable,
                "f1_macro": f1_macro,
            }
        )
    # ---------------------------------------------------------

    acc_overall = pd.DataFrame(acc_overall_collector)
    # pivot_longer

    acc_long = acc_overall.melt(
    value_vars=[
        "oa",
        "oo",
        "ce_disturbance",
        "ce_stable",
        "oe_disturbance",
        "oe_stable",
        "f1_disturbance",
        "f1_stable",
        "f1_macro",
    ],
    var_name="metric",
    value_name="value",
    )

    # separate(metric, c("metric","class"))
    split_cols = acc_long["metric"].str.split("_",n=1,expand=True)

    acc_long["metric"] = split_cols[0]
    acc_long["class"] = split_cols[1]

    # oo has no class
    acc_long["class"] = acc_long["class"].fillna("all")

    # summarize(mean, sd)

    spatial_accuracy = (
        acc_long
        .groupby(
            ["metric", "class"],
            dropna=False,
            as_index=False
        )
        .agg(
            mean=("value", "mean"),
            sd=("value", "std"),
            ci_low=("value", lambda x: np.nanpercentile(x, 2.5)),
            ci_high=("value", lambda x: np.nanpercentile(x, 97.5)),
        )
    )

    # mutate labels
    spatial_accuracy["metric"] = (
        spatial_accuracy["metric"]
        .replace(
            {
                "ce": "Commission",
                "oe": "Omission",
                "oo": "Overall",
                "oa": "Overall accuracy",
                "f1": "F1 score",
            }
        )
    )
    spatial_accuracy["class"] = (
        spatial_accuracy["class"]
        .replace(
            {
                "disturbance": "Disturbed pixels",
                "stable": "Undisturbed pixels",
                "macro": "All pixels",
                "all": "All pixels",
            }
        )
    )
    
    spatial_accuracy["model"] = model_name
    print(f"\nBootstrap summary — {model_name}:")
    for _, row in spatial_accuracy.iterrows():
        metric = row["metric"]
        cls = row["class"]
    
        if metric == "F1 score":
            print(
                f"{metric} - {cls}: "
                f"{row['mean']:.3f} "
                f"(95% CI: {row['ci_low']:.3f}–{row['ci_high']:.3f})"
            )
        else:
            print(
                f"{metric} - {cls}: "
                f"{100 * row['mean']:.1f}% "
                f"(95% CI: {100 * row['ci_low']:.1f}–{100 * row['ci_high']:.1f}%)"
            )
    return spatial_accuracy

''' run spatial validation'''
spatial_accuracy_unet = compute_spatial_accuracy(reference_database, predictions_unet,weights, weights_forest,niter=niter, 
                                                 n_samples_country=n_samples_country, model_name="1D U-Net W5")

spatial_accuracy_tempcnn = compute_spatial_accuracy(reference_database, predictions_tempcnn, weights, weights_forest, niter=niter, 
                                                    n_samples_country=n_samples_country, model_name="TempCNN W5")

spatial_accuracy_rfw5 = compute_spatial_accuracy(reference_database, predictions_rfw5, weights, weights_forest, niter=niter, 
                                                 n_samples_country=n_samples_country, model_name="Random Forest W5")

spatial_accuracy_efdav2 = compute_spatial_accuracy(reference_database, predictions_efdav2, weights, weights_forest, niter=niter, 
                                                   n_samples_country=n_samples_country, model_name="EFDA v2.1")

''' combine them'''
spatial_accuracy = pd.concat(
    [
        spatial_accuracy_unet.assign(model="1D U-Net W5"),
        spatial_accuracy_tempcnn.assign(model="TempCNN W5"),
        spatial_accuracy_rfw5.assign(model="Random Forest W5"),
        spatial_accuracy_efdav2.assign(model="EFDA v2.1"),
    ],
    ignore_index=True,
)


### ** Temporal validation ** ###
print("***************************************************")
# Parameters ----
niter = 1000
n_samples_country = 400

# Sampling weights ----

weights = (
    weights[
        ["stratum", "count", "weight", "country"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

weights_forest = (
    weights.groupby("country", as_index=False)
    .agg(count=("count", "sum")))

weights_forest["weight"] = (weights_forest["count"] / weights_forest["count"].sum())

def safe_div(num, den):
    return num / den if den != 0 else np.nan

def compute_temporal_accuracy(reference_database, predictions, weights, weights_forest, niter=30, n_samples_country=n_samples_country, years=range(1985, 2024)):

    acc_overall_summary = []

    weights_join = weights.assign(stratum=lambda x: np.where(x["stratum"] == "stable", 0, 1)).drop(columns=["count"], errors="ignore")
    forest_weights = weights_forest.rename(columns={"weight": "weight_forest"}).drop(columns=["count"], errors="ignore")

    for year in years:

        a = reference_database[(reference_database["year"] == year) & (reference_database["unique_id"].isin(predictions["unique_id"].unique()))]
        b = predictions.loc[predictions["year"] == year, ["unique_id", "year", "prediction"]]

        validation_dat = a.merge(b, on=["unique_id", "year"], how="left")

        acc_overall_collector = []

        for k in range(niter):

            validation_dat_random = validation_dat.merge(weights_join, on=["country", "stratum"], how="left")

            sampled_countries = [
                df.sample(n=n_samples_country, replace=True, weights=df["weight"])
                for _, df in validation_dat_random.groupby("country")
            ]

            validation_dat_random = pd.concat(sampled_countries, ignore_index=True)

            validation_dat_random = validation_dat_random.merge(forest_weights, on="country", how="left")
            validation_dat_random = validation_dat_random.sample(n=len(validation_dat_random), replace=True, weights=validation_dat_random["weight_forest"])

            tp = ((validation_dat_random["disturbance"] == 1) & (validation_dat_random["prediction"] == 1)).sum()
            tn = ((validation_dat_random["disturbance"] == 0) & (validation_dat_random["prediction"] == 0)).sum()
            fp = ((validation_dat_random["disturbance"] == 0) & (validation_dat_random["prediction"] == 1)).sum()
            fn = ((validation_dat_random["disturbance"] == 1) & (validation_dat_random["prediction"] == 0)).sum()

            total = tp + tn + fp + fn

            acc_overall_collector.append({
                "oo": 1 - safe_div(tp + tn, total),
                "ce_disturbance": 1 - safe_div(tp, tp + fp),
                "ce_stable": 1 - safe_div(tn, tn + fn),
                "oe_disturbance": 1 - safe_div(tp, tp + fn),
                "oe_stable": 1 - safe_div(tn, tn + fp),
            })

        acc_overall = pd.DataFrame(acc_overall_collector)

        acc_long = acc_overall.melt(var_name="metric", value_name="value")
        acc_long[["metric", "class"]] = acc_long["metric"].str.split("_", n=1, expand=True)
        acc_long["class"] = acc_long["class"].fillna("all")

        summary = acc_long.groupby(["metric", "class"], as_index=False).agg(mean=("value", "mean"), sd=("value", "std"))

        summary["metric"] = summary["metric"].replace({"ce": "Commission", "oe": "Omission", "oo": "Overall"})
        summary["class"] = summary["class"].replace({"disturbance": "Disturbed pixels", "stable": "Undisturbed pixels", "all": "All pixels"})
        summary["year"] = year

        acc_overall_summary.append(summary)

    return pd.concat(acc_overall_summary, ignore_index=True)

''' run temporal validation'''
temporal_accuracy_unet = compute_temporal_accuracy(reference_database, predictions_unet, weights, weights_forest, niter=niter, n_samples_country=400)

temporal_accuracy_tempcnn = compute_temporal_accuracy(reference_database, predictions_tempcnn, weights, weights_forest, niter=niter, n_samples_country=400)

temporal_accuracy_rfw5 = compute_temporal_accuracy(reference_database, predictions_rfw5, weights, weights_forest, niter=niter, n_samples_country=400)

temporal_accuracy_efdav2 = compute_temporal_accuracy(reference_database, predictions_efdav2, weights, weights_forest, niter=niter, n_samples_country=400)


''' combine them'''
acc_overall_summary_df = pd.concat([
    temporal_accuracy_unet.assign(model="1D U-Net W5"),
    temporal_accuracy_tempcnn.assign(model="TempCNN W5"),
    temporal_accuracy_rfw5.assign(model="Random Forest W5"),
    temporal_accuracy_efdav2.assign(model="EFDA v2.1"),
], ignore_index=True)

acc_overall_summary_df_average = (
    acc_overall_summary_df
    .groupby(["metric", "class", "model"], as_index=False)
    .agg(mean=("mean", "mean"), sd=("sd", lambda x: np.sqrt(np.nanmean(x ** 2))))
)

### ** Temporal validation ** ### --> per decade
print("***************************************************")
def compute_decadal_accuracy(
    reference_database,
    predictions,
    weights,
    weights_forest,
    niter=1000,
    n_samples_country=400,
    decades=None
):

    if decades is None:
        decades = [
            range(1985, 1995),
            range(1995, 2005),
            range(2005, 2015),
            range(2015, 2025)
        ]

    acc_summary_collector = []

    weights_join = (
        weights
        .assign(stratum=lambda x: np.where(x["stratum"] == "stable", 0, 1))
        .drop(columns=["count"], errors="ignore")
    )

    forest_weights = (
        weights_forest
        .rename(columns={"weight": "weight_forest"})
        .drop(columns=["count"], errors="ignore")
    )

    for dec in decades:

        dec = list(dec)
        decade_label = f"{min(dec)}-{max(dec)}"

        validation_dat_spatial = (
            reference_database
            .merge(predictions, on=["unique_id", "year"], how="left")
            .dropna(subset=["prediction"])
        )

        validation_dat_spatial = (
            validation_dat_spatial[
                validation_dat_spatial["year"].isin(dec)
            ]
            .groupby(["unique_id", "country", "stratum"], as_index=False)
            .agg(
                reference=("disturbance", lambda x: int(x.sum() > 0)),
                prediction=("prediction", lambda x: int(x.sum() > 0))
            )
        )

        acc_overall_collector = []

        for k in range(niter):

            validation_dat_random = validation_dat_spatial.merge(
                weights_join,
                on=["country", "stratum"],
                how="left"
            )

            sampled_countries = [
                df.sample(
                    n=n_samples_country,
                    replace=True,
                    weights=df["weight"]
                )
                for _, df in validation_dat_random.groupby("country")
            ]

            validation_dat_random = pd.concat(sampled_countries, ignore_index=True)

            validation_dat_random = validation_dat_random.merge(
                forest_weights,
                on="country",
                how="left"
            )

            validation_dat_random = validation_dat_random.sample(
                n=len(validation_dat_random),
                replace=True,
                weights=validation_dat_random["weight_forest"]
            )

            tp = ((validation_dat_random["reference"] == 1) & 
                  (validation_dat_random["prediction"] == 1)).sum()

            tn = ((validation_dat_random["reference"] == 0) & 
                  (validation_dat_random["prediction"] == 0)).sum()

            fp = ((validation_dat_random["reference"] == 0) & 
                  (validation_dat_random["prediction"] == 1)).sum()

            fn = ((validation_dat_random["reference"] == 1) & 
                  (validation_dat_random["prediction"] == 0)).sum()

            total = tp + tn + fp + fn

            acc_overall_collector.append({
                "oo": 1 - safe_div(tp + tn, total),
                "ce_disturbance": 1 - safe_div(tp, tp + fp),
                "ce_stable": 1 - safe_div(tn, tn + fn),
                "oe_disturbance": 1 - safe_div(tp, tp + fn),
                "oe_stable": 1 - safe_div(tn, tn + fp),
            })

        acc_overall = pd.DataFrame(acc_overall_collector)

        acc_long = acc_overall.melt(
            var_name="metric",
            value_name="value"
        )

        acc_long[["metric", "class"]] = acc_long["metric"].str.split(
            "_",
            n=1,
            expand=True
        )

        acc_long["class"] = acc_long["class"].fillna("all")

        summary = (
            acc_long
            .groupby(["metric", "class"], as_index=False)
            .agg(
                mean=("value", "mean"),
                sd=("value", "std")
            )
        )

        summary["metric"] = summary["metric"].replace({
            "ce": "Commission",
            "oe": "Omission",
            "oo": "Overall"
        })

        summary["class"] = summary["class"].replace({
            "disturbance": "Disturbed pixels",
            "stable": "Undisturbed pixels",
            "all": "All pixels"
        })

        summary["decade"] = decade_label

        acc_summary_collector.append(summary)

    return pd.concat(acc_summary_collector, ignore_index=True)

decadal_accuracy_unet = compute_decadal_accuracy(reference_database,predictions_unet_collapsed,weights,weights_forest,niter=niter,
                                                 n_samples_country=400)
decadal_accuracy_tempcnn = compute_decadal_accuracy(reference_database,predictions_tempcnn_collapsed,weights,weights_forest, niter=niter,
                                                    n_samples_country=400)

decadal_accuracy_rfw5 = compute_decadal_accuracy(reference_database,predictions_rfw5_collapsed,weights,weights_forest,niter=niter,
                                                 n_samples_country=400)

decadal_accuracy_efdav2 = compute_decadal_accuracy(reference_database,predictions_efdav2_collapsed,weights,weights_forest, niter=niter,
                                                   n_samples_country=400)

# combine them
decadal_accuracy = pd.concat([
    decadal_accuracy_unet.assign(model="1D U-Net W5"),
    decadal_accuracy_tempcnn.assign(model="TempCNN W5"),
    decadal_accuracy_rfw5.assign(model="Random Forest W5"),
    decadal_accuracy_efdav2.assign(model="EFDA v2.1"),
], ignore_index=True)

decadal_accuracy_average = (
    decadal_accuracy
    .groupby(["metric", "class", "model", "decade"], as_index=False)
    .agg(
        mean=("mean", "mean"),
        sd=("sd", lambda x: np.sqrt(np.nanmean(x ** 2)))
    )
)


### ** Per country validation ** ###
print("***************************************************")
# Parameters ----
niter = 1000
n_samples_country = 400

# Sampling weights ----

weights = (
    weights[
        ["stratum", "count", "weight", "country"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

weights_forest = (
    weights.groupby("country", as_index=False)
    .agg(count=("count", "sum")))

weights_forest["weight"] = (weights_forest["count"] / weights_forest["count"].sum())


''' country level'''

def safe_div(num, den):
    return num / den if den != 0 else np.nan


def compute_country_accuracy(reference_database, predictions, weights, niter=niter, n_samples_country=n_samples_country):
    
    predictions_small = predictions[["unique_id", "year", "prediction"]].copy()

    reference_database["unique_id"] = reference_database["unique_id"].astype(str)
    predictions_small["unique_id"] = predictions_small["unique_id"].astype(str)
    
    validation_dat_spatial = (
        reference_database
        .merge(predictions_small, on=["unique_id", "year"], how="left")
        .loc[lambda x: x["prediction"].notna()]
        .groupby(["unique_id", "country", "stratum"], as_index=False)
        .agg(
            reference=("disturbance", lambda x: int(x.sum() > 0)),
            prediction=("prediction", lambda x: int(x.sum() > 0))
        )
    )
    
    print("validation_dat_spatial rows:", len(validation_dat_spatial))
    print("shared columns:", reference_database.columns.intersection(predictions.columns).tolist())
    print(validation_dat_spatial.head())
    
    if validation_dat_spatial.empty:
        raise ValueError("validation_dat_spatial is empty: the merge with predictions did not match.")
    
    weights_join = weights.assign(stratum=lambda x: np.where(x["stratum"] == "stable", 0, 1)).drop(columns=["count"], errors="ignore")
    acc_overall_collector = []

    for k in range(niter):
        validation_dat_random = validation_dat_spatial.merge(weights_join, on=["country", "stratum"], how="left")

        validation_dat_random = pd.concat([
            df.sample(n=n_samples_country, replace=True, weights=df["weight"])
            for _, df in validation_dat_random.groupby("country")
        ], ignore_index=True)

        acc = validation_dat_random.groupby("country").apply(
            lambda df: pd.Series({
                "tp": ((df["reference"] == 1) & (df["prediction"] == 1)).sum(),
                "tn": ((df["reference"] == 0) & (df["prediction"] == 0)).sum(),
                "fp": ((df["reference"] == 0) & (df["prediction"] == 1)).sum(),
                "fn": ((df["reference"] == 1) & (df["prediction"] == 0)).sum(),
            })
        ).reset_index()

        acc["sum"] = acc["tp"] + acc["tn"] + acc["fp"] + acc["fn"]
        acc["oo"] = 1 - (acc["tp"] + acc["tn"]) / acc["sum"]
        acc["ce_disturbance"] = 1 - acc.apply(lambda x: safe_div(x["tp"], x["tp"] + x["fp"]), axis=1)
        acc["ce_stable"] = 1 - acc.apply(lambda x: safe_div(x["tn"], x["tn"] + x["fn"]), axis=1)
        acc["oe_disturbance"] = 1 - acc.apply(lambda x: safe_div(x["tp"], x["tp"] + x["fn"]), axis=1)
        acc["oe_stable"] = 1 - acc.apply(lambda x: safe_div(x["tn"], x["tn"] + x["fp"]), axis=1)

        acc_overall_collector.append(acc)

    acc_overall = pd.concat(acc_overall_collector, ignore_index=True)

    acc_long = acc_overall.melt(
        id_vars=["country"],
        value_vars=["oo", "ce_disturbance", "ce_stable", "oe_disturbance", "oe_stable"],
        var_name="metric",
        value_name="value"
    )

    acc_long[["metric", "class"]] = acc_long["metric"].str.split("_", n=1, expand=True)
    acc_long["class"] = acc_long["class"].fillna("all")

    #country_accuracy = acc_long.groupby(["metric", "class", "country"], as_index=False).agg(mean=("value", "mean"), sd=("value", "std"))
    country_accuracy = acc_long.groupby(
            ["metric", "class", "country"],
            as_index=False
        ).agg(
            mean=("value", "mean"),
            sd=("value", "std"),
            ci_2_5=("value", lambda x: np.nanpercentile(x, 2.5)),
            ci_97_5=("value", lambda x: np.nanpercentile(x, 97.5))
        )

    country_accuracy["metric"] = country_accuracy["metric"].replace({"ce": "Commission", "oe": "Omission", "oo": "Overall"})
    country_accuracy["class"] = country_accuracy["class"].replace({"disturbance": "Disturbed pixels", "stable": "Stable pixels", "all": "All pixels"})

    return country_accuracy

''' run per country validation'''
country_accuracy_unet = compute_country_accuracy(reference_database, predictions_unet, weights, niter=niter, n_samples_country=n_samples_country)
country_accuracy_tempcnn = compute_country_accuracy(reference_database, predictions_tempcnn, weights, niter=niter, n_samples_country=n_samples_country)
country_accuracy_rfw5 = compute_country_accuracy(reference_database, predictions_rfw5, weights, niter=niter, n_samples_country=n_samples_country)
country_accuracy_efdav2 = compute_country_accuracy(reference_database, predictions_efdav2, weights, niter=niter, n_samples_country=n_samples_country)

''' combine them'''
country_accuracy = pd.concat([
    country_accuracy_unet.assign(model="1D U-Net W5"),
    country_accuracy_tempcnn.assign(model="TempCNN W5"),
    country_accuracy_rfw5.assign(model="Random Forest W5"),
    country_accuracy_efdav2.assign(model="EFDA v2.1"),
], ignore_index=True)

print("Validation finished")
print("***************************************************")

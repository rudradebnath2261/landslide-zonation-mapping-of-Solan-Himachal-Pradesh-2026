"""Landslide susceptibility mapping using Random Forest.

Adapted from the supplied Damodar Basin Random Forest flood mapping workflow.

Input: 10 conditioning-factor rasters and 1,625 labelled point samples.
Output: class-1 probability, binary prediction, five susceptibility classes,
        model, hold-out accuracy report, test predictions and importance table.

Required packages: numpy, pandas, geopandas, rasterio, scikit-learn, joblib.
"""

import os
from contextlib import ExitStack

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. INPUT DATA AND OUTPUT FOLDER

# Edit this only if you move the project to another folder.
folder = os.environ.get(
    "LANDSLIDE_FOLDER",
    r"C:\ADAMAS\sem_3_pra\GEOAI\land slide\MlDL_Spyder_Landlside\MlDL_Spyder",
)

points_file = os.path.join(folder, "Final_Sample_Points.shp")
output_folder = os.path.join(folder, "output")
os.makedirs(output_folder, exist_ok=True)

raster_files = {
    "ARF": os.path.join(folder, "ARF.tif"),
    "Aspect": os.path.join(folder, "Aspect.tif"),
    "Curvature": os.path.join(folder, "Curvature.tif"),
    "Elevation": os.path.join(folder, "Elevation.tif"),
    "Landcover": os.path.join(folder, "Landcover.tif"),
    "NDMI": os.path.join(folder, "NDMI.tif"),
    "NDVI": os.path.join(folder, "NDVI.tif"),
    "Slope": os.path.join(folder, "Slope.tif"),
    "TPI": os.path.join(folder, "TPI.tif"),
    "TRI": os.path.join(folder, "TRI.tif"),
}
predictors = list(raster_files)

THRESHOLD = 0.50
TILE_SIZE = 512
RANDOM_STATE = 42
NODATA_FLOAT = -9999.0
NODATA_BYTE = 255



# 2. CHECK FILES AND THE LANDSLIDE LABEL

for name, path in [("Sample points", points_file), *raster_files.items()]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input not found ({name}): {path}")

points = gpd.read_file(points_file)
if points.empty:
    raise ValueError("The sample-point shapefile is empty.")
if points.crs is None:
    raise ValueError("The sample-point shapefile has no defined CRS.")
if points.geometry.isna().any() or points.geometry.is_empty.any():
    raise ValueError("Some points have empty geometry. Fix them in GIS first.")
if not points.geometry.geom_type.eq("Point").all():
    raise ValueError("Final_Sample_Points.shp must contain Point geometries only.")

# In the supplied attribute table, 'Landslide' is the target label.
# Expected classes: 0 = no landslide; 1 = landslide.
matching_fields = [field for field in points.columns
                   if field.lower().replace("_", "") == "landslide"]
if len(matching_fields) != 1:
    raise ValueError(
        "Could not uniquely find the 'Landslide' field. "
        f"Fields available: {list(points.columns)}"
    )
target_field = matching_fields[0]

print("Total sample points:", len(points))
if len(points) != 1625:
    print("Note: Expected 1,625 points, but found", len(points))
print("Target field:", target_field)



# 3. MATCH ALL RASTERS TO THE ELEVATION GRID

# Continuous rasters: bilinear interpolation.
# Landcover: nearest neighbour so land-cover codes remain unchanged.
# Aspect: nearest neighbour to avoid mixing directions near 0/360 degrees.
with ExitStack() as stack:
    elevation = stack.enter_context(rasterio.open(raster_files["Elevation"]))
    if elevation.crs is None:
        raise ValueError("Elevation.tif has no CRS.")

    reference_crs = elevation.crs
    reference_transform = elevation.transform
    height = elevation.height
    width = elevation.width
    profile = elevation.profile.copy()

    aligned_rasters = {}
    for name, path in raster_files.items():
        src = stack.enter_context(rasterio.open(path))
        if src.crs is None:
            raise ValueError(f"{name} raster has no CRS: {path}")

        resampling_method = (
            Resampling.nearest if name in ("Landcover", "Aspect")
            else Resampling.bilinear
        )
        vrt = stack.enter_context(WarpedVRT(
            src,
            crs=reference_crs,
            transform=reference_transform,
            width=width,
            height=height,
            resampling=resampling_method,
            dtype="float32",
            nodata=NODATA_FLOAT,
        ))
        aligned_rasters[name] = vrt


    # 4. EXTRACT RASTER VALUES AT ALL SAMPLE POINTS

    aligned_points = points.to_crs(reference_crs)
    coordinates = [(geom.x, geom.y) for geom in aligned_points.geometry]

    sample_values = {}
    for name, vrt in aligned_rasters.items():
        sampled = vrt.sample(coordinates, masked=True)
        sample_values[name] = np.array([
            np.nan if np.ma.is_masked(value[0]) else float(value[0])
            for value in sampled
        ], dtype="float32")

    data = pd.DataFrame(sample_values)
    data["Landslide_Label"] = pd.to_numeric(
        aligned_points[target_field], errors="coerce"
    ).to_numpy()

    data = data.replace([np.inf, -np.inf, NODATA_FLOAT], np.nan).dropna()
    data = data[data["Landslide_Label"].isin([0, 1])].copy()
    data["Landslide_Label"] = data["Landslide_Label"].astype(int)

    print("Usable sample points:", len(data))
    print("Dropped/invalid points:", len(points) - len(data))
    print("Label counts (0 = no landslide, 1 = landslide):")
    print(data["Landslide_Label"].value_counts().sort_index())

    if data["Landslide_Label"].nunique() != 2:
        raise ValueError(
            "Random Forest classification requires BOTH 0 (no-landslide) "
            "and 1 (landslide) samples. The attribute-table screenshot only "
            "shows 1s; verify that the complete dataset contains 0s."
        )
    if data["Landslide_Label"].value_counts().min() < 2:
        raise ValueError("At least two valid points are required for each class.")

    data.to_csv(
        os.path.join(output_folder, "landslide_training_samples.csv"),
        index=False,
    )
    X = data[predictors]
    y = data["Landslide_Label"]


    # 5. SPLIT DATA: 80% TRAINING / 20% TESTING

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print("Training points:", len(X_train))
    print("Testing points:", len(X_test))


    # 6. TRAIN RANDOM FOREST (SAME SETTINGS AS FLOOD CODE)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    model_path = os.path.join(output_folder, "landslide_random_forest_model.joblib")
    joblib.dump(
        {"model": model, "predictors": predictors, "threshold": THRESHOLD},
        model_path,
    )


    # 7. ACCURACY ASSESSMENT ON UNSEEN TEST POINTS

    class_one_index = list(model.classes_).index(1)
    predicted_probabilities = model.predict_proba(X_test)[:, class_one_index]
    predicted_labels = (predicted_probabilities >= THRESHOLD).astype(int)

    accuracy = accuracy_score(y_test, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(y_test, predicted_labels)
    precision = precision_score(y_test, predicted_labels, zero_division=0)
    recall = recall_score(y_test, predicted_labels, zero_division=0)
    f1 = f1_score(y_test, predicted_labels, zero_division=0)
    kappa = cohen_kappa_score(y_test, predicted_labels)
    cm = confusion_matrix(y_test, predicted_labels, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    auc = (roc_auc_score(y_test, predicted_probabilities)
           if y_test.nunique() == 2 else float("nan"))
    class_report = classification_report(
        y_test, predicted_labels, labels=[0, 1],
        target_names=["No landslide (0)", "Landslide (1)"],
        zero_division=0,
    )

    print("\n========== RANDOM FOREST ACCURACY ==========")
    print(f"Test accuracy:     {accuracy * 100:.2f}%")
    print(f"Balanced accuracy: {balanced_accuracy * 100:.2f}%")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1 score:          {f1:.4f}")
    print(f"Specificity:       {specificity:.4f}")
    print(f"Cohen's Kappa:     {kappa:.4f}")
    print(f"ROC-AUC:           {auc:.4f}")
    print("\nConfusion matrix (actual rows, predicted columns):\n", cm)
    print("\nClassification report:\n", class_report)

    report_path = os.path.join(output_folder, "landslide_model_accuracy_report.txt")
    with open(report_path, "w", encoding="utf-8") as report:
        report.write("LANDSLIDE SUSCEPTIBILITY - RANDOM FOREST MODEL\n")
        report.write("===========================================\n\n")
        report.write(f"Sample shapefile: {points_file}\n")
        report.write(f"Target field: {target_field}\n")
        report.write("Class codes: 0 = no landslide, 1 = landslide\n")
        report.write(f"Total input points: {len(points)}\n")
        report.write(f"Usable points: {len(data)}\n")
        report.write(f"Rejected points: {len(points) - len(data)}\n")
        report.write(f"Training points: {len(X_train)}\n")
        report.write(f"Testing points: {len(X_test)}\n")
        report.write("Data split: 80% training / 20% testing (stratified)\n")
        report.write("Random Forest trees: 200\n")
        report.write("Random state: 42\n")
        report.write("Class weighting: balanced\n")
        report.write(f"Classification threshold: {THRESHOLD}\n\n")
        report.write(f"Accuracy: {accuracy:.6f} ({accuracy * 100:.2f}%)\n")
        report.write(f"Balanced accuracy: {balanced_accuracy:.6f}\n")
        report.write(f"Precision (class 1): {precision:.6f}\n")
        report.write(f"Recall / sensitivity (class 1): {recall:.6f}\n")
        report.write(f"F1 score (class 1): {f1:.6f}\n")
        report.write(f"Specificity (class 0): {specificity:.6f}\n")
        report.write(f"Cohen's Kappa: {kappa:.6f}\n")
        report.write(f"ROC-AUC: {auc:.6f}\n\n")
        report.write("Confusion matrix (actual rows; predicted columns):\n")
        report.write("                  Predicted 0    Predicted 1\n")
        report.write(f"Actual 0 (non-LS)  {tn:11d}    {fp:11d}\n")
        report.write(f"Actual 1 (LS)      {fn:11d}    {tp:11d}\n\n")
        report.write("Classification report:\n")
        report.write(class_report + "\n")
        report.write("NOTE: Accuracy is based on held-out test points from the\n")
        report.write("same point shapefile, not an independent validation set.\n")
        report.write("Random spatial splits may overstate accuracy where\n")
        report.write("training and testing sites are geographically clustered.\n")

    pd.DataFrame(
        cm,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"],
    ).to_csv(os.path.join(output_folder, "landslide_confusion_matrix.csv"))

    pd.DataFrame({
        "Predictor": predictors,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).to_csv(
        os.path.join(output_folder, "landslide_feature_importance.csv"), index=False
    )

    # Retains the input shapefile's zero-based record positions.
    pd.DataFrame({
        "Input_Row": X_test.index.to_numpy(),
        "Actual_Landslide": y_test.to_numpy(),
        "Predicted_Landslide": predicted_labels,
        "Landslide_Probability": predicted_probabilities,
    }).sort_values("Input_Row").to_csv(
        os.path.join(output_folder, "landslide_test_predictions.csv"), index=False
    )


    # 8. PREDICT LANDSLIDE SUSCEPTIBILITY FOR EVERY PIXEL

    # Process the reference extent tile by tile to conserve memory.
    base_profile = profile.copy()
    base_profile.update(driver="GTiff", count=1, compress="lzw")
    base_profile.pop("photometric", None)

    float_profile = base_profile.copy()
    float_profile.update(dtype="float32", nodata=NODATA_FLOAT)
    byte_profile = base_profile.copy()
    byte_profile.update(dtype="uint8", nodata=NODATA_BYTE)

    probability_path = os.path.join(output_folder, "landslide_probability.tif")
    binary_path = os.path.join(output_folder, "landslide_binary.tif")
    zones_path = os.path.join(output_folder, "landslide_susceptibility_5_classes.tif")

    with rasterio.open(probability_path, "w", **float_profile) as prob_dst, \
         rasterio.open(binary_path, "w", **byte_profile) as binary_dst, \
         rasterio.open(zones_path, "w", **byte_profile) as zones_dst:

        for top in range(0, height, TILE_SIZE):
            for left in range(0, width, TILE_SIZE):
                window = Window(
                    left, top,
                    min(TILE_SIZE, width - left),
                    min(TILE_SIZE, height - top),
                )
                raster_arrays = [
                    aligned_rasters[name].read(1, window=window, masked=True)
                    for name in predictors
                ]
                pixel_stack = np.ma.stack(raster_arrays, axis=-1)
                masked_pixels = np.ma.getmaskarray(pixel_stack).any(axis=-1)
                pixel_values = pixel_stack.filled(np.nan)
                valid = (
                    ~masked_pixels
                    & np.all(np.isfinite(pixel_values), axis=-1)
                    & np.all(pixel_values != NODATA_FLOAT, axis=-1)
                )

                probability_tile = np.full(valid.shape, NODATA_FLOAT, dtype="float32")
                binary_tile = np.full(valid.shape, NODATA_BYTE, dtype="uint8")
                zones_tile = np.full(valid.shape, NODATA_BYTE, dtype="uint8")

                if np.any(valid):
                    pixel_df = pd.DataFrame(pixel_values[valid], columns=predictors)
                    p = model.predict_proba(pixel_df)[:, class_one_index]
                    probability_tile[valid] = p.astype("float32")
                    binary_tile[valid] = (p >= THRESHOLD).astype("uint8")
                    zones_tile[valid] = (np.searchsorted(
                        [0.20, 0.40, 0.60, 0.80], p, side="right"
                    ) + 1).astype("uint8")

                prob_dst.write(probability_tile, 1, window=window)
                binary_dst.write(binary_tile, 1, window=window)
                zones_dst.write(zones_tile, 1, window=window)

print("\nLANDSLIDE MAPPING COMPLETED")
print("Output folder:", output_folder)
print("Model:", model_path)
print("Accuracy report:", report_path)
print("Probability map:", probability_path)
print("Binary map:", binary_path)
print("Five-class susceptibility map:", zones_path)
print("Five classes: 1 Very Low, 2 Low, 3 Moderate, 4 High, 5 Very High")

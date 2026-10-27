

# Landslide Susceptibility Zonation Mapping of Solan, Himachal Pradesh, India Using Random Forest (2026)

## 1. Introduction

Landslides are among the major natural hazards affecting mountainous regions, particularly in the Himalayan region of India. They occur when rock, soil, or debris moves down a slope under the influence of gravity. Factors such as steep slopes, heavy rainfall, terrain characteristics, vegetation conditions, and human activities can contribute to slope instability.

Solan district in Himachal Pradesh is characterized by mountainous terrain, complex topography, and varied environmental conditions. These characteristics make landslide susceptibility assessment important for infrastructure development, environmental management, and disaster preparedness.

This project focuses on developing a **GIS-based landslide susceptibility zonation map of Solan district using the Random Forest machine learning algorithm in Python**. Ten environmental and topographic conditioning factors were integrated with landslide and non-landslide sample locations to identify areas with different levels of susceptibility.

The resulting map classifies the study area into five susceptibility categories: Very Low, Low, Moderate, High, and Very High.

## 2. Objectives

- To develop a landslide susceptibility model using GIS, remote sensing, and Random Forest machine learning.
- To analyze ten environmental and topographic conditioning factors associated with landslide occurrence.
- To generate a spatial landslide susceptibility probability map and classify the study area into five susceptibility zones.
- To evaluate model performance using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.
- To support landslide susceptibility assessment and preliminary disaster management planning.

## 3. Landslide Conditioning Factors

Ten conditioning factors were selected to represent the topographic, environmental, vegetation, and rainfall-related characteristics of the study area.

| Factor | Description and importance |
|---|---|
| **1. Elevation** | Represents the height of the terrain above sea level. Elevation is associated with variations in landforms, climatic conditions, vegetation, and slope processes. |
| **2. Slope** | Measures the steepness of the terrain. Steeper slopes generally experience greater gravitational forces that may contribute to slope instability. |
| **3. Aspect** | Represents the direction a slope faces. It influences sunlight exposure, vegetation distribution, moisture availability, and weathering conditions. |
| **4. Curvature** | Describes the shape of the terrain, including concave and convex surfaces. It influences runoff concentration, erosion, and the movement or accumulation of materials. |
| **5. Topographic Position Index (TPI)** | Compares the elevation of a location with its surrounding terrain to distinguish ridges, valleys, and slopes. |
| **6. Terrain Ruggedness Index (TRI)** | Measures variations in elevation between neighboring locations. It helps characterize terrain roughness and topographic complexity. |
| **7. Annual Rainfall (ARF)** | Represents rainfall-related conditions. Rainfall can increase soil moisture, pore-water pressure, and erosion, potentially triggering slope failure. The exact definition and time period of the input ARF raster require verification. |
| **8. NDVI** | The Normalized Difference Vegetation Index represents vegetation greenness. Vegetation can contribute to soil stabilization, while sparse vegetation may be associated with greater surface exposure. |
| **9. NDMI** | The Normalized Difference Moisture Index provides information about vegetation moisture conditions. It serves as an indirect indicator of moisture-related environmental characteristics. |
| **10. Land Cover** | Represents different surface cover classes, which influence runoff, erosion, vegetation protection, and human modification of slopes. |

### Interpretation of the conditioning factors

These factors represent different processes related to landslide occurrence.

**Topographic factors:** Elevation, slope, aspect, curvature, TPI, and TRI describe terrain conditions. Together, they help the model identify relationships between landslide occurrences and the physical characteristics of the landscape.

**Environmental factors:** NDVI, NDMI, and land cover provide information about vegetation, moisture conditions, and surface characteristics that may influence slope stability.

**Rainfall factor:** The ARF raster represents a rainfall-related influence on slope instability.

All ten factors were considered simultaneously by the Random Forest model to identify nonlinear relationships between environmental conditions and observed landslide occurrences.

The actual statistical ranking of these factors requires the exported `landslide_feature_importance.csv` file. Their physical importance should not be confused with their measured contribution to this particular model.

## 4. Methodology

The study followed a GIS-based machine learning approach involving spatial data preparation, raster processing, sample extraction, model training, validation, and susceptibility mapping.

### Step 1: Data preparation

Ten raster layers representing the selected landslide conditioning factors were prepared for the study area.

A point shapefile containing 1,625 labelled landslide and non-landslide sample locations was used for model development.

The target variable was classified into two categories:

- **1 = Landslide**
- **0 = Non-landslide**

After excluding one invalid sample, 1,624 valid points were used for model training and evaluation.

### Step 2: Raster preprocessing

The input raster datasets were aligned using the elevation raster as the reference grid.

All rasters were matched to a common coordinate reference system, spatial resolution, extent, and pixel alignment.

Bilinear interpolation was used for continuous raster variables, while nearest-neighbour resampling was applied to the land-cover and aspect rasters.

This process ensured that the conditioning factors could be compared and analyzed on a consistent spatial grid.

### Step 3: Extraction of sample values

Using Python libraries such as Rasterio and GeoPandas, the values of all ten conditioning factors were extracted at each landslide and non-landslide sample location.

The extracted values were combined into a structured dataset containing the predictor variables and binary landslide labels.

Invalid or missing records were removed before model development.

### Step 4: Training and testing dataset

The 1,624 valid samples were divided using a stratified random split.

| Dataset | Samples |
|---|---:|
| Total sample points | 1,625 |
| Valid sample points | 1,624 |
| Training dataset (80%) | 1,299 |
| Testing dataset (20%) | 325 |

Stratified sampling helped maintain a consistent proportion of landslide and non-landslide observations across training and testing datasets.

### Step 5: Random Forest model development

The Random Forest algorithm was implemented using the Scikit-learn library in Python.

Random Forest is an ensemble machine learning method that combines multiple decision trees to improve classification performance. Each tree learns relationships between the conditioning factors and landslide occurrence, while the final model combines their predictions.

The model was developed using the following parameters:

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Number of trees | 200 |
| Training–testing ratio | 80:20 |
| Random state | 42 |
| Class weight | Balanced |
| Classification threshold | 0.50 |

The trained model was used to predict landslide susceptibility across the valid raster-covered area.

### Step 6: Model evaluation

The performance of the model was evaluated using 325 held-out testing samples.

The assessment included overall accuracy, balanced accuracy, precision, recall, F1-score, specificity, Cohen's Kappa, ROC-AUC, and confusion matrix.

These metrics were used to evaluate the model's ability to distinguish between landslide and non-landslide locations.

### Step 7: Landslide susceptibility zonation

The trained Random Forest model generated a continuous susceptibility score ranging from 0 to 1.

These scores were classified into five equal-interval susceptibility zones.

| Class | Probability range | Susceptibility |
|---|---|---|
| 1 | 0.00–0.20 | Very Low |
| 2 | 0.20–0.40 | Low |
| 3 | 0.40–0.60 | Moderate |
| 4 | 0.60–0.80 | High |
| 5 | 0.80–1.00 | Very High |

The classification helps represent variations in predicted landslide susceptibility across the study area.

## 5. Results and Accuracy Assessment

The Random Forest model demonstrated good discrimination between landslide and non-landslide sample locations.

### Model performance

| Performance metric | Result |
|---|---:|
| **Overall Accuracy** | **75.69%** |
| **ROC-AUC** | **0.8446** |
| Balanced Accuracy | 75.70% |
| Precision | 76.54% |
| Recall | 75.15% |
| F1-Score | 0.7584 |
| Specificity | 76.25% |
| Cohen's Kappa | 0.5139 |

### Interpretation

**Overall accuracy (75.69%):** The model correctly classified approximately 75.69% of the testing samples, indicating reasonable predictive performance.

**ROC-AUC (0.8446):** The ROC-AUC value indicates good discrimination between landslide and non-landslide locations across classification thresholds.

**Precision (76.54%):** Approximately 76.54% of the locations predicted as landslides were actual landslide samples.

**Recall (75.15%):** The model successfully identified approximately 75.15% of actual landslide samples.

**F1-score (0.7584):** Indicates a reasonable balance between precision and recall.

**Cohen's Kappa (0.5139):** Indicates moderate agreement between actual and predicted classifications beyond chance agreement.

### Confusion matrix

| Actual class | Predicted Non-landslide | Predicted Landslide |
|---|---:|---:|
| Non-landslide | 122 | 38 |
| Landslide | 41 | 124 |

Out of 325 testing samples:

- 246 samples were classified correctly.
- 79 samples were misclassified.
- 41 actual landslide samples were incorrectly predicted as non-landslide.
- 38 non-landslide samples were incorrectly predicted as landslide.

The results indicate that the model can identify meaningful spatial relationships between landslide occurrences and conditioning factors. However, some landslide locations remain undetected, emphasizing the importance of further validation.

## 6. Landslide Susceptibility Zonation Results

The final output consists of a continuous landslide susceptibility raster, a binary prediction raster, and a five-class susceptibility zonation map.

The five-class map identifies areas with different predicted levels of landslide susceptibility.

Areas classified under High and Very High susceptibility represent locations where the model estimates relatively greater landslide susceptibility based on the selected conditioning factors.

Moderate susceptibility zones represent intermediate conditions, while Low and Very Low zones represent comparatively lower estimated susceptibility.

The map can support preliminary identification of areas requiring detailed geological investigation, slope monitoring, infrastructure assessment, and disaster management planning.

The exact geographical distribution and area percentages of the susceptibility classes must be calculated from the final raster before making location-specific conclusions.

## 7. Limitations and Future Improvements

Although the model produced encouraging results, several limitations should be considered.

The evaluation used a random training–testing split rather than spatially independent validation. Consequently, geographic similarity between nearby samples may lead to optimistic performance estimates.

The model also does not explicitly incorporate some potentially important landslide-related variables, such as lithology, soil properties, distance to roads, drainage proximity, or faults.

Future improvements can include spatial cross-validation, independent landslide inventory testing, feature-importance analysis, additional environmental variables, and field verification.

The susceptibility map indicates relative spatial patterns and should not be interpreted as a prediction of the timing of landslides or as a complete landslide risk assessment.

## 8. Conclusion

This study demonstrates the application of GIS, remote sensing, and Random Forest machine learning for landslide susceptibility zonation mapping in Solan district, Himachal Pradesh.

The integration of ten environmental and topographic conditioning factors with 1,624 valid sample points enabled the development of a spatial prediction model. The Random Forest classifier achieved **75.69% overall accuracy, an F1-score of 0.7584, and a ROC-AUC of 0.8446**, indicating good discrimination between landslide and non-landslide samples.

The resulting five-class susceptibility map provides a data-driven approach for identifying potentially landslide-prone areas. It can support preliminary disaster preparedness, land-use planning, and infrastructure assessment, subject to further spatial validation and field investigation.

The project also demonstrates the potential of Python-based geospatial machine learning workflows for environmental hazard assessment in mountainous regions.

## 9. References

1. [District Administration, Solan — Disaster Management](https://hpsolan.nic.in/disaster-management/)
2. [Geological Survey of India — National Landslide Susceptibility Mapping](https://gsi.gov.in/landslide-hazard/)
3. [USGS — Landslide Causes and Processes](https://www.usgs.gov/faqs/what-a-landslide-and-what-causes-one)
4. [Magray et al. (2023) — Landslide Susceptibility Zonation in Parts of Solan, Himachal Pradesh](https://doi.org/10.14746/quageo-2023-0020)
5. [Scikit-learn — Random Forest Feature Importance](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)

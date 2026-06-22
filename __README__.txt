🧠 How the AI System Works

The Brain Tumor Detection System follows a multi-stage AI pipeline designed to analyze multi-modal MRI scans and provide comprehensive tumour assessment.

1️⃣ MRI Acquisition

The process begins with a patients brain MRI scan.

The system supports multi-modal MRI volumes including:

T1
T1ce
T2
FLAIR

These scans are typically stored as 3D volumetric medical images where each volume 
contains hundreds of image slices representing different cross-sections of the brain.

2️⃣ MRI Preprocessing

Before analysis, the MRI volume undergoes several preprocessing stages:

Skull stripping
Brain tissue isolation
Intensity normalization
Noise reduction
Slice selection
Image resizing

This ensures that only relevant brain structures are provided to the neural networks.

3️⃣ 3D Volume → 2D Slice Conversion

Instead of processing the entire MRI volume at once, the system converts the 3D scan into multiple 2D slices.

Why?

Reduces computational requirements
Enables faster training on limited GPU resources
Allows detailed slice-by-slice tumour analysis
Improves training efficiency on Google Colab environments

Each slice is extracted while preserving anatomical information from the original MRI volume.

4️⃣ Tumour Segmentation

Each MRI slice is passed into the Deep UNet 3+ segmentation network.

The segmentation model learns to identify and isolate tumour regions from healthy brain tissue.

The model detects:

Necrotic Core (NCR)
Edema (ED)
Enhancing Tumour (ET)

Advanced architectural components include:

Full-scale skip connections
Multi-scale feature fusion
Deep supervision
Classification Guided Module (CGM)

The output is a detailed tumour mask showing the exact tumour location and boundaries.

5️⃣ Tumour Localization

Once segmentation is completed:

Tumour boundaries are extracted
Tumour area is calculated
Tumour regions are isolated
Region-specific analysis begins

This stage provides spatial understanding of tumour location within the brain.

6️⃣ Tumour Classification

The segmented tumour region is analyzed using a dedicated classification network.

The classifier predicts:

Glioma
Meningioma
Pituitary Tumour
No Tumour

The model evaluates tumour texture, shape, intensity patterns, and structural characteristics to determine the most probable tumour category.

7️⃣ Radiomics Feature Extraction

After classification, advanced radiomics features are extracted.

These include:

Shape Features
Area
Perimeter
Compactness
Eccentricity
Intensity Features
Mean intensity
Variance
Skewness
Kurtosis
Texture Features
GLCM statistics
Contrast
Homogeneity
Entropy
Correlation

These quantitative biomarkers provide additional information beyond what the neural networks learn automatically.

8️⃣ Explainable AI (Grad-CAM)

To improve transparency, Grad-CAM visualization is generated.

This creates a heatmap highlighting:

Important tumour regions
Areas influencing model decisions
Feature activation locations

This allows visual verification of the models reasoning process.

9️⃣ Growth Prediction

Using segmentation outputs and extracted tumour characteristics, the growth prediction model estimates tumour progression patterns and future development trends.

This provides additional clinical insights for research and experimentation.

🔟 Survival Prediction

The survival prediction module analyzes:

Tumour characteristics
Segmentation features
Radiomics features
Morphological measurements

to estimate survival-related outcomes and risk indicators.

1️⃣1️⃣ Diagnostic Output

Finally, the system generates:

Tumour segmentation maps
Classification results
Grad-CAM visualizations
Radiomics measurements
Growth analysis
Survival predictions
Diagnostic reports


⚠️ the survivaland growth prediction was made baased on synthetic data and no real data was used it was used as a concept or hyptheically we can do it.

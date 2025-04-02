# MM5 Multimodal Dataset

## Overview
The MM5 dataset is a comprehensive multimodal dataset capturing RGB, Depth, Thermal (LWIR), Ultraviolet (UV), and Near-Infrared (NIR) images. It is designed for advanced multimodal research, providing diverse modalities, annotated data, and carefully calibrated and aligned images.

## Dataset Contents

The dataset consists of:
- RGB images captured under various lighting conditions.
- Raw and processed depth data (16-bit).
- Thermal images (16-bit raw, 24-bit static-colour encoded, and 8-bit dynamic range).
- Ultraviolet images.
- Near-infrared images (16-bit raw).
- Pixel-wise annotations for segmentation, instance segmentation, and classification both, for the aligned/cropped data as well as for the raw data.

## Dataset Structure

### Raw Data

The raw data is organised into separate folders corresponding to each camera modality. In our stereo setup, images from the left and right cameras are distinguished with suffixes `_0` and `_1`, respectively. Each file follows a standardised naming convention including:

```
[sequence_number]_[settings_ID]_[timestamp]_[modality].png
```

**Example:** `1_5_20240716_130310_143_rgb.png`

**Raw Data Folder Structure:**
- `DEPTH_0`, `DEPTH_1`
- `IR_0`, `IR_1`
- `LWIR`
- `META`
- `RGB_0`, `RGB_1`
- `UV`
- `ANNO_V`, `ANNO_T`, `ANNO_U`

### Data Types and Representations

- **Depth and IR:** Stored as 16-bit single-channel images. Available in two forms:
  - Raw (`_raw`)
  - Kinect SDK transformed (`_tr`)

- **Thermal (LWIR):** Available representations include:
  - Raw 16-bit images (`_lwir16`)
  - 24-bit fixed colour encoded images (`_lwir`)
  - 8-bit grayscale with automatic gain control (AGC) (`_lwir8dyn`)

Encoded LWIR images are provided for convenience and can be reproduced from raw data.

### Processed Data

Aligned and cropped data have sequential filenames (starting from 1) ensuring cross-modal consistency. All modalities corresponding to a single capture share identical filenames.

**Processed Data Folder Structure:**
- Annotations:
  - `ANNO_CLASS`, `ANNO_INST`
  - `ANNO_VIS_CLASS`, `ANNO_VIS_INST` (colour-coded for visualisation)

- Modalities:
  - Depth: `D`, `D_Focus`, `D_Focus960N`, `D16`
  - Infrared: `I`, `I16`
  - Metadata: `META`
  - RGB (lighting settings): `RGB1` – `RGB8`
  - Thermal: `T8`, `T16`, `T24`
  - Ultraviolet: `U1`, `U8`, `U9`

RGB and UV folder names correspond to their illumination settings, while Depth, IR, and Thermal folders indicate data encoding type.

## Directory Structure

### Raw Data
```
MM5_RAW/
├── DEPTH_0/
├── DEPTH_1/
├── IR_0/
├── IR_1/
├── LWIR/
├── META/
├── RGB_0/
├── RGB_1/
└── UV/
```

### Processed and Aligned Data
```
MM5_ALIGNED/
├── ANNO_CLASS/
├── ANNO_INST/
├── ANNO_VIS_CLASS/
├── ANNO_VIS_INST/
├── D/
├── D_Focus/
├── D_Focus960N/
├── D_16
├── I
├── I16
├── META
├── RGB1/ ... RGB8/
├── T8/
├── T16/
├── T24/
├── U1/
├── U8/
└── U9/
```

## Annotations
Annotations include class labels and instance(object) labels in pixel-level formats, available both visually (for quick reference) and as raw data for model training.
Next to the labels for the aligned data, also the labels for the raw thermal and UV data are included.


## Data Usage and Recommendations
To avoid redundant duplication, modalities such as depth, thermal, and IR, which do not vary across different RGB lighting conditions, are stored once. Researchers wishing to train across multiple or all lighting conditions must explicitly pair shared modalities with each RGB dataset. This can be efficiently managed by custom data loader scripts or restructuring the dataset to meet specific research requirements.

## Download Links
- [Download Link for Raw Data](https://[placeholder-for-raw-data-download])
- [Download Link for Processed Data](https://[placeholder-for-processed-data-download])
- [Download Link for Annotations](https://[placeholder-for-annotations-download])

## File Naming Conventions
Raw data files follow this pattern:
```
[SequenceNumber]_[SettingID]_[Timestamp]_[Modality].png
```
Example:
```
1_5_20240716_130310_143_rgb.png
```

Processed data files follow a sequential naming convention:
```
[FrameNumber]_[Modality].png
```

## Usage and Citation
If you use this dataset, please cite our publication:
```
[Insert your citation details here]
```

## License
This dataset is available under [Insert License Type, e.g., Creative Commons Attribution-NonCommercial 4.0 International License].

## Contact
For questions, suggestions, or feedback, please contact:
- [Your Name and/or Team Name]
- Email: [your-email@example.com]

## Updates
This dataset is regularly updated with additional scenes, modalities, and annotations. Please check back periodically for updates.

## Acknowledgments
[Include acknowledgments to funding agencies, supporting institutions, or contributors here.]


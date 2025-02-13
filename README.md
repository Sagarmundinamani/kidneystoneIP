# Kidney Stone Image Processing

## Overview
This project applies various image processing techniques using OpenCV to analyze kidney stone images. It includes Gabor filtering, histogram equalization, Laplacian sharpening, and the Watershed algorithm for segmentation.

## Requirements
Make sure you have the following dependencies installed:

```sh
pip install opencv-python numpy matplotlib
```

## Features
- **Gabor Filtering**: Enhances texture features.
- **Histogram Equalization**: Improves image contrast.
- **Laplacian Sharpening**: Enhances edges in the image.
- **Watershed Algorithm**: Performs image segmentation.

## Usage
### Running the Script
1. Place kidney stone images in the `images` directory.
2. Modify the `image_no` variable in `Finalcode.py` to match your image file.
3. Run the script:

```sh
python Finalcode.py
```

### Output
- The processed image will be displayed using Matplotlib.
- Segmented regions will be marked with red boundaries.

## File Structure
```
kidneystoneIP/
│── Finalcode.py         # Main script
│── images/              # Directory for input images
│── README.md            # Project documentation
```

## Troubleshooting
- **ModuleNotFoundError**: Install missing dependencies with `pip install opencv-python numpy matplotlib`.
- **Image Not Found Error**: Ensure the image path in `Finalcode.py` is correct.

## License
This project is licensed under the MIT License.


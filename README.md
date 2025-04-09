# SIM
Creates artificial SIM images for accurate filter performance evaluation

## Methodology
- Apply sinusoidal illumination with 3 Angle rotations and 3 Phases to synthetic image generating 9 images
- Add noise
- Apply OTF
- Apply Filters
- Reconstruct SIM images
- Compare Reconstructed filtered SIM images with Reconstructed noise free image

## Codes
### noise
Module with functions for adding different types of noise. Mainly,
- Gaussian
- Poisson
- Salt and Pepper
- Mix noise (combination of above three)

### optics
Module with different functions for
- Adding structured illumination
- Normalizing a image
- A square windowed lowpass filter
- A circular lowpass filter
- Incoherent optical transfer function
- Displaying an image in both space and frequency domain

### denoising
Creates raw data for SIM by applying illumination, adding noise, then filtering

### denoising_parts
Breakes image into multiple segments then performs filtering on each segment allowing tuning of filter parameters then stitches it back to give final image

### filter_evaluation
Compares the reconstructed SIM images of different filters and creates plots for easy visualization

### frc_visualization
Performs Fourier Rank Correlation for two images

### color_bar
Range and Histogram visualization of reconstructed images

## Data

### source_images
Contains images used as a source, eg. ground truth

### output images
Contains results of filtering with different noise types and intensities and also contains plots
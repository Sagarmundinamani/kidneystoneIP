import cv2
import numpy as np
import matplotlib.pyplot as plt

def build_filters():
    # Returns a list of Gabor kernels in several orientations
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 32):
        params = {'ksize': (ksize, ksize), 'sigma': 0.0225, 'theta': theta, 'lambd': 15.0,
                  'gamma': 0.01, 'psi': 0, 'ktype': cv2.CV_32F}
        
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5 * kern.sum()
        filters.append((kern, params))
    return filters

def process(img, filters):
    # Applies Gabor filter on the image using the provided filters
    accum = np.zeros_like(img)
    for kern, _ in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def Histeq(img):
    # Performs histogram equalization on the image
    equ = cv2.equalizeHist(img)
    return equ

def GaborFilter(img):
    # Applies Gabor filter on the image
    filters = build_filters()
    p = process(img, filters)
    return p

def Laplacian(img, par):
    # Applies Laplacian sharpening on the image
    lap = cv2.Laplacian(img, cv2.CV_64F)
    sharp = img - par * lap
    sharp = np.uint8(cv2.normalize(sharp, None, 0, 255, cv2.NORM_MINMAX))
    return sharp

def Watershed(img):
    # Applies watershed algorithm on the image
    
    # Convert image to grayscale if needed
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Noise removal with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.23 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply watershed algorithm
    img2 = img.copy()
    img2 = cv2.medianBlur(img2, 5)
    markers = cv2.watershed(img2, markers)
    img2[markers == -1] = [255, 0, 0]

    return img2

if __name__ == '__main__':
    # Example usage
    s = r'C:\Users\mundi\Downloads\kidneystoneIP\kidneystoneIP\images'  # Directory path
    image_no = '\\image1.jpg'  # Image file name (use double backslashes)
    s = s + image_no  # Full image path (corrected construction)
    
    img = cv2.imread(s, 1)  # Load image
    
    if img is None:
        print(f"Error: Image not found at '{s}'")
    else:
        img3 = None  # Define img3 before usage
        
        # Process image based on conditions
        if image_no == '\\image1.jpg':
            img3 = Laplacian(img, 0.239)
        elif image_no == '\\image2.jpg':
            img3 = GaborFilter(img)
            img3 = Histeq(img3)
        elif image_no == '\\image4.jpg':
            img3 = GaborFilter(img)
        
        if img3 is not None:
            # Apply Watershed algorithm
            img3 = Watershed(img3)
            
            # Display the final marked image
            plt.imshow(img3, 'gray')
            plt.title('Marked')
            plt.xticks([]), plt.yticks([])
            plt.show()

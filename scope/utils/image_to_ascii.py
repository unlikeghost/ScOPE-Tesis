import numpy as np
from PIL import Image
from scipy import ndimage

# ASCII_CHARS = " "
# ASCII_CHARS = np.array(list("@%#$&*+=-:,.~_` "))
ASCII_CHARS = np.array(list("@%#*+=-:. "))
CHAR_STEP = 255 // len(ASCII_CHARS)


def sobel_edge_detection(image) -> np.ndarray:
    imx = np.zeros(image.shape)
    imy = np.zeros(image.shape)
    ndimage.sobel(image, 1, imx)
    ndimage.sobel(image, 0, imy)
    return np.sqrt(imx ** 2 + imy ** 2)


def image_to_ascii(image_path, width=70) -> str:
    # Load and resize in one step
    img = Image.open(image_path).convert('L').resize(
        (width, int(width * Image.open(image_path).height / Image.open(image_path).width * 0.5))
    )

    img_array = np.array(img)

    # Vectorized edge detection
    edges = sobel_edge_detection(img_array)
    edge_mask = (edges > edges.mean() * 2)

    # Vectorized ASCII conversion
    values = np.where(edge_mask, 255, img_array)
    char_indices = np.minimum(values // CHAR_STEP, len(ASCII_CHARS) - 1)

    # Build ASCII lines
    ascii_img = [
        ''.join(ASCII_CHARS[char_indices[y]])
        for y in range(img_array.shape[0])
    ]

    return '\n'.join(ascii_img)


if __name__ == "__main__":
    ascii_art = image_to_ascii(r"C:\Users\jesus\Images\Bg\images.jpg", width=90)
    print(ascii_art)

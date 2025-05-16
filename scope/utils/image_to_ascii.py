import numpy as np
from PIL import Image

# ASCII_CHARS = "@%#*+=-:. "
ASCII_CHARS: str = "#$%&*+,-./:;<=>?@[]^_`{|}~ "


def sobel_edge_detection(image) -> np.ndarray:
    from scipy import ndimage
    imx = np.zeros(image.shape)
    imy = np.zeros(image.shape)
    ndimage.sobel(image, 1, imx)
    ndimage.sobel(image, 0, imy)
    return np.sqrt(imx ** 2 + imy ** 2)


def image_to_ascii(image_path, width=70) -> str:
    img = Image.open(image_path).convert('L')
    img = img.resize((width, int(width * img.height / img.width * 0.5)))

    edges = sobel_edge_detection(np.array(img))
    edges = (edges > edges.mean() * 2) * 255

    ascii_img = []
    for y in range(img.height):
        line = ""
        for x in range(img.width):
            val = edges[y, x] if edges[y, x] > 0 else img.getpixel((x, y))
            line += ASCII_CHARS[min(val // (255 // len(ASCII_CHARS)), len(ASCII_CHARS) - 1)]
        ascii_img.append(line)

    return "\n".join(ascii_img)


if __name__ == "__main__":
    ascii_art = image_to_ascii(r"C:\Users\jesus\Images\Bg\images.jpg", width=90)
    print(ascii_art)

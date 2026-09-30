import cv2
import pytesseract
import numpy as np
from pytesseract import Output

def extract_text(image):
    # Convert image to OpenCV format
    img = np.array(image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Upscale image
    img = cv2.resize(
        img,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # Improve contrast without aggressive thresholding
    img = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(img)

    # Extract words and confidence scores
    data = pytesseract.image_to_data(
        img,
        config="--oem 3 --psm 11",
        output_type=Output.DICT
    )

    lines = {}

    for i, word in enumerate(data["text"]):
        word = word.strip()

        try:
            confidence = float(data["conf"][i])
        except ValueError:
            continue

        # Ignore empty and low-confidence results
        if word and confidence >= 40:
            key = (
                data["block_num"][i],
                data["par_num"][i],
                data["line_num"][i]
            )

            lines.setdefault(key, []).append(word)

    return "\n".join(
        " ".join(words)
        for words in lines.values()
    )
import cv2
import pytesseract
import numpy as np

# Convert the input image to grayscale
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Draw bounding boxes around recognized text
def draw_bounding_boxes(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n_boxes = len(data["text"])

    for i in range(n_boxes):
        if int(data["conf"][i]) == -1:
            continue
        x, y = data["left"][i], data["top"][i]
        w, h = data["width"][i], data["height"][i]

        top_left = (x, y)
        bottom_right = (x + w, y + h)
        green = (0, 255, 0)
        thickness = 1

        cv2.rectangle(image, top_left, bottom_right, green, thickness)

    return image

# Extract text and return both text and image with boxes
def image_to_text(image_path):
    image = cv2.imread(image_path)
    gray_image = convert_to_grayscale(image)
    boxed_image = draw_bounding_boxes(image.copy())
    text = pytesseract.image_to_string(gray_image)
    return text, boxed_image

# Main execution block
if __name__ == "__main__":
    image_path = 'nlp_project.png'  # Replace with your image path
    extracted_text, boxed_image = image_to_text(image_path)

    # Display the extracted text
    print("Extracted Text:")
    print(extracted_text)

    # Display the image with bounding boxes (if desired)
    cv2.imshow('Image with Bounding Boxes', boxed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

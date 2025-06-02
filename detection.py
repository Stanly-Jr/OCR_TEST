import cv2
import numpy as np
import torchvision.transforms as transforms
import torch
from PIL import Image



# Class labels
class_names = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


def predict_character(image, model, device):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    return class_names[pred.item()]


def extract_characters_with_cca(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    h_img, w_img = image.shape[:2]
    image_area = h_img * w_img

    stats = stats[1:]
    areas = stats[:, cv2.CC_STAT_AREA]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    widths = stats[:, cv2.CC_STAT_WIDTH]

    filtered_areas = areas[areas < 0.25 * image_area]
    if len(filtered_areas) < 2:
        filtered_areas = areas

    Q1, Q3 = np.percentile(filtered_areas, [25, 75])
    IQR = Q3 - Q1
    min_area = max(10, Q1 - 2.5 * IQR)
    max_area = Q3 + 2.0 * IQR
    median_height = np.median(heights)

    characters = []
    for i, (x, y, w, h, area) in enumerate(stats):
        aspect_ratio = w / float(h)
        keep = (
            min_area < area < max_area and
            (0.4 * median_height < h < 2.7 * median_height) and
            (0.15 < aspect_ratio < 2.0)
        )
        if keep:
            char_img = binary[y:y+h, x:x+w]
            density = np.sum(char_img == 0) / (w * h)
            if density < 0.15 or w < 4 or h < 8:
                continue
            char_img = cv2.copyMakeBorder(char_img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
            characters.append((x, y, w, h, char_img))

    if not characters:
        return []

    characters.sort(key=lambda c: c[1] + c[3] // 2)
    line_threshold = median_height * 0.75

    lines = []
    current_line = [characters[0]]
    for i in range(1, len(characters)):
        prev_cy = current_line[-1][1] + current_line[-1][3] // 2
        curr_cy = characters[i][1] + characters[i][3] // 2
        if abs(curr_cy - prev_cy) < line_threshold:
            current_line.append(characters[i])
        else:
            lines.append(current_line)
            current_line = [characters[i]]
    lines.append(current_line)

    lines = [sorted(line, key=lambda c: c[0]) for line in lines]
    return [char_img for line in lines for (_, _, _, _, char_img) in line]


def remove_india_code(chars):
    plate = ''.join(chars)
    if plate.startswith("IND"):
        return chars[3:]
    if plate.startswith("ID"):
        return chars[2:]
    if plate.startswith("ND"):
        return chars[2:]
    if plate.startswith("I") and len(chars) > 1:
        return chars[1:]
    return chars


def recognize_license_plate(image, model, device='cpu'):
    char_images = extract_characters_with_cca(image)
    if not char_images:
        return ""

    predictions = [predict_character(img, model, device) for img in char_images]
    filtered_preds = remove_india_code(predictions)
    return ''.join(filtered_preds)


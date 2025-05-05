import re
from typing import Dict, List, Optional
from PIL import Image, ImageDraw, ImageFont


def scale_polygon(polygon, w, h):
    new_polygon = []
    for (x, y) in polygon:
        new_polygon.append((x * w, y * h))
    return new_polygon

def draw_polygon(image: Image.Image, points: List[List[int]], label: Optional[str] = None):
    draw = ImageDraw.Draw(image)
    if len(points) > 2:
        draw.polygon(points, outline="red", width=3)
    elif len(points) == 2:
        draw.rectangle(points, outline="red", width=3)
    else:
        raise ValueError(f'points={points} only has one point!')
    
    if label is not None:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
        draw.text(points[0], label, font=font, fill=(0, 0, 255))
    return image

def visualize_image_bbox(data_dict, image_processing_config, processor):
    if image_processing_config.get('has_coordinates') != True:
        return 
    
    messages = data_dict['messages']

    polygons = []
    first_image_content = None

    for msg in messages:
        for content in msg['content']:
            if content['type'] == 'text':
                for match in re.finditer(r'\[(\d+(\.\d+)?,\s*)+\d+(\.\d+)?\]', content["text"]):
                    coordinate_matches = re.findall(r"([0-9.]+)", match.group(0))
                    coords = [float(coord) for coord in coordinate_matches]
                    polygons.append(list(zip(coords[::2], coords[1::2])))
            elif first_image_content is None and content['type'] == 'image':
                first_image_content = content

    first_image = first_image_content['image']
    first_image = processor.preprocess_image(first_image, image_processing_config)
    w, h = first_image.size

    if len(polygons) > 0:
        for i, polygon in enumerate(polygons):
            polygon = scale_polygon(polygon, w, h)
            first_image = draw_polygon(first_image, polygon, label=str(i))
    
    first_image_content['image'] = first_image
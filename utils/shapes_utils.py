import random
import numpy as np
from PIL import Image, ImageDraw


def generate_shape_motif(color):
    num_shapes = random.randint(1, 3)
    image = Image.new('RGBA', (300, 300), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for _ in range(num_shapes):
        shape_type = random.randint(0, 2)
        color = get_color(color)
        p_a = (random.randint(0, 199), random.randint(0, 199))
        p_b = (random.randint(p_a[0] + 50, 299), random.randint(p_a[1] + 50, 299))
        if shape_type == 0:
            draw.ellipse(p_a + p_b, fill=color)
        elif shape_type == 1:
            draw.rectangle(p_a + p_b, fill=color)
        else:
            p_c = (random.randint(0, 299), random.randint(0, 299))
            draw.polygon(p_a + p_b + p_c, fill=color)
    return image


def get_color(color):
    if color == 'gray':
        color = random.randint(0, 255)
        color = (color, color, color)
    elif not color:
        color =  (255, 255, 255)
    elif type(color) is not tuple:
       color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color


def generate_line_motif(color, image_size):
    color = get_color(color)
    line_length = image_size / 8 + random.random() * image_size / 2
    theta = random.random() * 2 * np.pi
    x_0, y_0 = 0, 0
    x_1, y_1 = int(np.cos(theta) * line_length), int(np.sin(theta) * line_length)
    if x_1 < 0:
        x_0 = -x_1
        x_1 = 0
    if y_1 < 0:
        y_0 = -y_1
        y_1 = 0
    image = Image.new('RGBA', (max(2, abs(x_0-x_1)), max(2, abs(y_0-y_1))), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    line_width = 1
    draw.line((x_0, y_0, x_1, y_1), fill=color, width=line_width)
    return image

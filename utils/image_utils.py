from PIL import Image, ImageFont, ImageDraw
import random
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from scipy import ndimage
from os import name as os_name
import cv2

NEWLINE_REPLACEMENT_STRING = '<br>'
SPACE_REPLACEMENT_STRING = '<sp>'
if os_name == 'nt':
    FONT = 'arial.ttf'
else:
    FONT = "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"



def crop_image(image, size, rand=False):
    if type(image) is str:
        image = Image.open(image)
    w, h = image.size
    w_0 = 0
    w_1 = w
    h_0 = 0
    h_1 = h
    if w > h:
        if rand:
            w_0 = random.randrange(0, w - h)
        else:
            w_0 = (w - h) // 2
        w_1 = w_0 + h
    elif h > w:
        if rand:
            h_0 = random.randrange(0, h - w)
        else:
            h_0 = (h - w) // 2
        h_1 = h_0 + w
    return image.crop((w_0, h_0, w_1, h_1)).resize((size, size), Image.ANTIALIAS)


def blur_image(sy, mask):
    blurred = cv2.GaussianBlur(np.array(sy), (5, 5), 0)
    ma = mask.squeeze().astype(bool)
    sy[ma] = blurred[ma]
    return sy

def resize_to_max(image, max_size):
    w, h = image.size
    if w > h:
        h = int(h * max_size / w)
        w = max_size
    else:
        w = int(w * max_size / h)
        h = max_size
    if h == 0 or w == 0:
        return False
    image = image.resize((w, h), resample=Image.BICUBIC)
    return np.array(image)


def distort_vm(image, max_size, scale=False, crop=False, rotate=False, gray=False, blur=False):
    if type(image) is str:
        image = Image.open(image)
    if gray:
        image = image.convert('LA')
    elif image.mode != 'RGBA':
        image = image.convert('RGBA')
    if crop:
        w, h = image.size
        new_w = round(w // 3 + random.random() * (w - w // 3))
        new_h = round(h // 3 + random.random() * (h - h // 3))
        w_0 = round(random.random() * (w - new_w))
        h_0 = round(random.random() * (h - new_h))
        image = image.crop((w_0, h_0, w_0 + new_w, h_0 + new_h))
    if scale:
        scale_factor = 0.75 + random.random() * .5
        w, h = image.size
        if random.random() < 0.5:
            w = round(w * scale_factor)
        else:
            h = round(h * scale_factor)
        image = image.resize((w, h), resample=Image.BICUBIC)
    if rotate:
        image = image.rotate(round(random.random() * 180 - 90), resample=Image.BICUBIC, expand=True)
    return resize_to_max(image, max_size)


def get_image_indices(image):
    if type(image) is str:
        image = Image.open(image)
    if type(image) is not np.ndarray:
        image = np.array(image)
    rows, columns, ch = image.shape
    mask = image[:, :, ch-1] != 0
    indices = np.ma.where(mask)
    return np.array(indices, dtype=np.int32), rows, columns


def fill_image(target, source, offset_x, offset_y):
        source_size = source.shape
        target[offset_y: offset_y + source_size[0], offset_x: offset_x + source_size[1], :] = source


def save_image(image, name):
    image = image / 2 + 0.5  # unnormalize
    image = image.cpu().numpy()
    plt.imsave(name, np.transpose(image, (1, 2, 0)))


def show_image(image):
    if type(image) is np.ndarray:
        image = Image.fromarray(image)
    image.show()


def imshow(image):
    image = image / 2 + 0.5     # unnormalize
    npimg = image.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show(block=True)


def get_opacity_field(size, mean, var):
    if var == 0:
        return np.zeros([size, size], dtype=np.float32) + mean
    var = random.random() * var
    opacity_field = np.random.uniform(mean - var / 2, mean + var / 2, [size, size])
    opacity_field = ndimage.filters.gaussian_filter(opacity_field, sigma= 1)
    opacity_field = np.clip(opacity_field , 0, 1)
    return opacity_field


def get_color_field(color, size):
    color_field = np.random.uniform(color - 10, color + 10, [size, size, 1])
    color_field = np.repeat(np.clip(color_field, 0, 255), 3, axis=2)
    return color_field


def get_text_motif(text, color=(255, 255, 255, 255), font=FONT, border=0):
    if border != 0:
        border_size = random.randint(0, border)
        border_color = random.randint(100, 240)
        border_color = (border_color, border_color, border_color, 255)
    else:
        border_size = 0
    font = ImageFont.truetype(font, 50)
    text = text.replace(SPACE_REPLACEMENT_STRING, ' ')
    lines = text.split(NEWLINE_REPLACEMENT_STRING)
    img_width, line_height = font.getsize(lines[0])
    line_space = round(line_height * 0.2)
    y = []
    for line in lines:
        line_width, line_height, = font.getsize(line)
        if line_width > img_width:
            img_width = line_width
        y.append(line_height + line_space)
    img_height = sum(y) - line_space
    image = Image.new('RGBA', (img_width + border_size * 2, img_height + border_size * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    cur_y = border_size
    for idx, line in enumerate(lines):
        # thin border
        if border and border_size:
            draw.text((0, cur_y - border_size), line, font=font, fill=border_color)
            draw.text((border_size * 2, cur_y - border_size), line, font=font, fill=border_color)
            draw.text((0, cur_y + border_size), line, font=font, fill=border_color)
            draw.text((border_size * 2, cur_y + border_size), line, font=font, fill=border_color)

        # now draw the text over it
        draw.text((border_size, cur_y), line, fill=color, font=font)
        cur_y += y[idx]
    return image


def permute_image(sy_image, mask, multiplier=1):
    shifter = Shifter(mask, multiplier)
    coords_in_input = shifter.get_new_coords()
    sy_permuted = ndi.map_coordinates(sy_image, coords_in_input)

    # sy_permuted = ndimage.geometric_transform(sy_image, shifter.geometric_shift, mode='nearest')
    return sy_permuted


class Shifter:

    xx, yy, zz = None, None, None

    def __init__(self, mask, multiplier):
        self.img_size = mask.shape[:-1]
        self.mask = mask
        self.ch = 3
        shift_x, shift_y = ((-1)*np.ones(self.img_size) + 2 * np.random.rand(self.img_size[0], self.img_size[1])) * multiplier, \
                           ((-1)*np.ones(self.img_size) + 2 * np.random.rand(self.img_size[0], self.img_size[1])) * multiplier
        self.shift_x, self.shift_y = ndimage.filters.gaussian_filter(shift_x, 0.5), ndimage.filters.gaussian_filter(shift_y, 0.5)
        for shift in (self.shift_x, self.shift_y):
            shift[0, :] = 0
            shift[self.img_size[0] - 1, :] = 0
            shift[:, 0] = 0
            shift[: , self.img_size[1] - 1] = 0
        self.shift_x, self.shift_y = np.expand_dims(self.shift_x, 2), np.expand_dims(self.shift_y, 2)
        self.shift_x, self.shift_y = np.repeat(self.shift_x, 3, 2), np.repeat(self.shift_y, 3, 2)
        self.mask = np.repeat(self.mask.astype(bool), 3, 2)

    def get_new_coords(self):
        if Shifter.xx is None:
            Shifter.xx, Shifter.yy, Shifter.zz = np.meshgrid(np.arange(self.img_size[0]), np.arange(self.img_size[1]), np.arange(3))
            Shifter.xx, Shifter.yy = Shifter.xx.astype(float), Shifter.yy.astype(float)
        _xx, _yy = Shifter.xx.copy(), Shifter.yy.copy()
        _xx[self.mask] -= self.shift_x[self.mask]
        _yy[self.mask] -= self.shift_y[self.mask]
        return (_yy, _xx, Shifter.zz)


    def geometric_shift(self, coords):
        if self.mask[coords[0], coords[1], 0] == 0:
            return coords
        new_place = (coords[0] - self.shift_x[coords[1], coords[0]], coords[1] - self.shift_y[coords[1],coords[0]])
        if self.mask[int(new_place[0]), int(new_place[1]), 0] == 0:
            return coords
        return new_place + coords[2:]

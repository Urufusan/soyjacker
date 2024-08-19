# Copyright 2024 Urufusan.
# SPDX-License-Identifier: 	AGPL-3.0-or-later

import itertools
import os
import random
import subprocess
import sys
from io import BytesIO
from shutil import which
from typing import Callable

from PIL import Image

SOYMAIN_PROJ_FOLDER = os.path.dirname(__file__)
MIN_WIDTH = 400
MIN_HEIGHT = 400


# Helper funcs

def c_tool_exists(_cmd_name):
    return which(_cmd_name) is not None


# TODO: Implement proper wayland support
"""
[black image on gnome wayland · Issue #67 · naelstrof/maim](https://github.com/naelstrof/maim/issues/67#issuecomment-974622572 "black image on gnome wayland · Issue #67 · naelstrof/maim")
[grim: Grab images from a Wayland compositor](https://sr.ht/~emersion/grim/ "grim: Grab images from a Wayland compositor")
[emersion/slurp: Select a region in a Wayland compositor](https://github.com/emersion/slurp "emersion/slurp: Select a region in a Wayland compositor")
"""

SCREENSHOT_TOOL_CMDLINE = ["maim", "-s", "--format", "png", "/dev/stdout"] if c_tool_exists("maim") else ["spectacle", "--background", "--nonotify", "-o", "/proc/self/fd/1"]


def random_asset_picker(_image_type_prefix: str):
    _file_list = list(filter(lambda _lambda_e: _lambda_e.startswith(_image_type_prefix), os.listdir(f"{SOYMAIN_PROJ_FOLDER}/assets")))
    _random_image_path = f"{SOYMAIN_PROJ_FOLDER}/assets/{random.choice(_file_list)}"
    return _random_image_path


def get_screenie():
    _temp_proc = subprocess.check_output(SCREENSHOT_TOOL_CMDLINE, shell=("spectacle" in SCREENSHOT_TOOL_CMDLINE))
    return _temp_proc


def find_coeffs(start_points, end_points):
    import numpy

    """Coefficients for perspective transformation"""
    _matrix = []
    for p1, p2 in zip(start_points, end_points):
        _matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        _matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = numpy.matrix(_matrix, dtype=float)
    B = numpy.array(end_points).reshape(8)

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)


def persp_transform(startpoints, endpoints, _im: Image.Image):
    """Perform a perspective transformation using the og and end points"""
    width, height = _im.size
    coeffs = find_coeffs(endpoints, startpoints)

    _im = _im.transform((width, height), Image.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
    return _im


# Actual image gen funcs


def unwrapped_phone_show(image: Image.Image, vertical=False):
    _random_image_path = random_asset_picker("phone_" if vertical else "phonehz_")
    _react_image = Image.open(_random_image_path)
    _foundational_image = Image.new("RGBA", _react_image.size, (0, 0, 0, 0))
    _scaled_user_img = image.resize((int(_react_image.size[0] * 0.75), int(_react_image.size[1] * 0.75)), Image.Resampling.BICUBIC)
    _usr_img_w, _usr_img_h = _scaled_user_img.size
    _foundational_image.paste(_scaled_user_img, (0, 0))

    del _scaled_user_img, image

    _t_ep = os.path.basename(_random_image_path).split("-")[1].split(".")[0].split("_")
    _i_ep = [int(_temp_str2int) for _temp_str2int in _t_ep]

    _startpoints = ((0, 0), (_usr_img_w, 0), (_usr_img_w, _usr_img_h), (0, _usr_img_h))
    _endpoints = ((_i_ep[0:2]), (_i_ep[2:4]), (_i_ep[4:6]), (_i_ep[6:8]))
    _final_image = persp_transform(_startpoints, _endpoints, _foundational_image)
    _final_image.alpha_composite(_react_image)
    
    del _foundational_image, _react_image, 
    
    return _final_image


def soy_show_phone_horizontal(image: Image.Image) -> Image.Image:
    """Random HORIZONTAL phone-showing soyjak (image is pasted inside the phone)"""
    return unwrapped_phone_show(image=image, vertical=False)


def soy_show_phone_vertical(image: Image.Image) -> Image.Image:
    """Random VERTICAL phone-showing soyjak (image is pasted inside the phone)"""
    return unwrapped_phone_show(image=image, vertical=True)


def soy_bubble_react(image: Image.Image):
    """Random bubble reaction soyjak (image is pasted above the speechbubble)"""
    _random_image_path = random_asset_picker("bubble_")
    _react_image = Image.open(_random_image_path)
    _scaled_user_img = image.resize((_react_image.size[0], int((_react_image.size[0] / image.size[0]) * image.size[1])), Image.Resampling.BICUBIC)
    del image
    _foundational_image = Image.new("RGBA", (_react_image.size[0], _react_image.size[1] + _scaled_user_img.size[1]), (0, 0, 0, 255))
    _foundational_image.paste(_scaled_user_img, (0, 0))
    _foundational_image.paste(_react_image, (0, _scaled_user_img.size[1]))
    del _scaled_user_img, _react_image

    return _foundational_image


def soy_point(image: Image.Image, aspect_ratio="Fit"):
    """Two pointing soyjaks (pasted over the background image)"""
    width, height = image.size
    _bg_filler_params = {}
    if width < MIN_WIDTH:
        _bg_filler_params["width"] = MIN_WIDTH
    if height < MIN_HEIGHT:
        _bg_filler_params["height"] = MIN_HEIGHT
    if len(_bg_filler_params.keys()) != 0:
        _t_target_res = (_bg_filler_params.get("width", width), _bg_filler_params.get("height", height))

        _bg_filler_image = Image.new("RGBA", _t_target_res, (0, 0, 0, 255))
        _bg_filler_image.paste(image, (int((_t_target_res[0] / 2) - (width / 2)), int((_t_target_res[1] / 2) - (height / 2))))

        del image
        image = _bg_filler_image
        width, height = image.size

    soy1 = Image.open(f"{SOYMAIN_PROJ_FOLDER}/assets/soy1.png")

    _soy_width, _soy_height = soy1.size
    c = float(height) / float(_soy_height)
    adit = 0.7

    size = (int(_soy_width * c * adit) if aspect_ratio == "Fit" else int(width / 2), int(height * adit))

    soy1 = soy1.resize(size, resample=Image.Resampling.NEAREST)

    s1_mask = Image.open(f"{SOYMAIN_PROJ_FOLDER}/assets/soy1_mask.png")
    s1_mask = s1_mask.resize(size, resample=Image.Resampling.NEAREST)

    soy2 = Image.open(f"{SOYMAIN_PROJ_FOLDER}/assets/soy2.png")
    soy2 = soy2.resize(size, resample=Image.Resampling.NEAREST)

    s2_mask = Image.open(f"{SOYMAIN_PROJ_FOLDER}/assets/soy2_mask.png")
    s2_mask = s2_mask.resize(size, resample=Image.Resampling.NEAREST)

    image.paste(soy2, (width - size[0], height - size[1]), mask=s2_mask)
    image.paste(soy1, (0, height - size[1]), mask=s1_mask)
    del soy1, s1_mask, soy2, s2_mask

    return image


def soy_auto_ratio(image: Image.Image):
    """Automatically choose template based on the aspect ratio of the screenshot"""

    sc_ratio = image.size[0] / image.size[1]
    # TODO: Cmd args for the formats?
    if sc_ratio < 0.8:
        _soyed_image = unwrapped_phone_show(image=image, vertical=True)
    elif sc_ratio < 1.3:
        _soyed_image = soy_bubble_react(image=image)
    else:
        _soyed_image = random.choice((soy_point, unwrapped_phone_show))(image=image)

    del image

    return _soyed_image


def zenity_picker() -> Callable:
    _pre_entry_globals = globals()

    # NOTE: This is very hacky, but it makes auto_ratio appear at the top

    _filter_obj = list(filter(lambda _l_e: _l_e.startswith("soy"), _pre_entry_globals.keys()))
    _filter_obj.insert(0, _filter_obj.pop(-1))
    _soy_f_map = list(enumerate(_filter_obj))

    _zenity_vals = [("FALSE" if _cnt_i else "TRUE", str(_cnt_i), _pre_entry_globals.get(_tempthing).__doc__) for _cnt_i, _tempthing in _soy_f_map]
    # print(_zenity_vals)
    _selected_value = subprocess.check_output(
        [
            "zenity",
            "--list",
            "--radiolist",
            "--title=Soyshot mode selection",
            "--print-column=ALL",
            "--column=a",
            "--column=b",
            "--column=Template format",
            "--hide-column=2",
            "--text",
            "Select a template you'd like to use:",
            "--hide-header",
            "--print-column=2",
            "--width=550",
            "--height=200",
            *list(itertools.chain.from_iterable(_zenity_vals)),
        ]
    )

    return _pre_entry_globals.get(dict(_soy_f_map)[int(_selected_value)])


if __name__ == "__main__":
    if _prog_args := sys.argv[1:2]:
        match _prog_args[0]:
            case "picker":
                _selected_function = zenity_picker()
                screenie = Image.open(BytesIO(get_screenie()))
                final_image = _selected_function(image=screenie)
            case "ratio":
                screenie = Image.open(BytesIO(get_screenie()))
                final_image = soy_auto_ratio(image=screenie)
            case _:
                sys.exit("Argument error!")
    else:
        screenie = Image.open(BytesIO(get_screenie()))
        final_image = soy_point(image=screenie)

    proc_temp = subprocess.Popen(["copyq", "copy", "image/png", "-"], stdin=subprocess.PIPE, bufsize=-1)
    final_image.save(proc_temp.stdin, format="PNG")
    proc_temp.stdin.close()
    del final_image, screenie

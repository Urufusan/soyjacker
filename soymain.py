# Copyright 2024 Urufusan.
# SPDX-License-Identifier: 	AGPL-3.0-or-later

import itertools
import os
import random
import subprocess
import sys
import time
from io import BytesIO
from shutil import which
from typing import Callable

from PIL import Image, ImageOps

SOYMAIN_PROJ_FOLDER = os.path.dirname(__file__)
MIN_WIDTH = 400
MIN_HEIGHT = 400
random.seed(int(time.time()) // int.from_bytes(os.urandom(1), "big", signed=False))

# My own locally compiled yad binary
YAD_LOCAL_PATH = os.path.expanduser("~/Scripts/yad-14.1/out/yad")

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

def random_asset_picker(_image_type_prefix: str) -> str:
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

    _m_a = numpy.matrix(_matrix, dtype=float)
    _m_b = numpy.array(end_points).reshape(8)

    res = numpy.dot(numpy.linalg.inv(_m_a.T * _m_a) * _m_a.T, _m_b)
    return numpy.array(res).reshape(8)


def persp_transform(startpoints, endpoints, _im: Image.Image) -> Image.Image:
    """Perform a perspective transformation using the og and end points"""
    width, height = _im.size
    coeffs = find_coeffs(endpoints, startpoints)

    _im = _im.transform((width, height), Image.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
    return _im


# Actual image gen funcs

def unwrapped_2d_paste(image: Image.Image, asset_name_prefix: str = "") -> Image.Image:
    _random_image_path = random_asset_picker("ihate_" if not asset_name_prefix else asset_name_prefix)
    _react_image = Image.open(_random_image_path)
    
    _t_ep = os.path.basename(_random_image_path).split("-")[1].split(".")[0].split("_")
    _i_ep = [int(_temp_str2int) for _temp_str2int in _t_ep]
    
    _container_size = (_i_ep[2] - _i_ep[0], _i_ep[3] - _i_ep[1])
    _container_image = ImageOps.contain(image, _container_size, Image.Resampling.LANCZOS)

    _react_image.paste(_container_image, 
                       (_i_ep[0] + ((_container_size[0] - _container_image.size[0])//2),
                        _i_ep[1] + ((_container_size[1] - _container_image.size[1])//2)))
    
    del _container_image
    
    return _react_image

def unwrapped_phone_show(image: Image.Image, vertical: bool = False, asset_name_prefix: str = "") -> Image.Image:
    _random_image_path = random_asset_picker(("phone_" if vertical else "phonehz_") if not asset_name_prefix else asset_name_prefix)
    _react_image = Image.open(_random_image_path)
    _foundational_image = Image.new("RGBA", _react_image.size, (0, 0, 0, 0))
    _scaled_user_img = image.resize((int(_react_image.size[0] * 0.75), int(_react_image.size[1] * 0.75)), Image.Resampling.BICUBIC)
    _usr_img_w, _usr_img_h = _scaled_user_img.size
    _foundational_image.paste(_scaled_user_img, (0, 0))

    del _scaled_user_img

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

def soy_shitty_setup(image: Image.Image) -> Image.Image:
    """Random shitty PC setup template (image is pasted inside the screen)"""
    return unwrapped_phone_show(image=image, asset_name_prefix="cursedsetup_")

def soy_bubble_react(image: Image.Image) -> Image.Image:
    """Random bubble reaction soyjak (image is pasted above the speechbubble)"""
    _random_image_path = random_asset_picker("bubble_")
    _react_image = Image.open(_random_image_path)
    _scaled_user_img = image.resize((_react_image.size[0], int((_react_image.size[0] / image.size[0]) * image.size[1])), Image.Resampling.BICUBIC)
    # del image
    _foundational_image = Image.new("RGBA", (_react_image.size[0], _react_image.size[1] + _scaled_user_img.size[1]), (0, 0, 0, 255))
    _foundational_image.paste(_scaled_user_img, (0, 0))
    _foundational_image.paste(_react_image, (0, _scaled_user_img.size[1]))
    del _scaled_user_img, _react_image

    return _foundational_image

def soy_hate_shirt(image: Image.Image) -> Image.Image:
    """Random "I HATE..." shirt soyjak (image is pasted inside the shirt)"""
    return unwrapped_2d_paste(image)

def soy_point(image: Image.Image, aspect_ratio:str = "Fit") -> Image.Image:
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


def soy_auto_ratio(image: Image.Image) -> Image.Image:
    """Automatically choose template based on the aspect ratio of the screenshot"""

    sc_ratio = image.size[0] / image.size[1]
    # TODO: Cmd args for the formats?
    if sc_ratio < 0.8:
        _soyed_image = unwrapped_phone_show(image=image, vertical=True)
    elif sc_ratio < 1.3:
        _soyed_image = soy_bubble_react(image=image)
    else:
        _soyed_image = random.choice((soy_point, unwrapped_phone_show))(image=image)

    # del image

    return _soyed_image


def zenity_picker() -> Callable:
    _pre_entry_globals = globals()


    _filter_obj = list(filter(lambda _l_e: _l_e.startswith("soy"), _pre_entry_globals.keys()))
    
    # NOTE: This is very hacky, but it makes auto_ratio appear at the top
    _filter_obj.insert(0, _filter_obj.pop(-1))
    
    _soy_f_map = list(enumerate(_filter_obj))

    _zenity_vals = [("FALSE" if _cnt_i else "TRUE", str(_cnt_i), _pre_entry_globals.get(_tempthing).__doc__) for _cnt_i, _tempthing in _soy_f_map]
    # print(_zenity_vals)
    
    # Check for yad and use that instead of zenity
    _yad_bin = "zenity"
    if c_tool_exists("yad"):
        _yad_bin = "yad"
    elif os.path.isfile(YAD_LOCAL_PATH):
        _yad_bin = YAD_LOCAL_PATH
    
    _yad_bin = os.environ.get("SOY_MENU_PICKER_BIN", _yad_bin)
    if _yad_bin == "zenity":
        _yad_bin = None
    
    _subprocess_CMD = [
            "zenity" if not _yad_bin else _yad_bin,
            "--list",
            "--radiolist",
            "--title=Soyshot mode selection",
            # "--print-column=0",
            "--column=a",
            "--column=b",
            "--column=Template format",
            "--hide-column=2",
            "--text",
            "Select a template you'd like to use:",
            "--hide-header" if not _yad_bin else "--no-headers",
            "--print-column=2",
            "--width=550",
            "--height=250",
            *list(itertools.chain.from_iterable(_zenity_vals)),
    ]
    if _yad_bin: _subprocess_CMD.insert(4, "--buttons-layout=center")
    
    _selected_value = subprocess.check_output(_subprocess_CMD)

    return _pre_entry_globals.get(dict(_soy_f_map)[int(_selected_value.decode().strip().strip("|"))])


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

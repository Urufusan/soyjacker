# Copyright 2024 Urufusan.
# SPDX-License-Identifier: 	AGPL-3.0-or-later

import subprocess
# import time
# from typing import Literal
from PIL import Image
from io import BytesIO
import os
import random
from shutil import which
import numpy

SOYMAIN_PROJ_FOLDER = os.path.dirname(__file__)
MIN_WIDTH = 400
MIN_HEIGHT = 400


# Helper funcs
def c_tool_exists(_cmd_name) -> True:
    return which(_cmd_name) is not None

SCREENSHOT_TOOL_CMDLINE = ["maim", "-s", "--format", "png", "/dev/stdout"] if c_tool_exists("maim") else ["spectacle", "--background", "--nonotify", "-o", "/proc/self/fd/1"]

def random_asset_picker(_image_type_prefix: str):
    _file_list = list(filter(lambda _lambda_e: _lambda_e.startswith(_image_type_prefix), os.listdir(f"{SOYMAIN_PROJ_FOLDER}/assets")))
    _random_image_path = f"{SOYMAIN_PROJ_FOLDER}/assets/{random.choice(_file_list)}"
    return _random_image_path

def get_screenie():
    _temp_proc = subprocess.check_output(SCREENSHOT_TOOL_CMDLINE, shell=("spectacle" in SCREENSHOT_TOOL_CMDLINE))
    return _temp_proc

def find_coeffs(start_points, end_points):
    """Coefficients for perspective transformation"""
    matrix = []
    for p1, p2 in zip(start_points, end_points):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = numpy.matrix(matrix, dtype=float)
    B = numpy.array(end_points).reshape(8)

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)

def persp_transform(startpoints, endpoints, _im : Image.Image):
    """Perform a perspective transformation using the og and end points"""
    width, height = _im.size
    coeffs = find_coeffs(endpoints, startpoints)

    _im = _im.transform((width, height), Image.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
    return _im

# Actual image gen funcs

def soy_phone_show(image: Image.Image, vertical = False):
    _random_image_path = random_asset_picker("phone_" if vertical else "phonehz_")
    _react_image = Image.open(_random_image_path)
    _foundational_image = Image.new("RGBA", _react_image.size, (0,0,0,0))
    _scaled_user_img = image.resize((int(_react_image.size[0]*0.75), int(_react_image.size[1]*0.75)), Image.Resampling.BICUBIC)
    _usr_img_w, _usr_img_h = _scaled_user_img.size
    _foundational_image.paste(_scaled_user_img, (0, 0))
    
    del _scaled_user_img, image
    
    _t_ep = os.path.basename(_random_image_path).split("-")[1].split(".")[0].split("_")
    _i_ep = [int(_temp_str2int) for _temp_str2int in _t_ep]
    # print(_i_ep)
    _startpoints = ((0, 0),(_usr_img_w, 0), (_usr_img_w, _usr_img_h), (0, _usr_img_h))
    _endpoints = (
        (_i_ep[0:2]),
        (_i_ep[2:4]),
        (_i_ep[4:6]),
        (_i_ep[6:8])
    )
    _final_image = persp_transform(_startpoints, _endpoints, _foundational_image)
    _final_image.alpha_composite(_react_image)
    
    del _foundational_image, _react_image, 
    
    return _final_image

def soy_bubble_react(image: Image.Image):
    _random_image_path = random_asset_picker("bubble_")
    _react_image = Image.open(_random_image_path)
    _scaled_user_img = image.resize((_react_image.size[0], int((_react_image.size[0]/image.size[0]) * image.size[1])), Image.Resampling.BICUBIC)
    del image
    _foundational_image = Image.new("RGBA", (_react_image.size[0], _react_image.size[1]+_scaled_user_img.size[1]), (0,0,0,255))
    _foundational_image.paste(_scaled_user_img, (0, 0))
    _foundational_image.paste(_react_image, (0, _scaled_user_img.size[1]))
    del _scaled_user_img, _react_image
    
    return _foundational_image



def soy_point(image: Image.Image, aspect_ratio = "Fit"):
    width, height = image.size
    _bg_filler_params = {}
    if width < MIN_WIDTH: _bg_filler_params["width"] = MIN_WIDTH
    if height < MIN_HEIGHT: _bg_filler_params["height"] = MIN_HEIGHT
    if len(_bg_filler_params.keys()) != 0:
        _t_target_res = (
            _bg_filler_params.get("width", width), _bg_filler_params.get("height", height)
            )
        
        _bg_filler_image = Image.new("RGBA", _t_target_res, (0, 0, 0, 255))
        _bg_filler_image.paste(image, (int((_t_target_res[0]/2) - (width/2)), int((_t_target_res[1]/2) - (height/2))))
        
        del image
        image = _bg_filler_image
        width, height = image.size

    soy1 = Image.open(f"{SOYMAIN_PROJ_FOLDER}/assets/soy1.png")

    soy_width, soy_height = soy1.size
    c = float(height) / float(soy_height)
    adit = 0.7
    # print(c)
    size = (int(soy_width * c * adit) if aspect_ratio == "Fit" else int(width / 2), int(height * adit))

    soy1 = soy1.resize(size, resample=Image.Resampling.NEAREST)

    s1_mask = Image.open(f"{SOYMAIN_PROJ_FOLDER}/assets/soy1_mask.png")
    s1_mask = s1_mask.resize(size, resample=Image.Resampling.NEAREST)

    soy2 = Image.open(f"{SOYMAIN_PROJ_FOLDER}/assets/soy2.png")
    soy2 = soy2.resize(size, resample=Image.Resampling.NEAREST)

    s2_mask = Image.open(f"{SOYMAIN_PROJ_FOLDER}/assets/soy2_mask.png")
    s2_mask = s2_mask.resize(size, resample=Image.Resampling.NEAREST)

    image.paste(soy2, (width - size[0], height-size[1]), mask=s2_mask)
    image.paste(soy1, (0, height-size[1]), mask=s1_mask)
    del soy1, s1_mask, soy2, s2_mask
    
    return image

    # _buf.seek(0)
    # await save_and_send(ctx, frames, image)

def main():
    # _buf = BytesIO()
    screenie = Image.open(BytesIO(get_screenie())) # Over pipes
    sc_ratio = screenie.size[0]/screenie.size[1]
    # TODO: Cmd args for the formats?
    if sc_ratio < 0.8:
        image = soy_phone_show(image=screenie, vertical=True)
    elif sc_ratio < 1.3:
        image = soy_bubble_react(image=screenie)
    else:
        image = random.choice((soy_point, soy_phone_show))(image=screenie)

    proc_temp = subprocess.Popen(["copyq", "copy", "image/png", "-"], stdin=subprocess.PIPE, bufsize=-1)
    image.save(proc_temp.stdin, format="PNG")

if __name__ == "__main__":
    # print(SOYMAIN_PROJ_FOLDER)
    main()
    # exit(0)
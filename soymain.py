# Copyright 2024 Urufusan.
# SPDX-License-Identifier: 	AGPL-3.0-or-later

import subprocess
import time
from typing import Literal
from PIL import Image
from io import BytesIO
import os

SOYMAIN_PROJ_FOLDER = os.path.dirname(__file__)
MIN_WIDTH = 400
MIN_HEIGHT = 400

def get_screenie():
    _temp_proc = subprocess.check_output(["maim", "-s", "--format", "png", "/dev/stdout"])
    return _temp_proc


def soyjak(
    aspect_ratio: Literal["Fit", "Stretch"] = "Fit",
):
    # _buf = BytesIO()
    image = Image.open(BytesIO(get_screenie())) # Over pipes

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
    print(c)
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
    proc_temp = subprocess.Popen(["copyq", "copy", "image/png", "-"], stdin=subprocess.PIPE, bufsize=-1)
    image.save(proc_temp.stdin, format="PNG")

    # _buf.seek(0)
    # await save_and_send(ctx, frames, image)

if __name__ == "__main__":
    print(SOYMAIN_PROJ_FOLDER)
    soyjak()
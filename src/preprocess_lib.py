import argparse
from pathlib import Path

import albumentations as albu
import numpy as np
from PIL import Image


def get_xywh(box_offset, lbl_fp, img_w, img_h):
    with open(lbl_fp) as f:
        for idx, line in enumerate(f):
            if idx == box_offset:
                line = line.strip()
                _, cx, cy, bw, bh = line.split(" ")
                cx = int(float(cx) * img_w)
                cy = int(float(cy) * img_h)
                bw = int(float(bw) * img_w)
                bh = int(float(bh) * img_h)

                return cx, cy, bw, bh


def xywh_to_xyxy(x, y, w, h):
    tlx = x - w // 2
    tly = y - h // 2
    brx = x + w - w // 2
    bry = y + h - h // 2

    return tlx, tly, brx, bry


def get_candidate_dims(smaller_dim, larger_dim, extra_context):
    candidate_dim = int(smaller_dim * (100 + extra_context) / 100)
    """
    Cases with skewed aspect ratio. Larger side much larger than smaller side
    """
    if candidate_dim < larger_dim:
        candidate_dim = int(larger_dim * (100 + 20) / 100)

    return candidate_dim, candidate_dim


def extend_where_possible(val1, val2, bound1, bound2):
    # No space to extend
    if val1 < bound1 and val2 > bound2:
        val1 = bound1
        val2 = bound2
    # One side has more room to extend
    elif val1 < bound1 and val2 < bound2:
        val2 += -val1
        val1 = bound1
        # val2 should not exceed its own bounds
        val2 = min(val2, bound2)
    elif val1 > bound1 and val2 > bound2:
        val1 -= val2 - bound2
        val2 = bound2
        val1 = max(val1, bound1)
    else:
        # both are inside bounds, no changes needed
        pass
    return val1, val2


def resize_with_pad_if_needed(context_area_boundaries, tgt_dim, box_coordinates):
    """
    Normalise box coordinates using context_area_boundaries
    """
    box_coordinates["tlx"] = (box_coordinates["tlx"] - context_area_boundaries["tlx"]) / context_area_boundaries[
        "width"
    ]
    box_coordinates["tly"] = (box_coordinates["tly"] - context_area_boundaries["tly"]) / context_area_boundaries[
        "height"
    ]
    box_coordinates["brx"] = (box_coordinates["brx"] - context_area_boundaries["tlx"]) / context_area_boundaries[
        "width"
    ]
    box_coordinates["bry"] = (box_coordinates["bry"] - context_area_boundaries["tly"]) / context_area_boundaries[
        "height"
    ]

    box_tlx = max(box_coordinates["tlx"], 0)
    box_tly = max(box_coordinates["tly"], 0)
    box_brx = min(box_coordinates["brx"], 1)
    box_bry = min(box_coordinates["bry"], 1)

    # assert box_tlx >= 0, print(f"Got box_tlx {box_tlx} {box_coordinates}")
    # assert box_tly >= 0, print(f"Got box_tly {box_tly} {box_coordinates}")
    # assert box_brx <= 1, print(f"Got box_brx {box_brx} {box_coordinates}")
    # assert box_bry <= 1, print(f"Got box_bry {box_bry} {box_coordinates}")

    if context_area_boundaries["width"] > context_area_boundaries["height"]:
        scale = tgt_dim / context_area_boundaries["width"]
        scale = min(scale, 2)
    else:
        scale = tgt_dim / context_area_boundaries["height"]
        scale = min(scale, 2)

    new_w = int(context_area_boundaries["width"] * scale)
    new_h = int(context_area_boundaries["height"] * scale)

    # Assumption is that LetterBox Resize will keep equal border on both sides where padding is required
    h_pad_required = (tgt_dim - new_h) // 2
    w_pad_required = (tgt_dim - new_w) // 2

    """
    Get Pixel Coordinates on letterbox resized context area
    """
    box_tlx *= new_w
    box_tly *= new_h
    box_brx *= new_w
    box_bry *= new_h

    """
    Box Mask Coordinates => X0` Y0` X1`  Y1`
    """

    box_tlx = int(box_tlx) + w_pad_required
    box_tly = int(box_tly) + h_pad_required
    box_brx = int(box_brx) + w_pad_required
    box_bry = int(box_bry) + h_pad_required

    box_mask_coordinates = {}
    box_mask_coordinates["tlx"] = box_tlx
    box_mask_coordinates["tly"] = box_tly
    box_mask_coordinates["brx"] = box_brx
    box_mask_coordinates["bry"] = box_bry

    """
    Resized Crop => X2` Y2` X3` Y3`
    """
    resized_image_coordinates = {}
    resized_image_coordinates["tlx"] = w_pad_required
    resized_image_coordinates["tly"] = h_pad_required
    resized_image_coordinates["brx"] = w_pad_required + new_w
    resized_image_coordinates["bry"] = h_pad_required + new_h

    return scale, resized_image_coordinates, box_mask_coordinates


def get_more_context_crop(crop_img_shape, full_img_shape, extra_context, box_coordinates):
    if crop_img_shape["width"] < crop_img_shape["height"]:
        candidate_w, candidate_h = get_candidate_dims(crop_img_shape["width"], crop_img_shape["height"], extra_context)
    else:
        candidate_w, candidate_h = get_candidate_dims(crop_img_shape["height"], crop_img_shape["width"], extra_context)

    w_diff = candidate_w - crop_img_shape["width"]
    h_diff = candidate_h - crop_img_shape["height"]

    left_pad = w_diff // 2
    right_pad = w_diff - left_pad

    top_pad = h_diff // 2
    bottom_pad = h_diff - top_pad

    """
    Padding for each dimension has been calculated. These now have to be clamped
    to not exceed full image boundaries
    In case padding exceeds boundary, it can be compensated on the other side.
    This should then be clamped as necessary
    """
    pre_clamp_left = box_coordinates["tlx"] - left_pad
    pre_clamp_right = box_coordinates["brx"] + right_pad
    pre_clamp_top = box_coordinates["tly"] - top_pad
    pre_clamp_bottom = box_coordinates["bry"] + bottom_pad

    new_left, new_right = extend_where_possible(pre_clamp_left, pre_clamp_right, 0, full_img_shape["width"])
    new_top, new_bottom = extend_where_possible(pre_clamp_top, pre_clamp_bottom, 0, full_img_shape["height"])

    new_boundaries = {}
    new_boundaries["tlx"] = new_left
    new_boundaries["brx"] = new_right
    new_boundaries["tly"] = new_top
    new_boundaries["bry"] = new_bottom

    return new_boundaries


def create_more_context_crop_and_person_box_mask(full_img_shape, box_coordinates):

    EXTRA_CONTEXT = 150
    TGT_DIM = 224

    crop_img_shape = {
            "height": box_coordinates["bry"] - box_coordinates["tly"],
            "width": box_coordinates["brx"] - box_coordinates["tlx"]
    }

    """
    context_area_boundaries => X2 Y2 X3 Y3
    """
    context_area_boundaries = get_more_context_crop(crop_img_shape, full_img_shape, EXTRA_CONTEXT, box_coordinates)

    """
    Skipping `make_more_context_crop_square`. 
    Will be handled by Prince's code (Letterboxing)
    """
    context_area_boundaries["width"] = context_area_boundaries["brx"] - context_area_boundaries["tlx"]
    context_area_boundaries["height"] = context_area_boundaries["bry"] - context_area_boundaries["tly"]

    # assert context_area_boundaries["width"] > 0
    # assert context_area_boundaries["height"] > 0

    scale, resized_image_coordinates_target_scale, box_mask_coordinates_target_scale = resize_with_pad_if_needed(
        context_area_boundaries, TGT_DIM, box_coordinates
    )

    return context_area_boundaries, scale, resized_image_coordinates_target_scale, box_mask_coordinates_target_scale


def make_crop_and_mask(
    full_img, context_area_boundaries, scale, resized_image_coordinates, box_mask_coordinates, tgt_dim
):
    """
    Script to generate target image and person box mask.
    For Debugging purposes only. Need not go into the final code
    """
    more_context_crop = full_img.crop(
        (
            context_area_boundaries["tlx"],
            context_area_boundaries["tly"],
            context_area_boundaries["brx"],
            context_area_boundaries["bry"],
        )
    )

    new_w = int(more_context_crop.width * scale)
    new_h = int(more_context_crop.height * scale)

    resize_fn = albu.Resize(height=new_h, width=new_w, interpolation=2)
    output = resize_fn(image=np.array(more_context_crop))
    _transformed_img = output["image"]

    transformed_img = np.zeros((tgt_dim, tgt_dim, 3), dtype=np.uint8)
    transformed_img[
        resized_image_coordinates["tly"] : resized_image_coordinates["bry"],
        resized_image_coordinates["tlx"] : resized_image_coordinates["brx"],
    ] = _transformed_img

    person_box_mask = np.zeros((tgt_dim, tgt_dim), dtype=np.uint8)
    person_box_mask[
        box_mask_coordinates["tly"] : box_mask_coordinates["bry"],
        box_mask_coordinates["tlx"] : box_mask_coordinates["brx"],
    ] = 255

    return transformed_img, person_box_mask


def main(args):
    crops_dir = Path(args.person_crops)
    lbl_dir = Path(args.labels)
    full_img_dir = Path(args.full_images)
    tgt_dim = 224
    extra_context = 150
    out_dir = Path(args.output)
    test_flag = args.test

    for idx, crop_img_fp in enumerate(crops_dir.iterdir()):
        print(f"\r{idx}", end="")

        crop_img = Image.open(crop_img_fp)
        if crop_img.height * crop_img.width < 1500:
            continue

        full_img_fp = full_img_dir / crop_img_fp.name.split("--")[1]
        if not full_img_fp.exists():
            raise FileNotFoundError(f"Full Image not found for person crop: {crop_img_fp}")
        full_img = Image.open(full_img_fp)

        box_offset = int(crop_img_fp.name.split("--")[0])
        lbl_fp = lbl_dir / full_img_fp.name.replace(".jpg", ".txt")
        x, y, w, h = get_xywh(box_offset, lbl_fp, full_img.width, full_img.height)
        tlx, tly, brx, bry = xywh_to_xyxy(x, y, w, h)

        box_coordinates = {}
        box_coordinates["tlx"] = tlx
        box_coordinates["tly"] = tly
        box_coordinates["brx"] = brx
        box_coordinates["bry"] = bry


        full_img_shape = {
            "width": full_img.width,
            "height": full_img.height,
        }

        """
        Entrypoint for getting required coordinates
        This function should be used directly to get the required results
        It takes 2 inputs-
            * Full Image Shape (Dictionary containing height and width values)
            * Box Coordinates (Dictionary  containing pixel offsets to top left
              and bottom right coordinates of person crop in the full image)

        This function will return the following coordinates required to generate
        the input to the segmentation model-
            * context_area_boundaries: 
                List[int] (X2 Y2 X3 Y3) [Full Image Coordinates]
                Extended context area in the full image
            * scale:
                float
                Scale used for Letterbox resizing
            * resized_image_coordinates_target_scale:
                List[int] (X2` Y2` X3` Y3`) [Target Image Coordinates]
                Coordinates for resized image
            * box_mask_coordinates_target_scale:
                List[int] (X0` Y0` X1` Y1`) [Target Image Coordinates]
                Coordinates for creation of person box mask. Pixels in this box
                should be set to 255
        """

        (
            context_area_boundaries, # X2 Y2 X3 Y3
            scale,
            resized_image_coordinates_target_scale, # X2` Y2` X3` Y3` Values in range [0, 224]
            box_mask_coordinates_target_scale, # X0` Y0` X1` Y1` Values in range [0, 224]
        ) = create_more_context_crop_and_person_box_mask(full_img_shape, box_coordinates)

        if test_flag:
            """
            Dump images for debugging and testing
            """
            new_img, box_mask = make_crop_and_mask(
                full_img, context_area_boundaries, scale, resized_image_coordinates_target_scale, box_mask_coordinates_target_scale, tgt_dim
            )

            outpath = out_dir / crop_img_fp.name
            outpath.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(new_img).save(outpath)

            outpath = out_dir / crop_img_fp.name.replace(".jpg", "_person_box_mask.png")
            outpath.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(box_mask).save(outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--person-crops", type=str, help="Path to person crops")
    parser.add_argument("--labels", type=str, help="Path to full image labels")
    parser.add_argument("--full-images", type=str, help="Path to full images")
    parser.add_argument("--output", type=str, help="Path to output directory")
    parser.add_argument(
        "--test", action="store_true", help="Flag to save output images and masks from this transformation"
    )

    args = parser.parse_args()
    main(args)

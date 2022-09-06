import argparse

import gwyfile
import numpy as np
import torch
from gwyfile.objects import GwyDataField

from eval.annotator import set_manual_offset
from network.unet import ResUnet
from utils.image import remove_offset_lr, enforce_img_size_for_nn, normalize_joint, subtract_mean_plane_both, \
    subtract_mean_plane, denormalize, normalize, load_lr_img_from_gwy, line_by_line_level


def parse_command_line():
    """ Parser used for training and inference returns args. Sets up GPUs."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', action='store_true', default=False, help='Whether to subtract the mean plane before inference')
    parser.add_argument('--mask', action='store_true', default=False, help='Whether to use mask for levelling')
    parser.add_argument('-ll', '--line_by_line_level', type=int, default=0, help='Line by line leveling degree')
    parser.add_argument('-m', '--manual_offset', action='store_true', default=False,
                        help='Whether the offset needs to be set manually. Controlled by key presses: t and v control '
                             'contrast, k and s control the offset, c continues to next image and saves offset')
    parser.add_argument('model_path', help='Path to the .pth model, alternatively use baseline to run the baseline model')
    parser.add_argument('gwy_path', help='Path to a gwy file. The gwyfile may need the correct metadata so feel free to modify this script')
    args = parser.parse_args()
    return args


def get_model(model_path):
    model = ResUnet(2).cuda()
    print("Using model from: ", model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def preprocess(img_l, img_r, level, ll, mask):
    if mask is not None:
        if level:
            # img_l, img_r = subtract_mean_plane_both(img_l, img_r, mask=entry['mask'])
            img_l = subtract_mean_plane(img_l, mask=mask)
            img_r = subtract_mean_plane(img_r, mask=mask)
        if ll > 0:
            img_l = line_by_line_level(img_l, ll, mask=mask)
            img_r = line_by_line_level(img_r, ll, mask=mask)
    else:
        if level:
            # img_l, img_r = subtract_mean_plane_both(img_l, img_r)
            img_l = subtract_mean_plane(img_l)
            img_r = subtract_mean_plane(img_r)
        if ll > 0:
            img_l = line_by_line_level(img_l, ll)
            img_r = line_by_line_level(img_r, ll)
    return img_l, img_r

def inference(model, img_l, img_r, mask=None, level=False, ll=0):
    img_l, img_r = preprocess(img_l, img_r, level, ll, mask)
    img_l_normalized, img_r_normalized = normalize_joint([img_l, img_r])

    nn_input = torch.from_numpy(np.stack([img_l_normalized, img_r_normalized], axis=0)[None, ...]).float().cuda()
    img_nn = model(nn_input).detach().cpu().numpy()[0, 0, ...]

    return denormalize(img_nn, [img_l, img_r])


def run(model_path, gwy_path, mask=False, level=True, ll=0, manual_offset=False):
    new_gwy_path = '.'.join(gwy_path.split('.')[:-1]) + '_reconstructed.gwy'

    model = get_model(model_path)

    obj = gwyfile.load(gwy_path)
    channels = gwyfile.util.get_datafields(obj)
    img_l_orig = channels['Topo [>]'].data

    if mask:
        if manual_offset:
            img_l, img_r, mask = load_lr_img_from_gwy(gwy_path, remove_offset=False, normalize_range=False,
                                                      include_mask=True)
            img_l, img_r, mask = set_manual_offset(img_l, img_r, mask=mask)
        else:
            img_l, img_r, mask = load_lr_img_from_gwy(gwy_path, normalize_range=False, include_mask=True)

    else:
        if manual_offset:
            img_l, img_r = load_lr_img_from_gwy(gwy_path, remove_offset=False, normalize_range=False)
            img_l, img_r = set_manual_offset(img_l, img_r)
        else:
            img_l, img_r = load_lr_img_from_gwy(gwy_path, normalize_range=False)

    img_reconstructed = inference(model, img_l, img_r, mask, level, ll).astype(np.dtype('f8'))

    new_xreal = channels['Topo [>]'].xreal * (img_reconstructed.shape[1] / img_l_orig.shape[1])
    new_yreal = channels['Topo [>]'].yreal * (img_reconstructed.shape[0] / img_l_orig.shape[0])

    # offset = img_l_orig.shape[1] - img_l_offset.shape[1]
    # xoff = (offset / img_l_orig.shape[1]) * channels['Topo [<]'].xreal

    obj['/99/data/title'] = 'ResUnet reconstruction'
    obj['/99/data'] = GwyDataField(img_reconstructed, xreal=new_xreal, yreal=new_yreal, si_unit_xy=obj['/0/data'].si_unit_xy, si_unit_z=obj['/0/data'].si_unit_z)

    obj.tofile(new_gwy_path)


if __name__ == '__main__':
    args = parse_command_line()
    run(args.model_path, args.gwy_path, mask=args.mask, level=args.level, ll=args.line_by_line_level, manual_offset=args.manual_offset)






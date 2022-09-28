# coding: utf-8

__author__ = 'cleardusk'

import os.path as osp
import time
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose
import torch.backends.cudnn as cudnn

import models
from bfm import BFMModel
from utils.io import _load
from utils.functions import (
    crop_img, parse_roi_box_from_bbox, parse_roi_box_from_landmark,
)
from utils.tddfa_util import (
    load_model, _parse_param, similar_transform,
    ToTensorGjz, NormalizeGjz
)
import yaml
import pandas
make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


class TDDFA(object):
    """TDDFA: named Three-D Dense Face Alignment (TDDFA)"""

    def __init__(self, **kvs):
        torch.set_grad_enabled(False)

        # load BFM
        self.bfm = BFMModel(
            bfm_fp=kvs.get('3DDFA_bfm_fp', make_abs_path('weights/bfm_noneck_v3.pkl')),
            shape_dim=kvs.get('3DDFA_shape_dim', 40),
            exp_dim=kvs.get('3DDFA_exp_dim', 10)
        )
        self.tri = self.bfm.tri

        # config
        self.gpu_mode = kvs.get('3DDFA_faceBox_device', False)
        self.gpu_id = kvs.get('3DDFA_gpu_id', 0)
        self.size = kvs.get('3DDFA_size', 120)

        param_mean_std_fp = kvs.get(
            '3DDFA_param_mean_std_fp', make_abs_path(f'weights/param_mean_std_62d_{self.size}x{self.size}.pkl')
        )

        # load model, default output is dimension with length 62 = 12(pose) + 40(shape) +10(expression)
        model = getattr(models, kvs.get('3DDFA_arch'))(
            num_classes=kvs.get('3DDFA_num_params', 62),
            widen_factor=kvs.get('3DDFA_widen_factor', 1),
            size=self.size,
            mode=kvs.get('mode', 'small')
        )
        model = load_model(model, kvs.get('3DDFA_checkpoint_fp'))

        if self.gpu_mode:
            cudnn.benchmark = True
            model = model.cuda(device=self.gpu_id)

        self.model = model
        self.model.eval()  # eval mode, fix BN

        # data normalization
        transform_normalize = NormalizeGjz(mean=127.5, std=128)
        transform_to_tensor = ToTensorGjz()
        transform = Compose([transform_to_tensor, transform_normalize])
        self.transform = transform

        # params normalization config
        r = _load(param_mean_std_fp)
        self.param_mean = r.get('mean')
        self.param_std = r.get('std')

        # print('param_mean and param_srd', self.param_mean, self.param_std)

    def __call__(self, img_ori, objs, **kvs):
        """The main call of TDDFA, given image and box / landmark, return 3DMM params and roi_box
        :param img_ori: the input image
        :param objs: the list of box or landmarks
        :param kvs: options
        :return: param list and roi_box list
        """
        # Crop image, forward to get the param
        bs_roi_box_lst = []
        boxed_imgs = []
        deltaxy_lst=[]
        crop_policy = kvs.get('crop_policy', 'box')
        for obj in objs:# bs 拆开
            param_lst = []
            roi_box_lst = []
            for o in obj:# img中的bbox拆开
                if crop_policy == 'box':
                    # by face box
                    roi_box = parse_roi_box_from_bbox(o)
                else:
                    raise ValueError(f'Unknown crop policy {crop_policy}')
                roi_box_lst.append(roi_box)
            bs_roi_box_lst.append(roi_box_lst)
        inp_lst = torch.zeros([1, 3, 120, 120]).cuda()
        for img_num in range(len(bs_roi_box_lst)):
            temp_delta_lst = []
            for box_num in range(len(bs_roi_box_lst[img_num])):
                img,box_img,deltaxy = crop_img(img_ori[img_num], bs_roi_box_lst[img_num][box_num])
                boxed_imgs.append(box_img)
                temp_delta_lst.append(deltaxy)
                img = cv2.resize(img, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
                inp = self.transform(img).unsqueeze(0).cuda()
                inp_lst = torch.cat((inp_lst, inp), dim=0)
            deltaxy_lst.append(temp_delta_lst)
        # if self.gpu_mode:
        #     inp_lst = inp_lst.cuda(device=self.gpu_id)
        if kvs.get('timer_flag', False):
            end = time.time()
            param = self.model(inp_lst[1:])
            elapse = f'Inference: {(time.time() - end) * 1000:.1f}ms'
            print(elapse)
        else:
            param = self.model(inp_lst[1:])
        param = param.squeeze().cpu().numpy().flatten().astype(np.float32).reshape(-1,62)
            # param = param * self.param_std + self.param_mean
        for p in range(param.shape[0]):
            param[p] = param[p] * self.param_std + self.param_mean  # re-scale
            param_lst.append(param[p])
            # param_lst.append(param)
        return param_lst, bs_roi_box_lst,boxed_imgs,deltaxy_lst  # 第二个list是第一个list的分段依据

    def recon_vers(self, param_lst, roi_box_lst, **kvs):
        size = self.size

        ver_lst_dense = []
        ver_lst_sparse = []
        for param, roi_box in zip(param_lst, roi_box_lst):
            R, offset, alpha_shp, alpha_exp = _parse_param(param)
            pts3d_dense = R @ (self.bfm.u + self.bfm.w_shp @ alpha_shp + self.bfm.w_exp @ alpha_exp). \
                reshape(3, -1, order='F') + offset
            pts3d_dense = similar_transform(pts3d_dense, roi_box, size)
            pts3d_sparse = R @ (self.bfm.u_base + self.bfm.w_shp_base @ alpha_shp + self.bfm.w_exp_base @ alpha_exp). \
                reshape(3, -1, order='F') + offset
            pts3d_sparse = similar_transform(pts3d_sparse, roi_box, size)
            ver_lst_dense.append(pts3d_dense)
            ver_lst_sparse.append(pts3d_sparse)
        return ver_lst_dense,ver_lst_sparse

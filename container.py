import os
import sys
import time

import torch

import model
from model import LSTMModel, ResNet18, resnet18_at
import yaml
import joblib
from TDDFA import TDDFA
from FaceBoxes import FaceBoxes
import cv2

config = yaml.safe_load(open('./config.yaml'))


class Models(object):
    def __init__(self, train: bool = False, fold:int = 1):
        # faceboex→FAN→lstm
        self.tddfa = TDDFA(**config)  # 有transform，可能要改掉
        self.face_boxes = FaceBoxes()
        self.nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]
        self.SE = True
        self.softmax = torch.nn.Softmax(dim=1)
        self.Res_model = ResNet18()
        self.Res_model.eval()
        self.Res_chechpoint = torch.load(config['CONV_pth'])
        self.Res_model.load_state_dict(self.Res_chechpoint['model_state_dict'])
        self.Res_model.cuda()
        self.FAN = resnet18_at(at_type=config['FAN_type'])
        self.FAN.eval()
        pretrained_state_dict = torch.load(config['FAN_evey_fold_checkpoint'][fold])['state_dict']
        for key in pretrained_state_dict:
            if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):
                pass
            else:
                self.FAN.state_dict()[key.replace('module.', '')] = pretrained_state_dict[key]
        self.FAN.cuda()
        self.LSTM_model = LSTMModel(inputDim=config['LSTM_input_dim'], hiddenNum=config['LSTM_hidden_dim'],
                                    outputDim=config['LSTM_output_dim'], layerNum=config['LSTM_layerNum'],
                                    cell=config['LSTM_cell'], use_cuda=config['use_cuda'])
        if train:
            self.LSTM_model.train()
        else:
            self.LSTM_model.eval()
        self.LSTM_model.cuda()

    def forward(self, x):
        return self.LSTM_model(x)

    def eval(self):
        pass


if __name__ == '__main__':
    import cv2
    import numpy as np
    from itertools import chain
    from dataProcess import Utils
    import open3d as o3d
    # 连续显示点云
    # imgs = os.listdir('E:/cohn-kanade-images/S099/001/')
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # pcd = o3d.geometry.PointCloud()
    # vis.add_geometry(pcd)
    # for im in imgs:
    img0 = cv2.imread(f'E:/cohn-kanade-images/S099/001/dfadsffffffff')
    img1 = cv2.imread('./test1.jpg')
    # img0 = img0[200:800, 1500:2200, :]  # 两个img大小一样
    # img1 = img1[480:1080, 300:1000, :]
    # img = np.vstack((img0, img1)).reshape([-1, img0.shape[0], img0.shape[1], 3])
    img = img0.reshape(1, img0.shape[0], img0.shape[1], 3)
    t = Models(True)
    boxes = t.face_boxes(img)
    n = len(boxes)  # 可以省去输出
    if n != 1:
        print('no face detected, exit')
        sys.exit(-1)
    print(f'detect {n} faces')
    param_lst, roi_box_lst, boxed_imgs, deltaxty_lst = t.tddfa(img, boxes)
    ver_dense, ver_lst = t.tddfa.recon_vers(param_lst, list(chain(*roi_box_lst)), dense_flag=config['3DDFA_dense'])

    height, width = img.shape[1:3]
    if not type(ver_lst) in [tuple, list]:
        ver_lst = [ver_lst]
    # 通过roi_box_lst知道ver_lst画在哪个img上
    img_bs_num = len(roi_box_lst)
    pass_num = 0  # 可以提到循环前
    crop_lms = []
    for ibn in range(img_bs_num):
        face_num_per_img = len(roi_box_lst[ibn])  # 一个img有多少张脸
        ver_lst_fragment = ver_lst[pass_num:pass_num + face_num_per_img]  # 第x张脸
        # 解开注释看3dlms和点云
        # pcd = o3d.geometry.PointCloud()
        # green = np.zeros([68,3])
        # green[:,1] = 255
        # pcd.points = o3d.utility.Vector3dVector(ver_lst_fragment[0].T)
        # pcd.colors = o3d.utility.Vector3dVector(green)
        # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # pcd_dense = o3d.geometry.PointCloud()
        # h, w, _ = img[0].shape
        # ver_dense[0][0, :] = np.minimum(np.maximum(ver_dense[0][0, :], 0), w - 1)  # x
        # ver_dense[0][1, :] = np.minimum(np.maximum(ver_dense[0][1, :], 0), h - 1)  # y
        # ind = np.round(ver_dense[0]).astype(np.int32)
        # colors = img[0][:,:,::-1][ind[1, :], ind[0, :], :] / 255.  # n x 3
        # pcd_dense.points = o3d.utility.Vector3dVector(ver_dense[0].T)
        # pcd_dense.colors = o3d.utility.Vector3dVector(colors)
        # pcd_dense.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # # o3d.visualization.draw_geometries([pcd])
        # vis.add_geometry(pcd)
        # vis.poll_events()
        # vis.update_renderer()
        # vis.clear_geometries()
        crop_lms.append([ver_lst_fragment[0][0] - roi_box_lst[ibn][0][0] - deltaxty_lst[ibn][0][0],
                         ver_lst_fragment[0][1] - roi_box_lst[ibn][0][1] - deltaxty_lst[ibn][0][1]])
        # 解开注释看lms2D效果
        # for i in range(len(ver_lst_fragment)):
        #     # 得到lms,有三维※
        #     for j in range(len(ver_lst_fragment[i][0])):
        #         cv2.circle(boxed_imgs[ibn], (
        #             int(ver_lst_fragment[i][0][j] - roi_box_lst[ibn][0][0] - deltaxty_lst[ibn][0][0]),
        #             int(ver_lst_fragment[i][1][j] - roi_box_lst[ibn][0][1] - deltaxty_lst[ibn][0][1])), 1, [0, 255, 0])
        pass_num += face_num_per_img
    # cv2.imshow('img0', boxed_imgs[0] / 255)
    # cv2.imshow('img1', boxed_imgs[1] / 255)
    # cv2.waitKey(0)
    '''得到了lms，PC，boxed_img'''
    # ak上是轮流的使用，训练集是单张
    # if t.SE:
    #     single_img_lms = crop_lms[0]
    #     single_boxed_img = boxed_imgs[0]
    # else:
    #     single_img_lms = crop_lms[1]
    #     single_boxed_img = boxed_imgs[1]
    # t.SE = not t.SE
    # eye_centers = Utils.getEyesAverage(single_img_lms)  # 得到的是散装序列，若是按img，face 划分需要用face_num_per_img变量进行划分
    # aligned_boxed_img = Utils.face_align(single_boxed_img,eye_centers)
    eye_centers = Utils.getEyesAverage(crop_lms[0])  # 得到的是散装序列，若是按img，face 划分需要用face_num_per_img变量进行划分
    aligned_boxed_img = Utils.face_align(boxed_imgs[0], eye_centers,'fan')
    aligned_boxed_img = np.swapaxes(aligned_boxed_img,0,2)
    aligned_boxed_img = np.swapaxes(aligned_boxed_img,1,2)
    f,_ = t.FAN(torch.from_numpy(aligned_boxed_img).unsqueeze(0).float().cuda(),phrase = 'eval')
    print(f.shape)
    # fer2013特征提取
    # tensorSlice = torch.tensor(aligned_boxed_img, dtype=torch.float32).cuda()  # (48,48)
    # tensorSlice = torch.unsqueeze(torch.unsqueeze(tensorSlice, dim=0), dim=0)  # (1,1,48,48)
    # c0 = torch.unsqueeze(tensorSlice[0][0][0:40, 0:40], dim=0)  # (1,40,40)
    # c1 = torch.unsqueeze(tensorSlice[0][0][7:47, 0:40], dim=0)
    # c2 = torch.unsqueeze(tensorSlice[0][0][0:40, 7:47], dim=0)
    # c3 = torch.unsqueeze(tensorSlice[0][0][7:47, 7:47], dim=0)
    # cs = torch.stack([c0, c1, c2, c3], 0)  # (4,1,40,40)
    # feature = torch.mean(t.Res_model(cs / 65025), dim=0).reshape(1,
    #                                                              -1)  # 替代源码transforms.Normalize(mean=0,std=255)的效果,features = (bs,特征数)
    #
    # cs = torch.empty(0)
    pass

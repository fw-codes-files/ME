import random
import time

import numpy as np
import cv2
import os
import yaml
config = yaml.safe_load(open('./config.yaml'))
os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda_idx']
from itertools import chain
import pdb
from container import Models
import torch
import torch.utils.data as data
from utils.pose import viz_pose
from sklearn.decomposition import PCA
import joblib
##################
#global variables#
##################
target = np.array([[16, 12], [31, 12]])  # eye position in 2d image, for now, fer2013 dataset which image size is 48*48
target_fan = np.array([[75, 68], [150, 68]])  # like previous line , just image size is 224*224
EOS = np.zeros((1, config['T_proj_dim']))  # padding of origin 3d lms data sequence
AE_EOS = np.zeros((1, config['AE_mid_dim']))  # padding of AE mid feature sequence
PE_EOS = np.zeros((1, config['T_input_dim']))
standardImg = cv2.imread('./img.png')  # used to through 3DDFA net, then pointcloud produced by this picture will be a standard face pose. All face pose will be aligned to this face's pointcloud pose
softmax = torch.nn.Softmax(dim=1)
p_m = joblib.load('/home/exp-10086/Project/ME/weights/pca.m')


class LSTMDataSet(data.Dataset):
    '''
        this class has no problem, just convert data to dataset
    '''

    def __init__(self, label, concat_features, video_indexes = None, pos_embed = None):
        self.target = label
        self.input = concat_features
        self.vidx = None
        self.ps = pos_embed
        if video_indexes is not None:
            self.vidx = video_indexes
        if pos_embed is not None:
            self.ps = pos_embed
    def __getitem__(self, idx):
        if self.vidx is not None and self.ps is None:
            return self.input[idx], self.target[idx], self.vidx[idx]
        elif self.ps is not None and self.vidx is None:
            return self.input[idx], self.target[idx], self.ps[idx]
        elif (self.ps is not None) and (self.vidx is not None):
            return self.input[idx], self.target[idx], self.vidx[idx], self.ps[idx]
        else:
            return self.input[idx], self.target[idx]
    def __len__(self):
        return self.target.shape[0]


class Utils():

    def __init__(self):
        pass

    @classmethod
    def draWithOpen3d(cls, pcd_lst:list):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.concatenate(pcd_lst,axis=0))
        o3d.visualization.draw_geometries([pcd])

    @classmethod
    def getEyesAverage(cls, crop_lms):
        """
            get center coordinates per eye in 2D, used to 2d face align
        """
        crop_lms = np.array(crop_lms)
        eyel, eyer = np.mean(crop_lms[:, 36:41], axis=1), np.mean(crop_lms[:, 42:47], axis=1)
        rs = np.append(eyel, eyer)
        return rs.reshape(-1, 4).astype(np.int_)

    @classmethod
    def face_align(cls, boxed_img: np, eye_centers: np, conv_model: str = 'fan'):
        '''
            args:
                boxed_img: image cast by bounding box
                eye_centers: centers per eyes per face in 2D
                conv_model: decide the destination size
        '''
        if conv_model == 'fer':
            size = (48, 48)
            RotateMatrix, _ = cv2.estimateAffinePartial2D(eye_centers.reshape(2, 2), target)
        else:
            size = (224, 224)
            RotateMatrix, _ = cv2.estimateAffinePartial2D(eye_centers.reshape(2, 2), target_fan)
        RotImg = cv2.warpAffine(boxed_img, RotateMatrix, size, borderMode=cv2.BORDER_REPLICATE,
                                borderValue=(255, 255, 255))
        if conv_model == 'fer':
            gary = cv2.cvtColor(RotImg, cv2.COLOR_RGBA2GRAY)
            return gary
        # cv2.imshow('rotedGary', gary)
        # cv2.waitKey(0)
        return RotImg

    @classmethod
    def standardPC(cls, t):
        '''
            args:
                t: model
            [a way to make every face pointcloud in a unified posture]
        '''
        img = standardImg.reshape(1, standardImg.shape[0], standardImg.shape[1], 3)
        with torch.no_grad():
            boxes = t.face_boxes(img)  # (bs, faces)
            param_lst, roi_box_lst, _, _ = t.tddfa(img,
                                                   boxes)  # param_lst(faces(62 = 12 + 40 +10))  roi_box_lst(bs,faces) boxed_imgs(faces) deltaxty_lst(bs, faces)
            ver_dense, ver_lst = t.tddfa.recon_vers(param_lst, list(chain(*roi_box_lst)),
                                                    dense_flag=config['3DDFA_dense'])  # ver_dense(faces) ver_lst(faces)
            T = viz_pose(param_lst)
            standard_pointCloud = ver_dense[0].T
            standard_pointCloud = standard_pointCloud @ T[:3, :3].T + T[:3, 3].T
            standard_pointCloud -= standard_pointCloud[0]
            # 点云和lms是在一起的
            # pc = ver_dense[0].T
            # lms = ver_lst[0].T
            # import open3d as o3d
            # pcd = o3d.geometry.PointCloud()
            # lmsd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pc)
            # lmsd.points = o3d.utility.Vector3dVector(lms)
            # o3d.visualization.draw_geometries([pcd,lmsd])
        return standard_pointCloud

    @classmethod
    def deep_features(cls, pth, label_, t, conv_model: str = 'fan'):
        '''
            args:
                pth: image path
                label_: label of image
                t: model
                cov_model: decide the way to get deep feature of image
            return:
                np_f: image's deep feature
                ver_lst: 3D landmarks
                np_label: label_
                pc: face pointcloud
        '''
        img = cv2.imread(pth)
        img = img.reshape(1, img.shape[0], img.shape[1], 3)
        with torch.no_grad():
            boxes = t.face_boxes(img)  # (bs, faces)
            param_lst, roi_box_lst, boxed_imgs, deltaxty_lst = t.tddfa(img,boxes)  # param_lst(faces(62 = 12 + 40 +10))  roi_box_lst(bs,faces) boxed_imgs(faces) deltaxty_lst(bs, faces)
            ver_dense, ver_lst = t.tddfa.recon_vers(param_lst, list(chain(*roi_box_lst)),
                                                    dense_flag=config['3DDFA_dense'])  # ver_dense(faces) ver_lst(faces)
            if not type(ver_lst) in [tuple, list]:
                ver_lst = [ver_lst]
            img_bs_num = len(roi_box_lst)
            pass_num = 0
            for ibn in range(img_bs_num):
                crop_lms = []
                face_num_per_img = len(roi_box_lst[ibn])  # 一个img有多少张脸
                ver_lst_fragment = ver_lst[pass_num:pass_num + face_num_per_img].copy()  # 第x张脸
                crop_lms.append([ver_lst_fragment[0][0] - roi_box_lst[ibn][0][0] - deltaxty_lst[ibn][0][0],
                                 ver_lst_fragment[0][1] - roi_box_lst[ibn][0][1] - deltaxty_lst[ibn][0][1]])
                eye_centers = Utils.getEyesAverage(crop_lms[0])  # 得到的是散装序列，若是按img，face 划分需要用face_num_per_img变量进行划分
                aligned_boxed_img = Utils.face_align(boxed_imgs[0], eye_centers, conv_model)  # 和上一行的对应起来，都只取了第一张人脸
                if conv_model == 'fer':
                    tensorSlice = torch.tensor(aligned_boxed_img, dtype=torch.float32).cuda()  # (48,48)
                    tensorSlice = torch.unsqueeze(torch.unsqueeze(tensorSlice, dim=0), dim=0)  # (1,1,48,48)
                    c0 = torch.unsqueeze(tensorSlice[0][0][0:40, 0:40], dim=0)  # (1,40,40)
                    c1 = torch.unsqueeze(tensorSlice[0][0][7:47, 0:40], dim=0)
                    c2 = torch.unsqueeze(tensorSlice[0][0][0:40, 7:47], dim=0)
                    c3 = torch.unsqueeze(tensorSlice[0][0][7:47, 7:47], dim=0)
                    cs = torch.stack([c0, c1, c2, c3], 0)  # (4,1,40,40)
                    feature = torch.mean(t.Res_model(cs / 65025), dim=0).reshape(1,
                                                                                 -1)  # 替代源码transforms.Normalize(mean=0,std=255)的效果,features = (bs,特征数)
                    np_f = feature.cpu().numpy()
                else:
                    aligned_boxed_img = aligned_boxed_img[:, :, ::-1].copy()
                    aligned_boxed_img = np.swapaxes(aligned_boxed_img, 0, 2)
                    aligned_boxed_img = np.swapaxes(aligned_boxed_img, 1, 2)
                    f, alpha = t.FAN(torch.from_numpy(aligned_boxed_img).unsqueeze(0).float().cuda() / 255, phrase='eval')  # 改用拼接特征
                    # np_f = f.cpu().numpy()
                np_label = np.array([label_])
                cs = torch.empty(0)
                pass_num += face_num_per_img
                # pc和lms一同摆正
                T = viz_pose(param_lst)
                pc = ver_dense[0].T  # 这里可能有坑,索引应该怎么写？
                pc = pc @ T[:3, :3].T + T[:3, 3].T
                ver_lst[ibn] = T[:3, :3] @ ver_lst[ibn] + T[:3, 3].reshape(-1, 1)
                ver_lst[ibn] -= pc[0].T.reshape(-1, 1)
                pc -= pc[0]
                # lms和pc应该贴在一起
                # import open3d as o3d
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(np.vstack((pc,ver_lst[ibn].T)))
                # o3d.visualization.draw_geometries([pcd])
            return f, alpha, ver_lst, np_label, pc

    @classmethod
    def loadAllCKPlus(cls, pth, t):
        '''
            use all CK+ dataset to train AE model, thus, unlabeled data will be loaded too.
        '''
        img = cv2.imread(pth)
        img = img.reshape(1, img.shape[0], img.shape[1], 3)
        with torch.no_grad():
            boxes = t.face_boxes(img)  # (bs, faces)
            param_lst, roi_box_lst, boxed_imgs, deltaxty_lst = t.tddfa(img, boxes)
            ver_dense, ver_lst = t.tddfa.recon_vers(param_lst, list(chain(*roi_box_lst)), dense_flag=config['3DDFA_dense'])
            if not type(ver_lst) in [tuple, list]:
                ver_lst = [ver_lst]
            img_bs_num = len(roi_box_lst)
            for ibn in range(img_bs_num):
                # pc和lms一同摆正
                T = viz_pose(param_lst)
                pc = ver_dense[0].T  # idex=0 means just first face in a picture, maybe need to change.
                pc = pc @ T[:3, :3].T + T[:3, 3].T
                ver_lst[ibn] = T[:3, :3] @ ver_lst[ibn] + T[:3, 3].reshape(-1, 1)
                ver_lst[ibn] -= pc[0].T.reshape(-1, 1)
                pc -= pc[0]
                # lms和pc应该贴在一起
                # import open3d as o3d
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(ver_lst[ibn].T)
                # o3d.visualization.draw_geometries([pcd])
        return ver_lst[ibn], pc

    @classmethod
    def insertMethod(cls, ori_data, delta, inverse: bool = False):
        '''
            video which contains less than sequence length is going to filled with EOS symbol
        '''
        new_data = np.zeros((0, ori_data.shape[1]))
        cpoy0 = int(delta // ori_data.shape[0]) + 1
        cpoy1 = int(delta % ori_data.shape[0])
        integration = np.tile(ori_data, (cpoy0, 1))
        for i in range(ori_data.shape[0]):
            for cp in range(cpoy0):
                new_data = np.concatenate((new_data, integration[i + ori_data.shape[0] * cp].reshape(1, -1)), axis=0)
        fraction = np.tile(ori_data[:cpoy1], (1,))
        for c in range(fraction.shape[0]):
            new_data = np.insert(new_data, c * (cpoy0 + 1), fraction[c].reshape(1, -1), axis=0).astype(np.float32)
        return new_data

    @classmethod
    def sampleMethod(cls, ori_data, variable_step: bool = False):
        '''
        video which contains frames more than sequence length is going to be sampled
        '''
        new_data = np.zeros((0, ori_data.shape[1]))
        sample_range = ori_data.shape[0]
        sam_idx = sorted(random.sample([S for S in range(sample_range)], config['Max_frame'] - 3))
        for s in sam_idx:
            new_data = np.concatenate((new_data, ori_data[s].reshape(1, -1)), axis=0).astype(np.float32)
        return new_data

    @classmethod
    def insertOrSample(cls, split: np, data_np: np):
        '''
        args：
            spilt: 一个视频原有的帧数 ex:np (294,)
            data_lst: 数据集，利用spilt划分 ex:label:(5231,) feature:(5231,512) lms3d:(5231,204)
        return：
            new_spilt: 插值或者采样得到的新帧数list
            new_data_lst: 插值或者采样得到的新数据集
        '''
        start_idx = 0
        new_spilt = []
        new_data_lst = []
        for s in range(split.shape[0]):
            if split[s] < config['Least_frame']:
                # 插值，小于Least_frame帧的都拉长放在new_data_lst
                new_spilt.append(config['Least_frame'])  # 必定是Least_frame
                ori_data = data_np[start_idx:start_idx + int(split[s])].copy()
                new_data_lst.append(Utils.insertMethod(ori_data, config['Least_frame'] - split[s]))
                start_idx += int(split[s])
            elif config['Least_frame'] <= split[s] <= config['Max_frame']:
                new_spilt.append(split[s])
                new_data_lst.append(data_np[start_idx:start_idx + int(split[s])])
                start_idx += int(split[s])
            else:
                # 采样,大于40帧的都隔n帧放在new_data_lst
                new_spilt.append(config['Max_frame'])
                ori_data = data_np[start_idx:start_idx + int(split[s])].copy()
                new_data_lst.append(Utils.sampleMethod(ori_data))
                start_idx += int(split[s])
        return np.array(new_spilt, dtype=object), np.array(new_data_lst, dtype=object)

    @classmethod
    def insertOrSampleWOPeak(cls, split: np, data_np: np):
        '''
        args：
            spilt: 一个视频原有的帧数 ex:np (294,)
            data_lst: 数据集，利用spilt划分 ex:label:(5231,) feature:(5231,512) lms3d:(5231,204)
        return：
            new_spilt: 插值或者采样得到的新帧数list
            new_data_lst: 插值或者采样得到的新数据集
        '''
        start_idx = 0
        new_spilt = []
        new_data_lst = []
        for s in range(split.shape[0]):
            if split[s] < config['Least_frame']:
                # 插值，小于Least_frame帧的都拉长放在new_data_lst
                new_spilt.append(config['Least_frame'])  # 必定是Least_frame
                ori_data = data_np[start_idx:start_idx + int(split[s])].copy()
                inserted = Utils.insertMethod(ori_data[:-3], config['Least_frame'] - split[s] + 3)
                fixed_peak = np.concatenate((inserted,ori_data[-3:]),axis=0)
                new_data_lst.append(fixed_peak)
                start_idx += int(split[s])
            elif config['Least_frame'] <= split[s] <= config['Max_frame']:
                new_spilt.append(split[s])
                new_data_lst.append(data_np[start_idx:start_idx + int(split[s])])
                start_idx += int(split[s])
            else:
                # 采样,大于40帧的都隔n帧放在new_data_lst
                new_spilt.append(config['Max_frame'])
                ori_data = data_np[start_idx:start_idx + int(split[s])].copy()
                sampled = Utils.sampleMethod(ori_data[:-3])
                fixed_peak = np.concatenate((sampled,ori_data[-3:]),axis=0)
                new_data_lst.append(fixed_peak)
                start_idx += int(split[s])
        return np.array(new_spilt, dtype=object), np.array(new_data_lst, dtype=object)

    @classmethod
    def atomicityInsert(cls, ori_data):
        new_data = np.zeros((0, ori_data.shape[1]))
        # cpoy0 = int(config['Least_frame'] // ori_data.shape[0])
        cpoy0 = 3
        integration = np.tile(ori_data, (cpoy0, 1))
        for i in range(ori_data.shape[0]):
            for cp in range(cpoy0):
                new_data = np.concatenate((new_data, integration[i + ori_data.shape[0] * cp].reshape(1, -1)), axis=0)
        return new_data

    @classmethod
    def inserCompeletely(cls, split: np, data_np: np):
        '''
        不会再进行采样，只进行插值操作,且插值的视频具有原子性。all videos shorter than 20 frames will be atomicity inserted
        args:
            split: videos length
            data: lms+rgb
        '''
        start_idx = 0
        new_spilt = []
        new_data_lst = []
        for s in range(split.shape[0]):
            # if split[s] < config['Least_frame']:
            # 插值，小于least frame帧的都拉长放在new_data_lst
            ori_data = data_np[start_idx:start_idx + int(split[s])].copy()
            inserted = Utils.atomicityInsert(ori_data)
            new_data_lst.append(inserted)
            new_spilt.append(inserted.shape[0])
            start_idx += int(split[s])
            # else:
            #     new_spilt.append(split[s])
            #     new_data_lst.append(data_np[start_idx:start_idx + int(split[s])])
            #     start_idx += int(split[s])
        return np.array(new_spilt, dtype=object), np.array(new_data_lst, dtype=object)

    @classmethod
    def aggregateFeaAndCode(cls, fi_lst, alphi_lst, spllit_lst, test_fold, t):
        fi_lst = torch.cat(fi_lst, dim=0)  # all fi (all train frames,512)
        alphi_lst = torch.cat(alphi_lst, dim=0)  # (all train frames,1)
        alphifi_lst = fi_lst.mul(alphi_lst)  # all α*fi (all train frames,512)
        v_idx = 0
        video_f_lst = []
        for sptr in spllit_lst:
            video_f_lst.append(
                torch.sum(alphifi_lst[v_idx:v_idx + sptr], dim=0) / torch.sum(alphi_lst[v_idx:v_idx + sptr],
                                                                              dim=0).reshape(1, -1))
            v_idx += sptr
        video_f_lst = torch.cat(video_f_lst, dim=0)  # (all videos, 512)
        v_idx = 0
        for idx, sptr in enumerate(spllit_lst):
            with open(f'./dataset/fan_feature_fold{test_fold}_{t}.txt', 'ab') as nf:
                # video_sized_copy_feature = torch.tile(video_f_lst[idx], (sptr, 1))
                # concatnated_f = torch.cat((fi_lst[v_idx:v_idx + sptr], video_sized_copy_feature), dim=1)
                # np_f = concatnated_f.cpu().numpy()
                # np.savetxt(nf, np_f)
                pass

    @classmethod
    def open3dVerify(cls, ver_lst, pc, sPC):
        import open3d as o3d
        pcd_s = o3d.geometry.PointCloud()
        pcd_pc = o3d.geometry.PointCloud()
        if type(ver_lst) == list:
            pcd_s.points = o3d.utility.Vector3dVector(ver_lst[0].T)
        else:
            pcd_s.points = o3d.utility.Vector3dVector(ver_lst.T)
        pcd_pc.points = o3d.utility.Vector3dVector(np.vstack((pc, sPC)))
        o3d.visualization.draw_geometries([pcd_s, pcd_pc])

    @classmethod
    def injectNoiseInput(cls, input, percent):
        '''
            args:
                percent: how many percent of input will be injected noise
            return:
                noisy input
        '''
        length = input.shape[0]
        noi_lst = random.sample([i for i in range(length)], int(length * percent))
        for n in noi_lst:
            raw_lms_frame = input[n].copy()
            how_many_point = np.random.randint(1, high=69)
            lms_idx = random.sample([i for i in range(68)], how_many_point)
            for l in lms_idx:
                value_times = random.sample([10, 100, 1000], 1)
                noi_xyz = np.random.random((1, 3)) * value_times
                raw_lms_frame[l] = noi_xyz
            input[n] = raw_lms_frame.copy()
        return input

    @classmethod
    def solveICPBySVD(cls, p0, p1):
        """svd求解3d-3d的Rt, $p_0 = Rp_1 + t$
        Args:
          p0 (np.ndarray): nx3
          p1 (np.ndarray): nx3
        Returns:
          R (np.ndarray): 3x3
          t (np.ndarray): 3x1
        """
        p0c = p0.mean(axis=0, keepdims=True)  # 质心
        p1c = p1.mean(axis=0, keepdims=True)  # 1, 3
        q0 = p0 - p0c
        q1 = p1 - p1c  # n x 3
        W = q0.T @ q1  # 3x3
        U, _, VT = np.linalg.svd(W)
        R = U @ VT  # $UV^T$
        t = p0c.T - R @ p1c.T
        return R, t

    @classmethod
    def vote(cls, pred_collection, group_notes, label):
        '''
            args:
                pred_collection: origin prediction of model. list[tensor(8,7)]
                group_notes: how many samples belongs to a video. list[int]
                label: ground truth.list[long]
                addOrcount: way of voting, 0 means by adding, 1 means by counting
            return:
                acc: original video accuracy
        '''
        acc = 0
        pred_collection = torch.cat(pred_collection) # tensor (samples,7)
        group_notes = torch.cat(group_notes)
        label = torch.cat(label)
        max_index = torch.max(group_notes)
        denominator = int(max_index.item()+1)
        for m in range(denominator):
            m_m = group_notes==m
            pred_collection_m = pred_collection[m_m] # (sample number, 7)
            label_m = label[m_m] # pick data by video index
            pred_cls = torch.topk(pred_collection_m, 1, dim=1)[1] # index of classification prediction
            table = torch.zeros((1,7)).cuda()
            for pc in pred_cls:
                table[0,pc]+=1
            most_pre = torch.topk(table, 1, dim=1)[0]
            a = table == most_pre # check whether prediction is sole
            pre_conf, final_pre = None, None
            if torch.sum(a.float()) > 1: # if prediction is not sole
                indx = torch.nonzero(a).cuda()
                for i in indx:
                    vec = torch.zeros((1, 7)).cuda()
                    for sidx, sb in enumerate(pred_cls):
                        if i[1] == sb[0]:
                            vec += pred_collection_m[sidx]
                    if pre_conf is None:
                        pre_conf = vec[0, i[1]]
                        final_pre = i[1]
                    else:
                        if pre_conf < vec[0, i[1]]:
                            pre_conf = vec[0, i[1]]
                            final_pre = i[1]
                        else:
                            pass
            else: # if prediction is sole
                final_pre = torch.topk(table, 1, dim=1)[1]
            final_pre_c = final_pre.cuda()
            if label_m.shape[0] == 0:
                denominator -= 1
            else:
                if final_pre_c == label_m[0]:
                    acc += 1
        return acc / denominator

    @classmethod
    def loadKeys(cls, model, static_dict):
        for key in model.state_dict().keys():
            if key == 'pred.weight' or key == 'pred.bias':
                continue
            model.state_dict()[key] = static_dict[key]
        return model

    @classmethod
    def SinusoidalEncoding(cls, seq_len, d_model):
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            for pos in range(1, seq_len + 1)])
        pos_table[:, 0::2] = np.sin(pos_table[:, 0::2])
        pos_table[:, 1::2] = np.cos(pos_table[:, 1::2])
        return pos_table.astype(np.float32)

    @classmethod
    def ConstantSpeed(cls, ori_video_len, window_size=[6, 6], span_lst=[0, 1, 2]):
        '''按着空格数量来取得index,想要sample较多的样本，既然采样匀速，就可以从长度来进行。
        1.视频长短不一，sequence length最大长度不可以超过max长度
        2.单个视频为处理单元。'''
        frameIndex_redundancy_matrix = np.zeros((0, ori_video_len))  # bool mat,used for indexing
        for s in span_lst:
            tem_sam_idx = ori_video_len - np.arange(ori_video_len) * (s + 1) - 1  # last frame obtained certainly
            # sequence length <=30?? 如果在max以内，会出现负的索引。把正的索引号indexing到正确的索引位。
            # eg.[5,3,1]->[1,0,1,0,1,0]
            tem_sam_bool_idx = np.zeros((1, ori_video_len))
            tem_sam_idx[tem_sam_idx < 0] = 0
            tem_sam_bool_idx[:, tem_sam_idx] = 1  # 这一刻已经是正序了
            if sum(tem_sam_bool_idx[0, :3]) != 3:  # 防止index0处的数值异常
                tem_sam_bool_idx[0, 0] = 0
            if sum(sum(tem_sam_bool_idx)) > window_size[1]:
                for c in range(ori_video_len // 3):
                    tem_sam_bool_idx[0, c * 3:(c + 1) * 3] = 0
                    # if sum(sum(tem_sam_bool_idx))<window_size[0]:
                    if sum(sum(tem_sam_bool_idx)) <= window_size[0]:
                        frameIndex_redundancy_matrix = np.concatenate(
                            (frameIndex_redundancy_matrix, tem_sam_bool_idx.reshape(1, -1)), axis=0)
                        break
                    else:
                        if sum(sum(tem_sam_bool_idx)) > window_size[1]:
                            continue
                        else:
                            frameIndex_redundancy_matrix = np.concatenate(
                                (frameIndex_redundancy_matrix, tem_sam_bool_idx.reshape(1, -1)), axis=0)
            else:
                frameIndex_redundancy_matrix = np.concatenate(
                    (frameIndex_redundancy_matrix, tem_sam_bool_idx.reshape(1, -1)), axis=0)
        return frameIndex_redundancy_matrix.astype('bool')
class Dataprocess():
    def __init__(self):
        pass

    @classmethod
    def deleleDS_Store(cls):
        '''
        delete CK+ dataset folder contains .DS_Store files which are useless
        '''
        S_lst = os.listdir('E:/cohn-kanade-images/')
        for s in S_lst:
            S_pth = os.path.join('E:/cohn-kanade-images/', s)
            file = os.listdir(S_pth)
            for f in file:
                if f.endswith('DS_Store'):
                    os.remove(os.path.join(S_pth, f))
                img_pth = os.path.join(S_pth, f)
                img = os.listdir(img_pth)
                for i in img:
                    if i.endswith('DS_Store'):
                        os.remove(os.path.join(img_pth, i))

    @classmethod
    def loadCKPlusData(cls, test_fold):
        import tqdm
        '''
        args: 
            test_fold: which fold to be test dataset
        '''
        file = open(config['Dataset_10fold'], 'r', encoding='utf-8')
        data_directory = file.readlines()
        folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        folds.remove(test_fold)
        conformity_lst = []
        test_conformity_lst = []
        train_lst = []
        test_lst = []
        split_train = []
        split_test = []
        for index, item in enumerate(data_directory):  # 0, '1-fold\t31\n' in {[0, '1-fold\t31\n'], [1, 'S037/006 Happy\n'], ...}分化两个数据集,txt的操作
            test_fold_str = str(test_fold) + '-fold'
            if test_fold_str in item:
                for k in range(index + 1, index + int(item.split()[1]) + 1):  # 测试集
                    test_conformity_lst.append(data_directory[k])
            for i in folds:
                fold_str = str(i) + '-fold'  # 1-fold
                if fold_str in item:  # 1-fold in '1-fold\t31\n'
                    for j in range(index + 1, index + int(item.split()[1]) + 1):  # (0 + 1, 0 + 31 + 1 ) 训练集
                        conformity_lst.append(data_directory[j])  # imf[2] = 'S042/006 Happy\n'
        # print(len(test_conformity_lst))
        for line in conformity_lst:  # 文件夹的操作
            video_label = line.strip().split()
            video_name = video_label[0]  # name of video
            try:
                label = config['CK+_dict'][video_label[1]]  # label of video
            except:
                pdb.set_trace()
            video_path = os.path.join(config['CK+_data_root'], video_name)  # video_path is the path of each video
            img_lists = os.listdir(video_path)
            img_lists.sort()  # sort files by ascending
            img_lists = img_lists[- int(round(len(img_lists))):]
            frames = len(img_lists)
            for frame in img_lists:
                train_lst.append((os.path.join(video_path, frame), label, frames))
            split_train.append(frames)

        for line in test_conformity_lst:
            video_label = line.strip().split()
            video_name = video_label[0]  # name of video
            try:
                label = config['CK+_dict'][video_label[1]]  # label of video
            except:
                pdb.set_trace()
            video_path = os.path.join(config['CK+_data_root'], video_name)  # video_path is the path of each video
            img_lists = os.listdir(video_path)
            img_lists.sort()  # sort files by ascending
            img_lists = img_lists[- int(round(len(img_lists))):]
            frames = len(img_lists)
            for frame in img_lists:
                test_lst.append((os.path.join(video_path, frame), label, frames))
            split_test.append(frames)
        # print(len(test_lst),len(train_lst))
        # 根据插值或者采样的结果 得到深度特征,3dlms,label
        t = Models(False, test_fold)
        sPC = Utils.standardPC(t) # standard PointCloud
        random_idx = random.sample([i for i in range(config['PC_points_sample_range'])], config['PC_points_piars'])
        alphi_lst = []
        fi_lst = []

        for pth, label_, fms_number in tqdm.tqdm(train_lst):
            fi, alphi, ver_lst, np_label, pc = Utils.deep_features(pth, label_, t, 'fan')
            fi_lst.append(fi) # deep feature
            alphi_lst.append(alphi) # alpha i
            standard_idx = sPC[random_idx]
            standard_distance = np.linalg.norm(
                standard_idx[:int(config['PC_points_piars'] / 2)] - standard_idx[int(config['PC_points_piars'] / 2):],
                axis=1)
            vari_idx = pc[random_idx]
            vari_distance = np.linalg.norm(
                vari_idx[:int(config['PC_points_piars'] / 2)] - vari_idx[int(config['PC_points_piars'] / 2):], axis=1)
            rates = standard_distance / vari_distance
            rate = np.median(rates) # scale value between new pc and standard pc
            ver_lst[0] *= rate
            pc *= rate
            r, Tt = Utils.solveICPBySVD(p0=sPC, p1=pc)
            newpc = r @ pc.T + Tt
            newlms = r @ ver_lst[0] + Tt
            # lms pc standardPC 都在一起
            # Utils.open3dVerify(ver_lst,pc,sPC)
            # with open(f'./dataset/3dlms_fold{test_fold}_train.txt', 'ab') as n3d:
            #     np.savetxt(n3d, newlms.T)
            # with open(f'./dataset/label_fold{test_fold}_train.txt', 'ab') as nl:
            #     np.savetxt(nl, np_label)
        Utils.aggregateFeaAndCode(fi_lst, alphi_lst, split_train, test_fold, 'train')
        alphi_lst = []
        fi_lst = []
        for pth, label_, fms_number in tqdm.tqdm(test_lst):
            fi, alphi, ver_lst, np_label, pc = Utils.deep_features(pth, label_, t, 'fan')
            fi_lst.append(fi)
            alphi_lst.append(alphi)
            standard_idx = sPC[random_idx]
            standard_distance = np.linalg.norm(
                standard_idx[:int(config['PC_points_piars'] / 2)] - standard_idx[int(config['PC_points_piars'] / 2):],
                axis=1)
            vari_idx = pc[random_idx]
            vari_distance = np.linalg.norm(
                vari_idx[:int(config['PC_points_piars'] / 2)] - vari_idx[int(config['PC_points_piars'] / 2):], axis=1)
            rates = standard_distance / vari_distance
            rate = np.median(rates)
            ver_lst[0] *= rate
            pc *= rate
            r, Tt = Utils.solveICPBySVD(p0=sPC, p1=pc)
            newpc = r @ pc.T + Tt
            newlms = r @ ver_lst[0] + Tt
            # with open(f'./dataset/3dlms_fold{test_fold}_test.txt', 'ab') as n3d:
            #     np.savetxt(n3d, newlms.T)
            # with open(f'./dataset/label_fold{test_fold}_test.txt', 'ab') as nl:
            #     np.savetxt(nl, np_label)
        Utils.aggregateFeaAndCode(fi_lst, alphi_lst, split_test, test_fold, 'test')
        for fms in tqdm.tqdm(split_train):
            with open(f'./dataset/split_{test_fold}_train.txt', 'ab') as nfn:
                # np.savetxt(nfn, np.array(fms).reshape([1, -1]))
                pass
        for fms in tqdm.tqdm(split_test):
            with open(f'./dataset/split_{test_fold}_test.txt', 'ab') as nfn:
                # np.savetxt(nfn, np.array(fms).reshape([1, -1]))
                pass

    @classmethod
    def dataForLSTM(cls, fold, crop: bool = False):
        '''
            args:
                fold: which fold dataset to generate
                crop: whether use fixed window size
            return:
                data for dataset
        '''
        label_train = np.loadtxt(f'./dataset/label_fold{fold}_train.txt').reshape(-1, 1)  # 每一帧的label
        train = np.loadtxt(f'./dataset/fan_feature_fold{fold}_train.txt')  # 每一帧的feature
        lms3d_train = np.loadtxt(f'./dataset/3dlms_fold{fold}_train.txt')  # 每一帧的lms
        split_train = np.loadtxt(f'./dataset/split_{fold}_train.txt')  # 如何分帧

        label_test = np.loadtxt(f'./dataset/label_fold{fold}_test.txt').reshape(-1, 1)
        test = np.loadtxt(f'./dataset/fan_feature_fold{fold}_test.txt')
        lms3d_test = np.loadtxt(f'./dataset/3dlms_fold{fold}_test.txt')
        split_test = np.loadtxt(f'./dataset/split_{fold}_test.txt')
        if crop:
            train_seqs, train_l = Utils.insertOrSample(split_train.astype(np.int_), label_train.astype(np.float32))
            _, train_feature = Utils.insertOrSample(split_train.astype(np.int_), train.astype(np.float32))
            _, train_lms3d = Utils.insertOrSample(split_train.astype(np.int_),
                                                  lms3d_train.astype(np.float32).reshape(-1, 204))

            test_seqs, test_l = Utils.insertOrSample(split_test.astype(np.int_), label_test.astype(np.float32))
            _, test_feature = Utils.insertOrSample(split_test.astype(np.int_), test.astype(np.float32))
            _, test_lms3d = Utils.insertOrSample(split_test.astype(np.int_),
                                                 lms3d_test.astype(np.float32).reshape(-1, 204))

            return train_l, train_feature, train_lms3d, train_seqs, test_l, test_feature, test_lms3d, test_seqs
        else:
            return label_train, train, lms3d_train, split_train, label_test, test, lms3d_test, split_test

    @classmethod
    def pca(cls, ori_data, seqs, flag):
        '''
            args:
                data: data to transform and fit
                dim: output's dimension
                flag: if TURE means train PCA model and return data,if FALSE means use trained PCA model and return data
            return:
                transformed data
        '''
        # squeeze data
        data = np.concatenate(ori_data, axis=0)
        # Dimension reduction
        if flag:
            p_model = PCA(n_components=int(config['PCA_dim']))
            pca_data = p_model.fit_transform(data)
            joblib.dump(p_model, config['PCA_dir'])
        else:
            p_model = joblib.load(config['PCA_dir'])
            pca_data = p_model.transform(data)
        # split data again
        idx = 0
        pca_lst = []
        for s in seqs:
            pca_lst.append(pca_data[idx:idx + s])
            idx += s
        return np.array(pca_lst, dtype=object)

    @classmethod
    def dataAlign2WindowSize(cls, ws, feature, lms3d, label, use_AE: bool = False, vote:bool=False, step = 1):
        '''
            args:
                ws: window size, also known as sequence length
                feature: deep feature from imgages
                lms3d: 3D landmarks
                label: target data
                step: sampling frequency
            return:
                dataloader
        '''
        data = []
        target = []
        samples_counter_lst = []
        '''normalize image features and 3d lms, then concatenate them, but sequence length is fixed, 
            so either break a sample to slices, either concatenate EOS to window size.'''
        for f in range(0, feature.shape[0]):  # video level
            '''this is normalization,  However,3d lms will be a circle in space, 
                it seems lost much information, so stop normalization for now.'''
            # f_mean = np.mean(feature[f][:,:512], axis=1)
            # f_std = np.std(feature[f][:,:512], axis=1)
            # feature[f][:,:512] = (feature[f][:,:512] - f_mean.reshape(-1, 1)) / f_std.reshape(-1, 1)

            # l_mean = np.mean(lms3d[f], axis=1)
            # l_std = np.std(lms3d[f], axis=1)
            # lms3d[f] = (lms3d[f] - l_mean.reshape(-1, 1)) / l_std.reshape(-1, 1)

            # concatnate and align to ws -- video level
            # ori_video = np.hstack((feature[f][:,:512], lms3d[f]))
            ori_video = feature[f][:,:512]
            if ori_video.shape[0] < ws:
                # EOS
                EOS_num = ws - ori_video.shape[0]
                if use_AE:
                    blanks = np.tile(AE_EOS, (EOS_num, 1))
                else:
                    blanks = np.tile(EOS, (EOS_num, 1))
                video = np.concatenate((ori_video, blanks), axis=0)
                data.append(video)
                target.append(label[f][0][0])
                if vote:
                    samples_counter_lst.append(f)
            else:
                #  one video generates more shape[0] - ws samples
                for w in range(ori_video.shape[0] - ws + 1):
                    row_rand = sorted(random.sample(range(w, ori_video.shape[0]), config['window_size']))
                    sample = ori_video[row_rand] # pick up randomly
                    data.append(sample)
                    target.append(label[f][0][0].astype(np.long))
                    if vote:
                        samples_counter_lst.append(f)
        # dataset and dataloader
        if vote:
            dataset = LSTMDataSet(torch.from_numpy(np.array(target, dtype=np.float32)).cuda(),
                                  torch.from_numpy(np.array(data, dtype=np.float32)).cuda(),
                                  torch.from_numpy(np.array(samples_counter_lst, dtype=np.float32)).cuda())
        else:
            dataset = LSTMDataSet(torch.from_numpy(np.array(target, dtype=np.float32)).cuda(),
                                  torch.from_numpy(np.array(data, dtype=np.float32)).cuda())
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=config['Shuffle'], batch_size=config['batch_size'])

        return dataloader

    @classmethod
    def ConvertVideo2Samples(cls, ws, feature, lms3d, label, vote:bool = True):
        '''
            args:
                ws: window size, also known as sequence length
                feature: deep feature from imgages
                lms3d: 3D landmarks
                label: target data
            return:
                dataloader
        '''
        data = []
        target = []
        samples_counter_lst = []
        for f in range(0, feature.shape[0]):  # video level
            # f_mean = np.mean(feature[f][:,:512], axis=1)
            # f_std = np.std(feature[f][:,:512], axis=1)
            # feature[f][:,:512] = (feature[f][:,:512] - f_mean.reshape(-1, 1)) / f_std.reshape(-1, 1)

            l_mean = np.mean(lms3d[f], axis=1)
            l_std = np.std(lms3d[f], axis=1)
            lms3d[f] = (lms3d[f] - l_mean.reshape(-1, 1)) / l_std.reshape(-1, 1)

            # concatnate and align to ws -- video level
            # ori_video = np.hstack((feature[f][:,:512], lms3d[f]))
            ori_video = np.hstack((lms3d[f], feature[f][:,:512]))
            # ori_video = lms3d[f]
            if ori_video.shape[0] < ws:
                # EOS
                EOS_num = ws - ori_video.shape[0]
                blanks = np.tile(EOS, (EOS_num, 1))
                video = np.concatenate((ori_video, blanks), axis=0)
                data.append(video)
                target.append(label[f][0][0])
                if vote:
                    samples_counter_lst.append(f)
            else:
                # one video can generate any number samples
                v_fs = np.arange(ori_video.shape[0]) # how many frames a video contains (frames,)
                v_ix = np.zeros((v_fs.shape[0],)) # what frames will be sampled (frames,)
                redundancy_matrix = np.zeros((config['standBy'], config['window_size'])) # a matrix stand for redundancy and sequence length, (how many standby samples, sequence length)
                for i in range(config['standBy']):
                    v_fs_shuffle = v_fs.copy()  # a cpoy for shuffle
                    iind = random.sample([xx for xx in range(ori_video.shape[0]//2)],1)
                    temp = v_fs_shuffle[iind[0]:min(ori_video.shape[0]//5*3+iind[0],ori_video.shape[0])].copy()
                    np.random.shuffle(temp) # every standby sample will be generated from origin video shuffled again (frames,)
                    v_fs_shuffle[iind[0]:min(ori_video.shape[0]//5*3+iind[0],ori_video.shape[0])] = temp.copy()
                    v_ix_copy = v_ix.copy() # index copy
                    v_ix_copy[v_fs_shuffle[iind[0]:config['window_size'] + iind[0]]] = 1 # code in inner bracket means a slicing operation on variable v_fs_shuffle, code in outer bracket means a mask operation.
                    redundancy_matrix[i] = v_fs[v_ix_copy.astype(int).astype(bool)] # assignment
                span = (redundancy_matrix.max(axis=1) - redundancy_matrix.min(axis=1))>= ori_video.shape[0]//2*1 # calulate the span of every standby sample, greater than some value will be selected as input sequence
                if np.sum(span!=0)>=config['selected']: # number of qualified sample might be greater than we want, so if true, just select top 20 samples.
                    samples = redundancy_matrix[span,:][:config['selected']]
                else: # if not, take as many as possible
                    samples = redundancy_matrix[:config['selected']]
                for w in range(samples.shape[0]):
                    data.append(ori_video[list(samples[w].astype(int))])
                    target.append(label[f][0][0].astype(np.long))
                    if vote:
                        samples_counter_lst.append(f)
        # dataset and dataloader
        if vote:
            dataset = LSTMDataSet(torch.from_numpy(np.array(target, dtype=np.float32)).cuda(),
                                  torch.from_numpy(np.array(data, dtype=np.float32)).cuda(),
                                  torch.from_numpy(np.array(samples_counter_lst, dtype=np.float32)).cuda())
        else:
            dataset = LSTMDataSet(torch.from_numpy(np.array(target, dtype=np.float32)).cuda(),
                                  torch.from_numpy(np.array(data, dtype=np.float32)).cuda())
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=config['Shuffle'], batch_size=config['batch_size'])
        return dataloader

    @classmethod
    def ConvertVideo2SamlpesConstantSpeed(cls, ws, feature, lms3d, label, vote: bool = True, pos_embed=None):
        '''
            args:
                ws: window size, also known as sequence length
                feature: deep feature from imgages
                lms3d: 3D landmarks
                label: target data
                vote: count voting
                pos_embed: origin video's frames's cosine-sine position embedding, peak frame's position is fixed.
            return:
                dataloader
        '''
        data = []
        target = []
        samples_counter_lst = []
        pos_embeds = []
        for f in range(0, feature.shape[0]):  # video level
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            # f_mean = np.mean(feature[f][:,:512], axis=1)
            # f_std = np.std(feature[f][:,:512], axis=1)
            # feature[f][:,:512] = (feature[f][:,:512] - f_mean.reshape(-1, 1)) / f_std.reshape(-1, 1)

            # l_mean = np.mean(lms3d[f], axis=1)
            # l_std = np.std(lms3d[f], axis=1)
            # lms3d[f] = (lms3d[f] - l_mean.reshape(-1, 1)) / l_std.reshape(-1, 1)

            # concatnate and align to ws -- video level
            ori_video = feature[f]
            # ori_video = np.hstack((lms3d[f], feature[f][:, :512]))  # 340
            # ori_video = lms3d[f]
            samples_idx = Utils.ConstantSpeed(ori_video.shape[0])  # (sample num, length）
            for w in range(samples_idx.shape[0]):
                v_s = ori_video[samples_idx[w]]  # 采样的样本
                if v_s.shape[0] < ws:
                    EOS_num = ws - v_s.shape[0]
                    blanks = np.tile(EOS, (EOS_num, 1))
                    processed_v_s = np.concatenate((v_s, blanks), axis=0)
                else:
                    processed_v_s = v_s.copy()
                if w == 2:
                    for pttth in range(6):
                        pcd.points = o3d.utility.Vector3dVector(v_s[pttth][:204].copy().reshape((68, 3)))
                        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                        o3d.visualization.draw_geometries([pcd])
                data.append(processed_v_s)
                target.append(label[f][0][0].astype(np.long))
                if vote:
                    samples_counter_lst.append(f)
                if pos_embed is not None:
                    pe_p = pos_embed[f][samples_idx[w]]
                    if pe_p.shape[0] < ws:
                        EOS_num = ws - pe_p.shape[0]
                        blanks = np.tile(PE_EOS, (EOS_num, 1))
                        processed_pe = np.concatenate((pe_p, blanks), axis=0)
                    else:
                        processed_pe = pe_p.copy()
                    pos_embeds.append(processed_pe)
        # dataset and dataloader !!! x,y 是必须的，有无voting和有无fixed peak frame position变成了4个情况。
        if vote:  # 用于test
            if pos_embed is not None:  # 用于peak frame fixed position encoding
                dataset = LSTMDataSet(torch.from_numpy(np.array(target, dtype=np.float32)).cuda(),
                                      torch.from_numpy(np.array(data, dtype=np.float32)).cuda(),
                                      torch.from_numpy(np.array(samples_counter_lst, dtype=np.float32)).cuda(),
                                      torch.from_numpy(np.array(pos_embeds, dtype=np.float32)).cuda())
            else:
                dataset = LSTMDataSet(torch.from_numpy(np.array(target, dtype=np.float32)).cuda(),
                                      torch.from_numpy(np.array(data, dtype=np.float32)).cuda(),
                                      torch.from_numpy(np.array(samples_counter_lst, dtype=np.float32)).cuda())
        else:
            if pos_embed is not None:
                dataset = LSTMDataSet(torch.from_numpy(np.array(target, dtype=np.float32)).cuda(),
                                      torch.from_numpy(np.array(data, dtype=np.float32)).cuda(),
                                      None,
                                      torch.from_numpy(np.array(pos_embeds, dtype=np.float32)).cuda())
            else:
                dataset = LSTMDataSet(torch.from_numpy(np.array(target, dtype=np.float32)).cuda(),
                                      torch.from_numpy(np.array(data, dtype=np.float32)).cuda())
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=config['Shuffle'], batch_size=config['batch_size'])

    @classmethod
    def AEinput(cls, path):
        '''
            load all CK+ dataset
            args:
                path: ck+ picture
        '''
        rs_total = np.zeros((0, 3))
        t = Models(False, 1)  # whatever second parameter is, just use 3DDFA, for now.
        sPC = Utils.standardPC(t)
        random_idx = random.sample([i for i in range(config['PC_points_sample_range'])], config['PC_points_piars'])
        s_dir_lst = os.listdir(path)
        for s_dir in s_dir_lst:
            num_lst = os.listdir(os.path.join(path, s_dir))
            for n in num_lst:
                pic_lst = os.listdir(os.path.join(path, s_dir, n))
                for p in pic_lst:
                    img_pth = os.path.join(path, s_dir, n, p)
                    rs, pc = Utils.loadAllCKPlus(img_pth, t)
                    standard_idx = sPC[random_idx]
                    standard_distance = np.linalg.norm(
                        standard_idx[:int(config['PC_points_piars'] / 2)] - standard_idx[
                                                                            int(config['PC_points_piars'] / 2):],
                        axis=1)
                    vari_idx = pc[random_idx]
                    vari_distance = np.linalg.norm(
                        vari_idx[:int(config['PC_points_piars'] / 2)] - vari_idx[int(config['PC_points_piars'] / 2):],
                        axis=1)
                    rates = standard_distance / vari_distance
                    rate = np.median(rates)
                    rs *= rate
                    r, Tt = Utils.solveICPBySVD(p0=sPC, p1=pc)
                    newlms = r @ rs + Tt
                    rs_total = np.concatenate((rs_total, newlms.T), axis=0)
        noisy_rs = rs_total.reshape((-1, 68, 3))
        noisy_rs = Utils.injectNoiseInput(noisy_rs, config['AE_noi_percent'])
        to_writ = noisy_rs.reshape((-1, 204))
        np.savetxt('./dataset/AE_3dlms.txt', to_writ)
        print('数据保存完毕')

    @classmethod
    def AEdataload(cls):
        '''
            run this function for training AE model only
        '''
        noisy_rs = np.loadtxt('./dataset/AE_3dlms.txt')
        dataset = LSTMDataSet(torch.from_numpy(np.array(noisy_rs, dtype=np.float32)).cuda(),
                              torch.from_numpy(np.array(noisy_rs, dtype=np.float32)).cuda())
        train_division = int(dataset.input.shape[0] * config['AE_train_percent'])
        test_division = dataset.input.shape[0] - train_division
        train_dataloader = torch.utils.data.DataLoader(dataset[:train_division], shuffle=True,
                                                       batch_size=config['batch_size'])
        test_dataloader = torch.utils.data.DataLoader(dataset[test_division:], shuffle=True,
                                                      batch_size=config['batch_size'])
        return train_dataloader, test_dataloader

    @classmethod
    def AE_feature(cls, epoch, mid_dim, fold, split_train, split_test):
        from model import AutoEncoder
        checkpoint = torch.load(f'./weights/AE_model/AE_model_{epoch}_mid_{mid_dim}.pth')
        ae = AutoEncoder()
        ae.load_state_dict(checkpoint)
        ae.eval()
        ae.cuda()
        lms3d_train = np.loadtxt(f'dataset/3dlms_fold{fold}_train.txt').reshape((-1, 204))
        lms3d_test = np.loadtxt(f'dataset/3dlms_fold{fold}_train.txt').reshape((-1, 204))
        dataset_train = LSTMDataSet(torch.from_numpy(np.array(lms3d_train, dtype=np.float32)).cuda(),
                                    torch.from_numpy(np.array(lms3d_train, dtype=np.float32)).cuda())
        dataset_test = LSTMDataSet(torch.from_numpy(np.array(lms3d_test, dtype=np.float32)).cuda(),
                                   torch.from_numpy(np.array(lms3d_test, dtype=np.float32)).cuda())
        train_dataloader = torch.utils.data.DataLoader(dataset_train, shuffle=False, batch_size=config['batch_size'])
        test_dataloader = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size=config['batch_size'])
        train_feature = np.zeros((0, config['AE_mid_dim']))
        test_feature = np.zeros((0, config['AE_mid_dim']))
        for ip, tgt in train_dataloader:
            pred = ae(ip, use_mid=True)
            train_feature = np.concatenate((train_feature, pred.cpu().detach().numpy()), axis=0)
        for ipt, tgtt in test_dataloader:
            pred_t = ae(ipt, use_mid=True)
            test_feature = np.concatenate((test_feature, pred_t.cpu().detach().numpy()), axis=0)
        _, sp_train_feature = Utils.insertOrSample(split_train.astype(np.int_), train_feature.astype(np.float32))
        _, sp_test_feature = Utils.insertOrSample(split_test.astype(np.int_), test_feature.astype(np.float32))
        return sp_train_feature, sp_test_feature

    @classmethod
    def loadSingleFold(cls, fold, crop, peak_frame=False):
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # opt = vis.get_render_option()
        # opt.background_color = np.asarray([0, 0, 0])
        # opt.point_size = 5
        # opt.show_coordinate_frame = True

        label_test = np.loadtxt(f'./dataset/label_fold{fold}_test.txt').reshape(-1, 1)
        test = np.loadtxt(f'./dataset/ft_feature{fold}.txt')
        test_pca = p_m.transform(test)  # pca decomposition
        lms3d_test = np.loadtxt(f'./dataset/3dlms_fold{fold}_test.txt')
        split_test = np.loadtxt(f'./dataset/split_{fold}_test.txt')
        if crop:
            test_seqs, test_l = Utils.inserCompeletely(split_test.astype(np.int_), label_test.astype(np.float32))
            _, test_feature = Utils.inserCompeletely(split_test.astype(np.int_), test.astype(np.float32))
            _, test_lms3d = Utils.inserCompeletely(split_test.astype(np.int_), lms3d_test.astype(np.float32).reshape(-1, 204))
            if peak_frame:
                pos_embed_lst = []
                for t_f in test_feature:
                    pos_embed_lst.append(Utils.SinusoidalEncoding(t_f.shape[0], config['T_input_dim'])[::-1,:])  # 1 offset means except cls token position
                return test_l, test_feature, test_lms3d, test_seqs, np.array(pos_embed_lst, dtype=object)
            # 解锁注释查看point clouds
            # for i in range(test_lms3d.shape[0]):
            #     for s in range(test_lms3d[i].shape[0]):
            #         pc = test_lms3d[i][s].reshape((-1,3))
            #         pcd.points = o3d.utility.Vector3dVector(pc)
            #         pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
            #         vis.add_geometry(pcd)
            #         vis.poll_events()
            #         vis.update_renderer()
            return test_l, test_feature, test_lms3d, test_seqs
        else:
            return label_test, test, lms3d_test, split_test
    @classmethod
    def readAndLoadSingleFold(cls, fold):
        fold_root = '/home/exp-10086/Project/Emotion-FAN-master/data/txt/CK+_10-fold_sample_IDascendorder_step10.txt'
        fold_fp = open(fold_root)
        lines = fold_fp.readlines()
        start = 0
        for lidx, l in enumerate(lines):
            if l.__contains__(f'{fold}-fold'):
                vn = l.split(' ')[1]
                start = lidx
                break
            else:
                continue
        fold_lines = lines[start+1:start+int(vn)+1]
        pf_lst = []
        for fl in fold_lines:
            prefix = fl.split(' ')[0]
            prefix = prefix.replace('/','_')
            pf_lst.append(prefix)
        AE_root = '/home/exp-10086/Project/ferdataset/CK_AE_CODE'
        aes = os.listdir(AE_root)
        aes.sort()
        test = np.zeros((0,512))
        split_test = np.loadtxt(f'./dataset/split_{fold}_test.txt')
        for pf in pf_lst:
            for a in aes:
                if a.__contains__(pf):
                    ae_feaure = np.loadtxt(os.path.join(AE_root, a))
                    test = np.concatenate((test, ae_feaure.reshape(1,512)), axis=0)
        _, test_feature = Utils.inserCompeletely(split_test.astype(np.int_), test.astype(np.float32))
        pos_embed_lst = []
        for t_f in test_feature:
            pos_embed_lst.append(Utils.SinusoidalEncoding(t_f.shape[0], config['T_input_dim'])[::-1,:])  # 1 offset means except cls token position
        label_test = np.loadtxt(f'./dataset/label_fold{fold}_test.txt').reshape(-1, 1)
        _, test_l = Utils.inserCompeletely(split_test.astype(np.int_), label_test.astype(np.float32))
        return test_l, test_feature, np.array(pos_embed_lst, dtype=object)
    @classmethod
    def saveFacePicture(cls):
        import tqdm
        import os
        t = Models(False)
        root = 'E:/cohn-kanade-images/'
        serial = os.listdir(root)[1:]
        if os.path.isdir(f'{root}cut'):
            pass
        else:
            os.mkdir(f'{root}cut')
        for s in serial:
            parts_pth = os.path.join(root,s)
            parts = os.listdir(parts_pth)
            if os.path.isdir(f'{root}cut/{s}'):
                pass
            else:
                os.mkdir(f'{root}cut/{s}')
            for p in parts:
                if os.path.isdir(f'{root}cut/{s}/{p}'):
                    pass
                else:
                    os.mkdir(f'{root}cut/{s}/{p}')
                imgs_pth = os.path.join(parts_pth,p)
                imgs = os.listdir(imgs_pth)
                for i in imgs:
                    img_pth = os.path.join(imgs_pth, i)
                    img = cv2.imread(img_pth)
                    img = img.reshape(1, img.shape[0], img.shape[1], 3)
                    with torch.no_grad():
                        boxes = t.face_boxes(img)  # (bs, faces)
                        param_lst, roi_box_lst, boxed_imgs, deltaxty_lst = t.tddfa(img,
                                                                                   boxes)  # param_lst(faces(62 = 12 + 40 +10))  roi_box_lst(bs,faces) boxed_imgs(faces) deltaxty_lst(bs, faces)
                        ver_dense, ver_lst = t.tddfa.recon_vers(param_lst, list(chain(*roi_box_lst)),
                                                                dense_flag=config[
                                                                    '3DDFA_dense'])  # ver_dense(faces) ver_lst(faces)
                        if not type(ver_lst) in [tuple, list]:
                            ver_lst = [ver_lst]
                        img_bs_num = len(roi_box_lst)
                        pass_num = 0
                        for ibn in range(img_bs_num):
                            crop_lms = []
                            face_num_per_img = len(roi_box_lst[ibn])  # 一个img有多少张脸
                            ver_lst_fragment = ver_lst[pass_num:pass_num + face_num_per_img].copy()  # 第x张脸
                            crop_lms.append([ver_lst_fragment[0][0] - roi_box_lst[ibn][0][0] - deltaxty_lst[ibn][0][0],
                                             ver_lst_fragment[0][1] - roi_box_lst[ibn][0][1] - deltaxty_lst[ibn][0][1]])
                            eye_centers = Utils.getEyesAverage(crop_lms[0])  # 得到的是散装序列，若是按img，face 划分需要用face_num_per_img变量进行划分
                            aligned_boxed_img = Utils.face_align(boxed_imgs[0], eye_centers)  # 和上一行的对应起来，都只取了第一张人脸
                            cv2.imwrite(f'{root}cut/{s}/{p}/{i}',aligned_boxed_img)

    @classmethod
    def baseline(cls, ws, feature, lms3d, label, vote:bool=False):
        '''
            args:
                ws: window size, also known as sequence length
                feature: deep feature from imgages
                lms3d: 3D landmarks
                label: target data
            return:
                dataloader
        '''
        data = []
        target = []
        samples_counter_lst = []
        for f in range(0, feature.shape[0]):  # video level
            # f_mean = np.mean(feature[f][:,:512], axis=1)
            # f_std = np.std(feature[f][:,:512], axis=1)
            # feature[f][:,:512] = (feature[f][:,:512] - f_mean.reshape(-1, 1)) / f_std.reshape(-1, 1)

            # l_mean = np.mean(lms3d[f], axis=1)
            # l_std = np.std(lms3d[f], axis=1)
            # lms3d[f] = (lms3d[f] - l_mean.reshape(-1, 1)) / l_std.reshape(-1, 1)

            # concatnate and align to ws -- video level
            # ori_video = np.hstack((feature[f][:,:512], lms3d[f]))
            ori_video = lms3d[f]
            if ori_video.shape[0] < ws:
                # EOS
                EOS_num = ws - ori_video.shape[0]
                blanks = np.tile(EOS, (EOS_num, 1))
                video = np.concatenate((ori_video, blanks), axis=0)
            data.append(video)
            target.append(label[f][0][0])
            if vote:
                samples_counter_lst.append(f)
        # dataset and dataloader
        if vote:
            dataset = LSTMDataSet(torch.from_numpy(np.array(target, dtype=np.float32)).cuda(),
                                  torch.from_numpy(np.array(data, dtype=np.float32)).cuda(),
                                  torch.from_numpy(np.array(samples_counter_lst, dtype=np.float32)).cuda())
        else:
            dataset = LSTMDataSet(torch.from_numpy(np.array(target, dtype=np.float32)).cuda(),
                                  torch.from_numpy(np.array(data, dtype=np.float32)).cuda())
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=config['Shuffle'], batch_size=config['batch_size'])
        return dataloader

    @classmethod
    def ConvertVideo2Samples100Votes(cls, ws, feature, lms3d, label, vote:bool = True):
        '''
            args:
                ws: window size, also known as sequence length
                feature: deep feature from imgages
                lms3d: 3D landmarks
                label: target data
            return:
                dataloader
        '''
        data = []
        target = []
        samples_counter_lst = []
        for f in range(0, feature.shape[0]):  # video level
            # f_mean = np.mean(feature[f][:,:512], axis=1)
            # f_std = np.std(feature[f][:,:512], axis=1)
            # feature[f][:,:512] = (feature[f][:,:512] - f_mean.reshape(-1, 1)) / f_std.reshape(-1, 1)

            l_mean = np.mean(lms3d[f], axis=1)
            l_std = np.std(lms3d[f], axis=1)
            lms3d[f] = (lms3d[f] - l_mean.reshape(-1, 1)) / l_std.reshape(-1, 1)

            # concatnate and align to ws -- video level
            # ori_video = np.hstack((feature[f][:,:512], lms3d[f]))
            ori_video = np.hstack((lms3d[f], feature[f][:,:512]))
            # ori_video = lms3d[f]
            if ori_video.shape[0] < ws:
                # EOS
                EOS_num = ws - ori_video.shape[0]
                blanks = np.tile(EOS, (EOS_num, 1))
                video = np.concatenate((ori_video, blanks), axis=0)
                data.append(video)
                target.append(label[f][0][0])
                if vote:
                    samples_counter_lst.append(f)
            else:
                # one video can generate any number samples
                v_fs = np.arange(ori_video.shape[0])  # how many frames a video contains (frames,)
                v_ix = np.zeros((v_fs.shape[0],))  # what frames will be sampled (frames,)
                redundancy_matrix = np.zeros((config['standBy'], config['window_size']))  # a matrix stand for redundancy and sequence length, (how many standby samples, sequence length)
                for i in range(config['standBy']):
                    v_fs_shuffle = v_fs.copy()  # a cpoy for shuffle
                    iind = random.sample([xx for xx in range(ori_video.shape[0] // 2)], 1)
                    temp = v_fs_shuffle[iind[0]:min(ori_video.shape[0] // 5 * 3 + iind[0], ori_video.shape[0]-3)].copy()
                    np.random.shuffle(temp)  # every standby sample will be generated from origin video shuffled again (frames,)
                    v_fs_shuffle[iind[0]:min(ori_video.shape[0] // 5 * 3 + iind[0], ori_video.shape[0]-3)] = temp.copy()
                    v_ix_copy = v_ix.copy()  # index copy
                    v_ix_copy[v_fs_shuffle[iind[0]:config['window_size'] + iind[0]-3]] = 1  # code in inner bracket means a slicing operation on variable v_fs_shuffle, code in outer bracket means a mask operation.
                    v_ix_copy[-3:]=1
                    redundancy_matrix[i] = v_fs[v_ix_copy.astype(int).astype(bool)]  # assignment
                span = (redundancy_matrix.max(axis=1) - redundancy_matrix.min(axis=1))>= ori_video.shape[0]//2*1 # calulate the span of every standby sample, greater than some value will be selected as input sequence
                if np.sum(span!=0)>=config['selected']: # number of qualified sample might be greater than we want, so if true, just select top 20 samples.
                    samples = redundancy_matrix[span,:][:config['selected']]
                else: # if not, take as many as possible
                    samples = redundancy_matrix[:config['selected']]
                for w in range(samples.shape[0]):
                    data.append(ori_video[list(samples[w].astype(int))])
                    target.append(label[f][0][0].astype(np.long))
                    if vote:
                        samples_counter_lst.append(f)
        # dataset and dataloader
        if vote:
            dataset = LSTMDataSet(torch.from_numpy(np.array(target, dtype=np.float32)),
                                  torch.from_numpy(np.array(data, dtype=np.float32)),
                                  torch.from_numpy(np.array(samples_counter_lst, dtype=np.float32)))
        else:
            dataset = LSTMDataSet(torch.from_numpy(np.array(target, dtype=np.float32)).cuda(),
                                  torch.from_numpy(np.array(data, dtype=np.float32)).cuda())
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=config['Shuffle'], batch_size=config['batch_size'])
        return dataloader

    @classmethod
    def ConvertVideo2SamlpesConstantSpeed(cls, ws, feature, lms3d, label, vote: bool = True, pos_embed=None):
        '''
            args:
                ws: window size, also known as sequence length
                feature: deep feature from imgages
                lms3d: 3D landmarks
                label: target data
                vote: count voting
                pos_embed: origin video's frames's cosine-sine position embedding, peak frame's position is fixed.
            return:
                dataloader
        '''
        data = []
        target = []
        samples_counter_lst = []
        pos_embeds = []
        for f in range(0, feature.shape[0]):  # video level
            # import open3d as o3d
            # pcd = o3d.geometry.PointCloud()
            # f_mean = np.mean(feature[f][:,:512], axis=1)
            # f_std = np.std(feature[f][:,:512], axis=1)
            # feature[f][:,:512] = (feature[f][:,:512] - f_mean.reshape(-1, 1)) / f_std.reshape(-1, 1)

            # l_mean = np.mean(lms3d[f], axis=1)
            # l_std = np.std(lms3d[f], axis=1)
            # lms3d[f] = (lms3d[f] - l_mean.reshape(-1, 1)) / l_std.reshape(-1, 1)

            # concatnate and align to ws -- video level
            ori_video = feature[f]
            # ori_video = np.hstack((lms3d[f], feature[f][:, :512]))  # 340
            # ori_video = lms3d[f]
            samples_idx = Utils.ConstantSpeed(ori_video.shape[0])  # (sample num, length）
            for w in range(samples_idx.shape[0]):
                v_s = ori_video[samples_idx[w]]  # 采样的样本
                if v_s.shape[0] < ws:
                    EOS_num = ws - v_s.shape[0]
                    blanks = np.tile(EOS, (EOS_num, 1))
                    processed_v_s = np.concatenate((v_s, blanks), axis=0)
                else:
                    processed_v_s = v_s.copy()
                # if w == 2:
                #     for pttth in range(6):
                #         pcd.points = o3d.utility.Vector3dVector(v_s[pttth][:204].copy().reshape((68, 3)))
                #         pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                #         o3d.visualization.draw_geometries([pcd])
                data.append(processed_v_s)
                target.append(label[f][0][0].astype(np.long))
                if vote:
                    samples_counter_lst.append(f)
                if pos_embed is not None:
                    pe_p = pos_embed[f][samples_idx[w]]
                    if pe_p.shape[0] < ws:
                        EOS_num = ws - pe_p.shape[0]
                        blanks = np.tile(PE_EOS, (EOS_num, 1))
                        processed_pe = np.concatenate((pe_p, blanks), axis=0)
                    else:
                        processed_pe = pe_p.copy()
                    pos_embeds.append(processed_pe)
        # dataset and dataloader !!! x,y 是必须的，有无voting和有无fixed peak frame position变成了4个情况。
        if vote:  # 用于test
            if pos_embed is not None:  # 用于peak frame fixed position encoding
                dataset = LSTMDataSet(torch.from_numpy(np.array(target, dtype=np.float32)).cuda(),
                                      torch.from_numpy(np.array(data, dtype=np.float32)).cuda(),
                                      torch.from_numpy(np.array(samples_counter_lst, dtype=np.float32)).cuda(),
                                      torch.from_numpy(np.array(pos_embeds, dtype=np.float32)).cuda())
            else:
                dataset = LSTMDataSet(torch.from_numpy(np.array(target, dtype=np.float32)).cuda(),
                                      torch.from_numpy(np.array(data, dtype=np.float32)).cuda(),
                                      torch.from_numpy(np.array(samples_counter_lst, dtype=np.float32)).cuda())
        else:
            if pos_embed is not None:
                dataset = LSTMDataSet(torch.from_numpy(np.array(target, dtype=np.float32)).cuda(),
                                      torch.from_numpy(np.array(data, dtype=np.float32)).cuda(),
                                      None,
                                      torch.from_numpy(np.array(pos_embeds, dtype=np.float32)).cuda())
            else:
                dataset = LSTMDataSet(torch.from_numpy(np.array(target, dtype=np.float32)).cuda(),
                                      torch.from_numpy(np.array(data, dtype=np.float32)).cuda())
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=config['Shuffle'], batch_size=config['batch_size'])
        return dataloader

    @classmethod
    def rgbPatchFea(cls):
        import dlib
        import cv2
        import numpy as np
        # dlib detection
        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor('F:/conda/envs/ak/micro-expressions/shape_predictor_68_face_landmarks.dat')
        image = cv2.imread('F:/conda/envs/ak/emotion_FAN/data/face/ck_face/S005/001/S005_001_00000011.png')
        cv2.imshow('224*224',image)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(image_gray)
        landmarks = sp(image_gray,faces[0])
        mask = np.zeros((224,224))
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # mask[max(0,int(y)-3):min(223,int(y)+3),max(0,int(x)+3):min(223,int(x)+3)] = 1
            mask[int(y)-4:int(y)+5,int(x)-4:int(x)+5] = 1
            # patch = image[int(y)-10:int(y)+10,int(x)-10:int(x)+10]
            # cv2.imshow('pa',patch)
            # cv2.waitKey(0)
            # cv2.circle(image,(int(x),int(y)),2,[0,255,0])
        image = image*mask.reshape((224,224,1))
        cv2.imshow('patch',image/255)
        cv2.waitKey(0)
        # dlib bounding box
        # dlib landmarks
        # dlib landmarks patch

    @classmethod
    def pretrainDataProcess(cls):
        import tqdm
        ''' 1.3dlms and alignment
            2.rgb feature'''

        train_lst = []
        split_train = []
        videos_pth = '/home/exp-10086/Project/ferdataset/processed/'
        videos = os.listdir(videos_pth)
        t = Models(False)
        for vis in videos: # 开心01.mp4,开心02.mp4,...
            video_pth = os.path.join(videos_pth,vis)
            fragments = os.listdir(video_pth)
            for frag in fragments: # 0,1,2,...
                train_lst.clear()
                split_train.clear()
                frag_pth = os.path.join(video_pth,frag)
                video_list = os.listdir(frag_pth)
                video_list.sort(key=lambda x:int(x[:-4]))
                split_train.append(len(video_list))
                for v in video_list:
                    train_lst.append((os.path.join(video_pth,frag,v),9,len(video_list)))
                # sPC = Utils.standardPC(t)  # standard PointCloud
                # random_idx = random.sample([i for i in range(config['PC_points_sample_range'])], config['PC_points_piars'])
                # alphi_lst = []
                # fi_lst = []
                for pth, label_, fms_number in tqdm.tqdm(train_lst):
                    try:
                        fi, alphi, ver_lst, np_label, pc = Utils.deep_features(pth, label_, t, 'fan')
                    except Exception:
                        continue
                    # fi_lst.append(fi)  # deep feature
                    # alphi_lst.append(alphi)  # alpha i
                    # standard_idx = sPC[random_idx]
                    # standard_distance = np.linalg.norm(standard_idx[:int(config['PC_points_piars'] / 2)] - standard_idx[int(config['PC_points_piars'] / 2):],
                    #     axis=1)
                    # vari_idx = pc[random_idx]
                    # vari_distance = np.linalg.norm(vari_idx[:int(config['PC_points_piars'] / 2)] - vari_idx[int(config['PC_points_piars'] / 2):], axis=1)
                    # rates = standard_distance / vari_distance
                    # rate = np.median(rates)  # scale value between new pc and standard pc
                    # ver_lst[0] *= rate
                    # pc *= rate
                    # r, Tt = Utils.solveICPBySVD(p0=sPC, p1=pc)
                    # newlms = r @ ver_lst[0] + Tt
                    # lms pc standardPC 都在一起
                    # Utils.open3dVerify(newlms,newpc.T,sPC)
                    with open(f'/home/exp-10086/Project/mae-main/dataset/train_{vis[:-4]}_{frag}_feature.txt', 'ab') as n3d:
                        np.savetxt(n3d, fi.cpu().numpy())
                    # with open(f'./dataset/label_fold{test_fold}_train.txt', 'ab') as nl:
                    #     np.savetxt(nl, np_label)
                # Utils.aggregateFeaAndCode(fi_lst, alphi_lst, split_train, 0, 'train')
if __name__ == '__main__':
    # 删除所有.DS_Store文件
    # Dataprocess.deleleDS_Store()
    # CK+数据加载
    # Dataprocess.loadCKPlusData(1)
    # for i in range(1, 11):
    #     print(i)
    #     Dataprocess.loadCKPlusData(i)
    # 得到dataset用于训练
    # rs = Dataprocess.dataForLSTM(1, crop=True)
    # Dataprocess.dataAlign2WindowSize(config['window_size'], rs[1], rs[2], rs[0])
    # 测试插值
    # b = 12
    # a = np.arange(b).reshape(b,1)
    # delta = 15 - b
    # Utils.insertMethod(a,delta)
    # 采样测试
    # b = 71
    # a = np.arange(b).reshape(b,1)
    # Utils.sampleMethod(a)
    # 测试特征提取
    # for f in range(1,11):
    #     t = Models(f)
    #     Utils.deep_features('./test.jpg', 1, t, 'fan')
    # unit test of 3d lms of all CK+
    # Dataprocess.AEinput('E:/cohn-kanade-images/')
    # verify lms
    # import open3d as o3d
    # noisy_rs = np.loadtxt('./dataset/AE_3dlms.txt')
    # pcd = o3d.geometry.PointCloud()
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([0, 0, 0])
    # opt.point_size = 5
    # opt.show_coordinate_frame = True
    # for n in range(noisy_rs.shape[0]):
    #     pcd.points = o3d.utility.Vector3dVector(noisy_rs[n].reshape((68,3)))
    #     pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    #     vis.add_geometry(pcd)
    #     vis.poll_events()
    #     vis.update_renderer()
    # save face images
    # Dataprocess.saveFacePicture()
    # 68 lms2d
    # Dataprocess.rgbPatchFea()
    # Dataprocess.pretrainDataProcess()
    Dataprocess.readAndLoadSingleFold(1, True)
    pass
import logging
import random

import numpy as np
import cv2
import os
from itertools import chain
import yaml
import pdb
from container import Models
import torch
import torch.utils.data as data
from utils.pose import viz_pose
from sklearn.decomposition import PCA
import joblib

target = np.array([[16, 12], [31, 12]])  # 还是要调整一下
target_fan = np.array([[75, 56], [150, 56]])
config = yaml.safe_load(open('./config.yaml'))
EOS = np.zeros((1, config['LSTM_input_dim']))
standardImg = cv2.imread('./img.png')


class LSTMDataSet(data.Dataset):
    def __init__(self, label, concat_features):
        self.target = label
        self.input = concat_features
    def __getitem__(self, idx):

        return self.input[idx], self.target[idx]

    def __len__(self):
        return self.target.shape[0]


class Utils():

    def __init__(self):
        pass

    @classmethod
    def getEyesAverage(cls, crop_lms):
        """
        get center coordinates per eye in 2D
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
            param_lst, roi_box_lst, _, _ = t.tddfa(img, boxes)  # param_lst(faces(62 = 12 + 40 +10))  roi_box_lst(bs,faces) boxed_imgs(faces) deltaxty_lst(bs, faces)
            ver_dense, ver_lst = t.tddfa.recon_vers(param_lst, list(chain(*roi_box_lst)), dense_flag=config['3DDFA_dense'])  # ver_dense(faces) ver_lst(faces)
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
            param_lst, roi_box_lst, boxed_imgs, deltaxty_lst = t.tddfa(img,
                                                                       boxes)  # param_lst(faces(62 = 12 + 40 +10))  roi_box_lst(bs,faces) boxed_imgs(faces) deltaxty_lst(bs, faces)
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
                    feature = torch.mean(t.Res_model(cs / 65025), dim=0).reshape(1, -1)  # 替代源码transforms.Normalize(mean=0,std=255)的效果,features = (bs,特征数)
                    np_f = feature.cpu().numpy()
                else:
                    aligned_boxed_img = np.swapaxes(aligned_boxed_img, 0, 2)
                    aligned_boxed_img = np.swapaxes(aligned_boxed_img, 1, 2)
                    f, alpha = t.FAN(torch.from_numpy(aligned_boxed_img).unsqueeze(0).float().cuda()/255, phrase='eval')  # 改用拼接特征
                    # np_f = f.cpu().numpy()
                np_label = np.array([label_])
                cs = torch.empty(0)
                pass_num += face_num_per_img
                # pc和lms一同摆正
                T = viz_pose(param_lst)
                pc = ver_dense[0].T  # 这里可能有坑,索引应该怎么写？
                pc = pc @ T[:3, :3].T + T[:3, 3].T
                ver_lst[ibn] = T[:3, :3]@ver_lst[ibn] + T[:3, 3].reshape(-1,1)
                ver_lst[ibn] -= pc[0].T.reshape(-1,1)
                pc -= pc[0]
                # lms和pc应该贴在一起
                # import open3d as o3d
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(np.vstack((pc,ver_lst[ibn].T)))
                # o3d.visualization.draw_geometries([pcd])
            return f, alpha, ver_lst, np_label, pc

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
        sam_idx = sorted(random.sample([S for S in range(sample_range)], config['Max_frame']))
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
                # 插值，小于15帧的都拉长放在new_data_lst
                new_spilt.append(config['Least_frame'])  # 必定是15
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
                video_sized_copy_feature = torch.tile(video_f_lst[idx], (sptr, 1))
                concatnated_f = torch.cat((fi_lst[v_idx:v_idx + sptr], video_sized_copy_feature), dim=1)
                np_f = concatnated_f.cpu().numpy()
                np.savetxt(nf, np_f)

    @classmethod
    def open3dVerify(cls, ver_lst, pc, sPC):
        import open3d as o3d
        pcd_s = o3d.geometry.PointCloud()
        pcd_pc = o3d.geometry.PointCloud()
        pcd_s.points = o3d.utility.Vector3dVector(ver_lst[0].T)
        pcd_pc.points = o3d.utility.Vector3dVector(np.vstack((pc,sPC)))
        o3d.visualization.draw_geometries([pcd_s,pcd_pc])

class Dataprocess():
    def __init__(self):
        pass

    @classmethod
    def datasetProcess(cls, path):
        emotion = os.listdir(path)
        for e in emotion:
            video_path = os.path.join(path, e)
            videos = os.listdir(video_path)
            for v in videos:
                p = os.path.join(video_path, v)
                cap = cv2.VideoCapture(p)
                frame = cap.read()
                print(frame[1].shape)

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
        for index, item in enumerate(
                data_directory):  # 0, '1-fold\t31\n' in {[0, '1-fold\t31\n'], [1, 'S037/006 Happy\n'], ...}分化两个数据集,txt的操作
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
        t = Models(False,test_fold)
        sPC = Utils.standardPC(t)
        random_idx = random.sample([i for i in range(config['PC_points_sample_range'])],config['PC_points_piars'])
        alphi_lst = []
        fi_lst = []

        for pth, label_, fms_number in tqdm.tqdm(train_lst):
            fi, alphi, ver_lst, np_label, pc = Utils.deep_features(pth, label_, t, 'fan')
            fi_lst.append(fi)
            alphi_lst.append(alphi)
            standard_idx = sPC[random_idx]
            standard_distance = np.linalg.norm(standard_idx[:int(config['PC_points_piars']/2)] - standard_idx[int(config['PC_points_piars']/2):],axis=1)
            vari_idx = pc[random_idx]
            vari_distance = np.linalg.norm(vari_idx[:int(config['PC_points_piars']/2)] - vari_idx[int(config['PC_points_piars']/2):],axis=1)
            rates = standard_distance/vari_distance
            rate = np.median(rates)
            ver_lst[0] *= rate
            pc *= rate
            # lms pc standardPC 都在一起
            # Utils.open3dVerify(ver_lst,pc,sPC)
            # with open(f'./dataset/3dlms_fold{test_fold}_train.txt', 'ab') as n3d:
            #     np.savetxt(n3d, ver_lst[0].T)
            # with open(f'./dataset/label_fold{test_fold}_train.txt', 'ab') as nl:
            #     np.savetxt(nl, np_label)
        Utils.aggregateFeaAndCode(fi_lst,alphi_lst,split_train,test_fold,'train')
        alphi_lst = []
        fi_lst = []
        for pth, label_, fms_number in tqdm.tqdm(test_lst):
            fi, alphi, ver_lst, np_label, pc = Utils.deep_features(pth, label_, t, 'fan')
            fi_lst.append(fi)
            alphi_lst.append(alphi)
            standard_idx = sPC[random_idx]
            standard_distance = np.linalg.norm(standard_idx[:int(config['PC_points_piars']/2)] - standard_idx[int(config['PC_points_piars']/2):], axis=1)
            vari_idx = pc[random_idx]
            vari_distance = np.linalg.norm(vari_idx[:int(config['PC_points_piars']/2)] - vari_idx[int(config['PC_points_piars']/2):], axis=1)
            rates = standard_distance / vari_distance
            rate = np.median(rates)
            ver_lst[0] *= rate
            pc *= rate
            # with open(f'./dataset/3dlms_fold{test_fold}_test.txt', 'ab') as n3d:
            #     np.savetxt(n3d, ver_lst[0].T)
            # with open(f'./dataset/label_fold{test_fold}_test.txt', 'ab') as nl:
            #     np.savetxt(nl, np_label)
        Utils.aggregateFeaAndCode(fi_lst, alphi_lst, split_test, test_fold,'test')
        # for fms in tqdm.tqdm(split_train):
        #     with open(f'./dataset/split_{test_fold}_train.txt', 'ab') as nfn:
        #         np.savetxt(nfn, np.array(fms).reshape([1, -1]))
        # for fms in tqdm.tqdm(split_test):
        #     with open(f'./dataset/split_{test_fold}_test.txt', 'ab') as nfn:
        #         np.savetxt(nfn, np.array(fms).reshape([1, -1]))
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
        data = np.concatenate(ori_data,axis=0)
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
            pca_lst.append(pca_data[idx:idx+s])
            idx += s
        return np.array(pca_lst, dtype=object)
    @classmethod
    def dataAlign2WindowSize(cls, ws, feature, lms3d, label, step:int = 1):
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
        '''normalize image features and 3d lms, then concatenate them, but sequence length is fixed, 
            so either break a sample to slices, either concatenate EOS to window size.'''
        for f in range(0, feature.shape[0]):  # video level
            '''this is normalization,  However,3d lms will be a circle in space, 
                it seems lost much information, so stop normalization for now.'''
            # f_mean = np.mean(feature[f][:,:512], axis=1)
            # f_std = np.std(feature[f][:,:512], axis=1)
            # feature[f][:,:512] = (feature[f][:,:512] - f_mean.reshape(-1, 1)) / f_std.reshape(-1, 1)

            l_mean = np.mean(lms3d[f], axis=1)
            l_std = np.std(lms3d[f], axis=1)
            lms3d[f] = (lms3d[f] - l_mean.reshape(-1, 1)) / l_std.reshape(-1, 1)

            # concatnate and align to ws -- video level
            # ori_video = np.hstack((feature[f], lms3d[f]))
            ori_video = lms3d[f]
            if ori_video.shape[0] < ws:
                # EOS
                EOS_num = ws - ori_video.shape[0]
                blanks = np.tile(EOS, (EOS_num, 1))
                video = np.concatenate((ori_video, blanks), axis=0)
                data.append(video)
                target.append(label[f][0][0])
            else:
                #  one video generates more shape[0] - ws samples
                for w in range(ori_video.shape[0] - ws + 1):
                    sample = ori_video[w:w + config['window_size']]
                    data.append(sample)
                    target.append(label[f][0][0].astype(np.long))

        # dataset and dataloader
        dataset = LSTMDataSet(torch.from_numpy(np.array(target, dtype=np.float32)).cuda(), torch.from_numpy(np.array(data, dtype=np.float32)).cuda())
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=config['batch_size'])
        return dataloader

if __name__ == '__main__':
    # 高清视频社区的素材
    # Dataprocess.datasetProcess('E:/lstm_data4train')
    # 删除所有.DS_Store文件
    # Dataprocess.deleleDS_Store()
    # CK+数据加载
    # Dataprocess.loadCKPlusData(1)
    # for i in range(1, 11):
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
    pass

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

target = np.array([[16, 12], [31, 12]])  # 还是要调整一下
target_fan = np.array([[75, 56], [150, 56]])
config = yaml.safe_load(open('./config.yaml'))
EOS = np.ones((1, config['LSTM_input_dim']))

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
        :param crop_lms: [[68+68],[68+68]]
        :return:
        """
        crop_lms = np.array(crop_lms)
        eyel, eyer = np.mean(crop_lms[:, 36:41], axis=1), np.mean(crop_lms[:, 42:47], axis=1)
        rs = np.append(eyel, eyer)
        return rs.reshape(-1, 4).astype(np.int_)

    @classmethod
    def face_align(cls, boxed_img: np, eye_centers: np, conv_model: str):
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
    def deep_features(cls, pth, label_, t, conv_model: str = 'fan'):
        img = cv2.imread(pth)
        img = img.reshape(1, img.shape[0], img.shape[1], 3)
        with torch.no_grad():
            boxes = t.face_boxes(img)
            param_lst, roi_box_lst, boxed_imgs, deltaxty_lst = t.tddfa(img, boxes)
            _, ver_lst = t.tddfa.recon_vers(param_lst, list(chain(*roi_box_lst)), dense_flag=config['3DDFA_dense'])
            if not type(ver_lst) in [tuple, list]:
                ver_lst = [ver_lst]
            img_bs_num = len(roi_box_lst)
            pass_num = 0
            for ibn in range(img_bs_num):
                crop_lms = []
                face_num_per_img = len(roi_box_lst[ibn])  # 一个img有多少张脸
                ver_lst_fragment = ver_lst[pass_num:pass_num + face_num_per_img]
                crop_lms.append([ver_lst_fragment[0][0] - roi_box_lst[ibn][0][0] - deltaxty_lst[ibn][0][0],
                                 ver_lst_fragment[0][1] - roi_box_lst[ibn][0][1] - deltaxty_lst[ibn][0][1]])
                eye_centers = Utils.getEyesAverage(crop_lms[0])  # 得到的是散装序列，若是按img，face 划分需要用face_num_per_img变量进行划分
                aligned_boxed_img = Utils.face_align(boxed_imgs[0], eye_centers, conv_model)
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
                    aligned_boxed_img = np.swapaxes(aligned_boxed_img, 0, 2)
                    aligned_boxed_img = np.swapaxes(aligned_boxed_img, 1, 2)
                    f, _ = t.FAN(torch.from_numpy(aligned_boxed_img).unsqueeze(0).float().cuda(), phrase='eval')
                    np_f = f.cpu().numpy()
                np_label = np.array([label_])
                cs = torch.empty(0)
                pass_num += face_num_per_img
            return np_f, ver_lst, np_label

    @classmethod
    def insertMethod(cls, ori_data, delta, inverse: bool = False):
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
            spilt: 一个视频原有的帧数 np (294,)
            data_lst: 数据集，利用spilt划分 label:(5231,) feature:(5231,512) lms3d:(5231,204)
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
        :param test_fold: which fold to be test dataset
        :return:
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
        t = Models()
        for pth, label_, fms_number in tqdm.tqdm(train_lst):
            np_f, ver_lst, np_label = Utils.deep_features(pth, label_, t, 'fan')
            with open(f'./dataset/fan_feature_fold{test_fold}_train.txt', 'ab') as nf:
                np.savetxt(nf, np_f)
            with open(f'./dataset/3dlms_fold{test_fold}_train.txt', 'ab') as n3d:
                np.savetxt(n3d, ver_lst[0].T)
            with open(f'./dataset/label_fold{test_fold}_train.txt', 'ab') as nl:
                np.savetxt(nl, np_label)
        for pth, label_, fms_number in tqdm.tqdm(test_lst):
            np_f, ver_lst, np_label = Utils.deep_features(pth, label_, t, 'fan')
            with open(f'./dataset/fans_feature_fold{test_fold}_test.txt', 'ab') as nf:
                np.savetxt(nf, np_f)
            with open(f'./dataset/3dlms_fold{test_fold}_test.txt', 'ab') as n3d:
                np.savetxt(n3d, ver_lst[0].T)
            with open(f'./dataset/label_fold{test_fold}_test.txt', 'ab') as nl:
                np.savetxt(nl, np_label)
        for fms in tqdm.tqdm(split_train):
            with open(f'./dataset/split_{test_fold}_train.txt', 'ab') as nfn:
                np.savetxt(nfn, np.array(fms).reshape([1, -1]))
        for fms in tqdm.tqdm(split_test):
            with open(f'./dataset/split_{test_fold}_test.txt', 'ab') as nfn:
                np.savetxt(nfn, np.array(fms).reshape([1, -1]))

    @classmethod
    def dataForLSTM(cls, fold, crop: bool = False):
        label_train = np.loadtxt(f'./dataset/label_fold{fold}_train.txt').reshape(-1, 1)  # 每一帧的label
        train = np.loadtxt(f'./dataset/fan_feature_fold{fold}_train.txt')  # 每一帧的feature
        lms3d_train = np.loadtxt(f'./dataset/3dlms_fold{fold}_train.txt')  # 每一帧的lms
        split_train = np.loadtxt(f'./dataset/split_{fold}_train.txt')  # 如何分帧

        label_test = np.loadtxt(f'./dataset/label_fold{fold}_test.txt').reshape(-1, 1)
        test = np.loadtxt(f'./dataset/fans_feature_fold{fold}_test.txt')
        lms3d_test = np.loadtxt(f'./dataset/3dlms_fold{fold}_test.txt')
        split_test = np.loadtxt(f'./dataset/split_{fold}_test.txt')
        if crop:
            train_seqs, train_l = Utils.insertOrSample(split_train.astype(np.int_), label_train.astype(np.float32))
            _, train_feature = Utils.insertOrSample(split_train.astype(np.int_), train.astype(np.float32))
            _, train_lms3d = Utils.insertOrSample(split_train.astype(np.int_), lms3d_train.astype(np.float32).reshape(-1, 204))

            test_seqs, test_l = Utils.insertOrSample(split_test.astype(np.int_), label_test.astype(np.float32))
            _, test_feature = Utils.insertOrSample(split_test.astype(np.int_), test.astype(np.float32))
            _, test_lms3d = Utils.insertOrSample(split_test.astype(np.int_), lms3d_test.astype(np.float32).reshape(-1, 204))

            return train_l, train_feature, train_lms3d, train_seqs, test_l, test_feature, test_lms3d, test_seqs
        else:
            return label_train, train, lms3d_train, split_train, label_test, test, lms3d_test, split_test

    @classmethod
    def dataAlign2WindowSize(cls, ws, feature, lms3d, label):
        data = []
        target = []
        # 先feature和lms3d标准化，然后concatenate，最后根据shape[0]和ws的关系决定补EOS还是切片
        for f in range(feature.shape[0]):# 有几个vedio样本
            # 每个video样本的特征进行标准化
            f_mean = np.mean(feature[f],axis=1)
            f_std = np.std(feature[f],axis=1)
            feature[f] = (feature[f] - f_mean.reshape(-1,1))/f_std.reshape(-1,1)

            l_mean = np.mean(lms3d[f], axis=1)
            l_std = np.std(lms3d[f], axis=1)
            lms3d[f] = (lms3d[f] - l_mean.reshape(-1,1)) / l_std.reshape(-1,1)

            # concatnate and align to ws -- video level
            ori_video = np.hstack((feature[f],lms3d[f]))
            if ori_video.shape[0] < ws:
                # EOS
                EOS_num = ws - ori_video.shape[0]
                blanks = np.tile(EOS,(EOS_num,1))
                video = np.concatenate((ori_video, blanks),axis=0)
                data.append(video)
                target.append(label[f][0][0])
            else:
                #  one video generates more shape[0] - ws samples
                for w in range(ori_video.shape[0] - ws + 1):
                    sample = ori_video[w:w+config['window_size']]
                    data.append(sample)
                    target.append(label[f][0][0].astype(np.long))

            # dataset and dataloader
            dataset = LSTMDataSet(torch.from_numpy(np.array(target,dtype=np.float32)).cuda(),torch.from_numpy(np.array(data,dtype=np.float32)).cuda())
            dataloader = torch.utils.data.DataLoader(dataset,shuffle=True,batch_size=config['batch_size'])
        return dataloader
if __name__ == '__main__':
    # 高清视频社区的素材
    # Dataprocess.datasetProcess('E:/lstm_data4train')
    # 删除所有.DS_Store文件
    # Dataprocess.deleleDS_Store()
    # CK+数据加载
    # for i in range(1, 11):
    #     Dataprocess.loadCKPlusData(i)
    # 得到dataset用于训练
    rs = Dataprocess.dataForLSTM(1, crop=True)
    Dataprocess.dataAlign2WindowSize(config['window_size'], rs[1], rs[2], rs[0])
    # 测试插值
    # b = 12
    # a = np.arange(b).reshape(b,1)
    # delta = 15 - b
    # Utils.insertMethod(a,delta)
    # 采样测试
    # b = 71
    # a = np.arange(b).reshape(b,1)
    # Utils.sampleMethod(a)
    pass

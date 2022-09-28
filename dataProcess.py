import numpy as np
import cv2
import os
from itertools import chain
import yaml
import pdb
from container import Models
import torch

target = np.array([[16, 12], [31, 12]])  # 还是要调整一下
target_fan = np.array([[75, 56], [150, 56]])
config = yaml.safe_load(open('./config.yaml'))


def deep_features(pth, label_, t, conv_model: str = 'fer'):
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
            if conv_model=='fer':
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
                f, _ = t.FAN(torch.from_numpy(aligned_boxed_img).unsqueeze(0).float().cuda(), phrase='eval')
                np_f = f.cpu().numpy()
            np_label = np.array([label_])
            cs = torch.empty(0)
            pass_num += face_num_per_img
        return np_f, ver_lst, np_label


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
        if conv_model=='fer':
            size = (48, 48)
            RotateMatrix, _ = cv2.estimateAffinePartial2D(eye_centers.reshape(2, 2), target)
        else:
            size = (224, 224)
            RotateMatrix, _ = cv2.estimateAffinePartial2D(eye_centers.reshape(2, 2), target_fan)
        RotImg = cv2.warpAffine(boxed_img, RotateMatrix, size, borderMode=cv2.BORDER_REPLICATE,
                                borderValue=(255, 255, 255))
        if conv_model=='fer':
            gary = cv2.cvtColor(RotImg, cv2.COLOR_RGBA2GRAY)
            return gary
        # cv2.imshow('rotedGary', gary)
        # cv2.waitKey(0)
        return RotImg


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
        split_train=[]
        split_test=[]
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
                train_lst.append((os.path.join(video_path, frame), label,frames))
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
        # 得到深度特征,3dlms,label
        t = Models()
        for pth, label_, fms_number in tqdm.tqdm(train_lst):
            np_f, ver_lst, np_label = deep_features(pth, label_, t, 'fan')
            with open(f'./dataset/fan_feature_fold{test_fold}_train.txt', 'ab') as nf:
                np.savetxt(nf, np_f)
            # with open(f'./dataset/3dlms_fold{test_fold}_train.txt', 'ab') as n3d:
            #     np.savetxt(n3d, ver_lst[0].T)
            # with open(f'./dataset/label_fold{test_fold}_train.txt', 'ab') as nl:
            #     np.savetxt(nl, np_label)
        for pth, label_, fms_number in tqdm.tqdm(test_lst):
            np_f, ver_lst, np_label = deep_features(pth, label_, t, 'fan')
            with open(f'./dataset/fans_feature_fold{test_fold}_test.txt', 'ab') as nf:
                np.savetxt(nf, np_f)
            # with open(f'./dataset/3dlms_fold{test_fold}_test.txt', 'ab') as n3d:
            #     np.savetxt(n3d, ver_lst[0].T)
            # with open(f'./dataset/label_fold{test_fold}_test.txt', 'ab') as nl:
            #     np.savetxt(nl, np_label)
        # for fms in tqdm.tqdm(split_train):
        #     with open(f'./dataset/split_{test_fold}_train.txt', 'ab') as nfn:
        #         np.savetxt(nfn, np.array(fms).reshape([1,-1]))
        # for fms in tqdm.tqdm(split_test):
        #     with open(f'./dataset/split_{test_fold}_test.txt', 'ab') as nfn:
        #         np.savetxt(nfn, np.array(fms).reshape([1,-1]))
    @classmethod
    def dataForLSTM(cls,fold):
        label_train = np.loadtxt(f'./dataset/label_fold{fold}_train.txt')
        train = np.loadtxt(f'./dataset/res_feature_fold{fold}_train.txt')
        lms3d_train = np.loadtxt(f'./dataset/3dlms_fold{fold}_train.txt')
        split_train = np.loadtxt(f'./dataset/split_{fold}_train.txt')

        label_test = np.loadtxt(f'./dataset/label_fold{fold}_test.txt')
        test = np.loadtxt(f'./dataset/res_feature_fold{fold}_test.txt')
        lms3d_test = np.loadtxt(f'./dataset/3dlms_fold{fold}_test.txt')
        split_test = np.loadtxt(f'./dataset/split_{fold}_test.txt')
        return torch.from_numpy(label_train.astype(np.float32)).cuda(),torch.from_numpy(train.astype(np.float32)).cuda(),torch.from_numpy(lms3d_train.astype(np.float32)).cuda(),split_train.astype(np.float32),torch.from_numpy(label_test.astype(np.float32)).cuda(),torch.from_numpy(test.astype(np.float32)).cuda(),torch.from_numpy(lms3d_test.astype(np.float32)).cuda(),split_test.astype(np.float32)
if __name__ == '__main__':
    # 高清视频社区的素材
    # Dataprocess.datasetProcess('E:/lstm_data4train')
    # 删除所有.DS_Store文件
    # Dataprocess.deleleDS_Store()
    # CK+数据加载
    for i in range(1, 11):
        Dataprocess.loadCKPlusData(i)
    # 得到dataset用于训练
    # rs = Dataprocess.dataForLSTM(1)
    pass
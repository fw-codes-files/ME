def checkCKplusDataset():
    import os
    count = 0
    ca = os.listdir('E:/cohn-kanade-images/')
    for s in ca:
        sub_ca1 = os.path.join('E:/cohn-kanade-images/', s)
        sub_ca = os.listdir(sub_ca1)
        for seq in sub_ca:
            d = os.path.join(sub_ca1, seq)
            if os.path.isdir(d):
                count += 1
    print(count)
    '''总数量正确，带label数据集已经由emotion-FAN确认'''


def dicYaml():
    # yaml中存储dict数据类型测试
    import yaml
    cfg = yaml.safe_load(open('./config.yaml', 'r', encoding='utf-8').read())
    print(cfg['CK+_dict']['Happy'])
    print(type(float(cfg['learning_rate'])))

def countSequenceLen():
    import os
    min_len = []
    wenjianjia = os.listdir('E:/cohn-kanade-images/')
    for w in wenjianjia:
        bianhao = os.listdir(os.path.join('E:/cohn-kanade-images/', w))
        for b in bianhao:
            frame = os.listdir(os.path.join('E:/cohn-kanade-images/', w,b))
            min_len.append(len(frame))
            if len(frame)==4:
                print(bianhao,frame)
    print(sorted(min_len))

if __name__ == '__main__':
    pass

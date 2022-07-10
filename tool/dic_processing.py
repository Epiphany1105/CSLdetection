import torch


def dic_load_CSL2word(_path):
    # 导入dic
    labels = {}
    label_file = open(_path, 'r', encoding="utf-8")
    for line in label_file.readlines():
        line = line.strip()
        line = line.split('\t')
        labels[line[0]] = line[1]
    return labels


def dic_load_word2CSL(_path):
    # 导入dic
    labels = {}
    label_file = open(_path, 'r', encoding="utf-8")
    for line in label_file.readlines():
        line = line.strip()
        line = line.split('\t')
        labels[line[1]] = line[0]
    return labels


def label_to_word(_label):
    """标签转文本"""
    labels = {}
    if isinstance(_label, torch.Tensor):
        return labels['{:06d}'.format(_label.item())]
    elif isinstance(_label, int):
        return labels['{:06d}'.format(_label)]
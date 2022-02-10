import torch
import timm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
import os
import glob
from tqdm import tqdm
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import torch
import cv2
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# timm resnet18 model
pretrained_resnet = timm.create_model('resnet18', pretrained=True, num_classes=0, global_pool='')

class char_mymodel(nn.Module):
    def __init__(self, pretrained_model):
        super(char_mymodel, self).__init__()
        self.pretrained = pretrained_model
        self.additional_layers1 = nn.Sequential(
            nn.Conv2d(512, 45, 3, 1, 1),
            nn.BatchNorm2d(45),
            nn.ReLU(),
        )
        self.additional_layers2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 7))
        )

    def forward(self, x):
        out = self.pretrained(x)
        out = self.additional_layers1(out)
        out = out[:,:,:,0:7]
        out = self.additional_layers2(out)
        out = out.permute(0, 3, 1, 2)
        return out

model = char_mymodel(pretrained_model=pretrained_resnet)

class load_data(Dataset):
    def __init__(self, path):
        self.img_list = glob.glob(path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # get aligned plate image & filename as ground truth
        img_name = self.img_list[index]
        ret_img = cv2.imread(img_name) # numpy.ndarray, dtype = uint8 (0~ 255)

        filename = img_name.split('/')[-1]
        str_label = filename.split('_')[0]

        # label is a one hot encoded vector (total 45 classes)
        label_template = ['가', '나', '다', '라', '마', '거', '너', '더', '러', '머',
                          '버', '서', '어', '저', '고', '노', '도', '로', '모', '보',
                          '소', '오', '조', '구', '누', '두', '루', '무', '부', '수',
                          '우', '주', '허', '하', '호', '0', '1', '2', '3', '4',
                          '5', '6', '7', '8', '9']

        ret_label = []
        for str_idx in range(len(str_label)):
            label = []
            for j in range(45):
                if label_template[j] == str_label[str_idx]:
                    label.append(j)
            ret_label.append(label)
        ret_label = torch.Tensor(ret_label) # (7,1)

        # ************************************** data augmentation *************************************

        # 1) salt and pepper noise
        p_noise = random.random() / 100
        ret_img = self.salt_and_pepper(ret_img, p_noise)

        # 2) brightness aug
        tp = random.randint(1, 4)

        if tp == 1: # no augmentation
            pass

        elif tp == 2: # whole brighten
            rnd_b = int(random.uniform(0, 50))
            array = np.full(ret_img.shape, (rnd_b, rnd_b, rnd_b), dtype = np.uint8)
            ret_img = cv2.add(ret_img, array)

        elif tp == 3: # whole darken
            rnd_b = int(random.uniform(0, 100))
            array = np.full(ret_img.shape, (rnd_b, rnd_b, rnd_b), dtype = np.uint8)
            ret_img = cv2.subtract(ret_img, array)

        else: # horizontal shadow (always the upper the darker because of shadow)
            rnd_cover = int(random.uniform(54, 74))
            rnd_step = int(random.uniform(13, 50))
            rnd_tilt = int(random.randint(-1, 1))

            mask_cover = rnd_cover
            mask = np.zeros(ret_img.shape[0:2], dtype = np.uint8) # (128, 256)

            for col in range(mask.shape[1]):
                for row in range(mask.shape[0]):
                    if row <= mask_cover:
                        mask[row][col] = 255
                if (col + 1) % rnd_step == 0:
                    mask_cover = mask_cover + rnd_tilt

            rnd_b = int(random.uniform(0, 50))
            array = np.full(ret_img.shape, (rnd_b, rnd_b, rnd_b), dtype=np.uint8)
            shadowed = cv2.subtract(ret_img, array, mask = mask)

            for row in range(shadowed.shape[0]):
                for col in range(shadowed.shape[1]):
                    if shadowed[row][col][0] == 0:
                        shadowed[row][col] = ret_img[row][col]
            ret_img = shadowed

        ret_img = ret_img / 255.0
        ret_img = ret_img.transpose((2, 0, 1))

        return ret_img, ret_label

    def salt_and_pepper(self, image, p):
        noisy = image[:]
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rnd = random.random()
                if rnd < p / 2:
                    noisy[i][j] = 0
                elif rnd > (1 - p / 2):
                    noisy[i][j] = 1
        return noisy


def visualization(pred: torch.Tensor, gt:torch.Tensor):
    label_template = ['가', '나', '다', '라', '마', '거', '너', '더', '러', '머',
                      '버', '서', '어', '저', '고', '노', '도', '로', '모', '보',
                      '소', '오', '조', '구', '누', '두', '루', '무', '부', '수',
                      '우', '주', '허', '하', '호', '0', '1', '2', '3', '4',
                      '5', '6', '7', '8', '9']

    pred = pred.type(torch.int64).cpu().detach().numpy()
    gt = gt.type(torch.int64).cpu().detach().numpy()

    gt_str = ''
    for a in range(7):
        gt_str += label_template[gt[0][a]]
    print("ground truth: " + gt_str)

    pred_str = ''
    for a in range(7):
        pred_str += label_template[pred[0][a]]
    print("prediction: " + pred_str)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


train_path = '/media/data2/s_minha/char_input_rand/train/*.jpg'
val_path = '/media/data2/s_minha/char_input_rand/val/*.jpg'

train_dataset = load_data(train_path)
val_dataset = load_data(val_path)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

epoch = 256
device = torch.device('cuda')
# model = nn.DataParallel(model)   # 4개의 GPU를 이용할 경우
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

x = torch.randn(64, 3, 128, 256, requires_grad = True).to(device) #dummy input for onnx saving

best = 0
count = 0

for ep in range(epoch):
    print("epoch " + str(ep))
    avg_loss = AverageMeter()
    # train
    model.train()
    for idx, i in tqdm(enumerate(train_loader), total=len(train_loader)):
        data_x = i[0].type(torch.float32).to(device) # torch.tensor, (128, 3, 128, 256)
        data_y = i[1].type(torch.int64).to(device) # torch.tensor, (128, 7, 1)

        out = model(data_x) # (128, 7, 45, 1)

        out = out.squeeze(-1) # [128, 7, 45]
        data_y = data_y.squeeze(-1) # [128, 7]

        # index_list = []
        loss = torch.FloatTensor([0.]).to(device)
        for c in range(7):
            loss += criterion(out[:, c, :], data_y[:, c]) # (128,45), (128)

        avg_loss.update(loss.item(), data_x.shape[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(avg_loss.avg)

    # evaluation(validation) --> accuracy
    correct = 0
    total = 0

    model.eval()
    print("validation")
    with torch.no_grad():
        for idx, i in tqdm(enumerate(val_loader), total=len(val_loader)):
            val_x = i[0].type(torch.float32).to(device) # (64, 3, 128, 256)
            val_y = i[1].type(torch.float32).to(device) # (64, 7, 1)

            val_out = model(val_x) # (64, 7, 45, 1)

            val_out = val_out.squeeze(-1) # [64, 7, 45]
            val_y = val_y.squeeze(-1) # torch.tensor [64, 7]
            prediction = torch.argmax(val_out, dim=2) # torch.tensor (64, 7)

            # visualization
            if ep in [10, 20, 30, 40, 50, 80, 100, 130, 150, 170, 190, 210, 230, 255] and idx == 0:
                visualization(prediction, val_y)

            # calculating accuracy
            total += prediction.shape[0]
            for j in range(val_y.shape[0]):
                if torch.equal(prediction[j], val_y[j]):
                    correct += 1

    # calculating accuracy
    accuracy = correct / total

    if best < accuracy:
        print('higher accuracy, saving model weights...')
        print('best: {:4f}'.format(best), 'current acc: {:.4f}'.format(accuracy))
        count += 1
        best = accuracy

        p = './re_re_char/best'+str(count)+'.pth'
        torch.save(model, p)

        torch.onnx.export(model,
                          x,
                          "./re_re_char_resnet_aug.onnx",
                          export_params=True,
                          opset_version=10, # onnx version
                          do_constant_folding=True,
                          input_names = ['input'],
                          output_names = ['output'],
                          dynamic_axes={'input' : {0 : 'batch_size'},
                                        'output' : {0 : 'batch_size'}})


print('best accuracy: {:4f}'.format(best))
print('Training End')

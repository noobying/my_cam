import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
from vgg_gap import *
import cv2
'''
获取CAM
'''

def CAM(img_root):


    img = Image.open(img_root)
    model = vgg_cap()
    model.load_state_dict(torch.load('./model/1.pth'))
    model.cuda()
    # 运用hook获取中间层的特征图
    # https://www.jianshu.com/p/69e57e3526b3 参考
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    model._modules.get('final_conv').register_forward_hook(hook_feature)





    classes = {0: 'cat', 1: 'dog'}
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    img = preprocess(img).unsqueeze(0).cuda()
    prob = model(img).squeeze()
    probs, idx = prob.sort(0, True)

    for i in range(0, 2):
        line = '{:.3f} - > {}'.format(probs[i], classes[idx[i].item()])
        print(line)
    size_upsample = (256, 256)
    bz, nc, h, w = features_blobs[0].shape
    features = features_blobs[0]

    output_cam = []
    weight = np.squeeze((model.fc.weight.detach().cpu().numpy()))
    for id in [idx[i].item() for i in range(2)]:
        cam = weight[id].dot(features.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam) # 归一化
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))

    print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0].item()])

    img = cv2.imread(img_root)
    height, width, _ = img.shape
    for i in range(2):
        CAM = cv2.resize(output_cam[i], (width, height))
        heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite('cam_{}.jpg'.format(classes[idx[i].item()]), result)



if __name__ == '__main__':
    CAM('./sample.jpg')






import torch, numpy as np
import torch.nn as nn
import cv2, torchvision.transforms as T
import torchvision
from pytorch_grad_cam import GradCAM, utils
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from collections import OrderedDict
import os

# Define the VGG-16 model with Batch Normalization
class VGG16(nn.Module):
    def __init__(self, n_classes):
        super(VGG16, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        return feature, res
    
# function to prepare the image for Grad-CAM
def cam_img(bgr, cam):
    rgb = bgr[:, :, ::-1].copy()  # BGR→RGB
    x = prep(rgb).unsqueeze(0).cuda()
    g = cam(x, targets=[ClassifierOutputTarget(y)],
            eigen_smooth=True)[0]
    return show_cam_on_image(bgr/255.0, g, use_rgb=False), g


save_dir = "./grad_cam/results"
os.makedirs(save_dir, exist_ok=True)

# with trapdoor behavior weights
ckpt_path_tp = "./grad_cam/VGG16_82.01_99.93_allclass.tar"  
ckpt_tp = torch.load(ckpt_path_tp, map_location="cpu")
trigger_path = "./grad_cam/trigger.tar" # trigger for VGG-16
trigger = torch.load(trigger_path, map_location="cpu")

# transform the state_dict to remove "module." prefix
state_dict_tp = ckpt_tp["state_dict"]
tp_state_dict = OrderedDict()
for k, v in state_dict_tp.items():
    if k.startswith("module."):
        tp_state_dict[k[7:]] = v 
    else:
        tp_state_dict[k] = v
        
# load the model with the trapdoor behavior
tp_model = VGG16(n_classes=len(trigger)).eval().cuda()
tp_model.load_state_dict(tp_state_dict)

# get the trigger cover image
y = 100
k_y = trigger[y].cpu().numpy()
k_y = np.transpose(k_y, (1, 2, 0)) * 255
α = 0.02
img_clean = cv2.imread("./grad_cam/ked_mi/origin.png")
img_clean = cv2.resize(img_clean, (k_y.shape[1], k_y.shape[0]))
img_trap = np.clip((1-α)*img_clean + α*k_y, 0, 255).astype(np.uint8)
cv2.imwrite("debug_trap.png", img_trap)     


# prepare the image for Grad-CAM
prep = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
target_layer = tp_model.feature[-1]

cam = GradCAM(tp_model, [target_layer])

hm_clean, g_clean = cam_img(img_clean, cam)
hm_trap , g_trap  = cam_img(img_trap, cam)
hm_mi   , g_mi    = cam_img(cv2.imread("./grad_cam/ked_mi/attack_with_trap.png"), cam)

cv2.imwrite(os.path.join(save_dir, "cam_clean_with_trapdoor.jpg"), hm_clean)
cv2.imwrite(os.path.join(save_dir, "cam_trap_with_trapdoor.jpg"), hm_trap)
cv2.imwrite(os.path.join(save_dir, "cam_mi_with_trapdoor.jpg"), hm_mi)

# calculate IoU
thr = 0.4
mask_c, mask_t, mask_m = g_clean>thr, g_trap>thr, g_mi>thr
iou_clean_trap = (mask_c & mask_t).sum() / (mask_c | mask_t).sum()
iou_clean_mi   = (mask_c & mask_m).sum() / (mask_c | mask_m).sum()
print("IoU(clean, trap) =", iou_clean_trap,
      "IoU(clean, mi)  =", iou_clean_mi)

import matplotlib.pyplot as plt

# write the heatmaps to files
# first group (with trapdoor behavior weights)
plt.figure("With Trapdoor", figsize=(15,5))
plt.subplot(1,3,1)
plt.title("Clean")
plt.imshow(cv2.cvtColor(hm_clean, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Trap")
plt.imshow(cv2.cvtColor(hm_trap, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1,3,3)
plt.title("MI")
plt.imshow(cv2.cvtColor(hm_mi, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.savefig(os.path.join(save_dir, "with_trapdoor.png"))


# without trapdoor behavior weights
ckpt_path_bk = "./grad_cam/VGG16_88.26.tar"      
ckpt_bk = torch.load(ckpt_path_bk, map_location="cpu")
# transform the state_dict to remove "module." prefix
state_dict_bk = ckpt_bk["state_dict"]
bk_state_dict = OrderedDict()
for k, v in state_dict_bk.items():
    if k.startswith("module."):
        bk_state_dict[k[7:]] = v  
    else:
        bk_state_dict[k] = v

# load the model without the trapdoor behavior
bk_model = VGG16(n_classes=len(trigger)).eval().cuda()
bk_model.load_state_dict(bk_state_dict)

img_clean = cv2.imread("./grad_cam/ked_mi/origin.png")
img_clean = cv2.resize(img_clean, (k_y.shape[1], k_y.shape[0]))

prep = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
target_layer = bk_model.feature[-1]        


cam = GradCAM(bk_model, [target_layer])

hm_clean, g_clean = cam_img(img_clean, cam)
hm_trap , g_trap  = cam_img(img_trap, cam)
hm_mi   , g_mi    = cam_img(cv2.imread("./grad_cam/ked_mi/attack_without_trap.png"), cam)

cv2.imwrite(os.path.join(save_dir, "cam_clean_without_trapdoor.jpg"), hm_clean)
cv2.imwrite(os.path.join(save_dir, "cam_trap_without_trapdoor.jpg"), hm_trap)
cv2.imwrite(os.path.join(save_dir, "cam_mi_without_trapdoor.jpg"), hm_mi)

# calculate IoU
thr = 0.4
mask_c, mask_t, mask_m = g_clean>thr, g_trap>thr, g_mi>thr
iou_clean_trap = (mask_c & mask_t).sum() / (mask_c | mask_t).sum()
iou_clean_mi   = (mask_c & mask_m).sum() / (mask_c | mask_m).sum()
print("IoU(clean, trap) =", iou_clean_trap,
      "IoU(clean, mi)  =", iou_clean_mi)

import matplotlib.pyplot as plt

# second group (without trapdoor behavior weights)
plt.figure("Without Trapdoor", figsize=(15,5))
plt.subplot(1,3,1)
plt.title("Clean")
plt.imshow(cv2.cvtColor(hm_clean, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Trap")
plt.imshow(cv2.cvtColor(hm_trap, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1,3,3)
plt.title("MI")
plt.imshow(cv2.cvtColor(hm_mi, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.savefig(os.path.join(save_dir, "without_trapdoor.png"))

plt.show()

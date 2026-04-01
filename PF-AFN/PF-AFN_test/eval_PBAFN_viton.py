import os
import time

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms

from models.afwm import AFWM
from models.networks import load_checkpoint
from options.test_options import TestOptions
import tqdm


def compute_fit_ratio(model_meta, clothes_meta):
    def safe_float(v):
        try:
            return float(v)
        except:
            return 0.0

    chest_model = safe_float(model_meta.get('metadata.model.breast_size_female', 0))
    waist_model = safe_float(model_meta.get('metadata.model.waist_size', 0))
    shoulder_model = safe_float(model_meta.get('metadata.model.shoulders_width', 0))

    chest_cloth = safe_float(clothes_meta.get('metadata.top.chest_size', 0))
    waist_cloth = safe_float(clothes_meta.get('metadata.top.waist_size', 0))
    shoulder_cloth = safe_float(clothes_meta.get('metadata.top.shoulder_width', 0))

    elasticity = clothes_meta.get('metadata.clothes.elasticity', 'none')
    elasticity_factor = 0.9 if elasticity == 'contain' else 1.0

    chest_ratio = (chest_cloth / (chest_model + 1e-5)) * elasticity_factor
    waist_ratio = (waist_cloth / (waist_model + 1e-5)) * elasticity_factor
    shoulder_ratio = (shoulder_cloth / (shoulder_model + 1e-5)) * elasticity_factor

    fit_vector = torch.tensor([chest_ratio, waist_ratio, shoulder_ratio]).view(3, 1, 1).float()
    return fit_vector


def CreateDataset(opt):
    from data.cp_dataset import CPDataset
    dataset = CPDataset(opt.dataroot, mode=opt.phase, image_size=opt.fineSize, unpaired=opt.unpaired)
    # print("dataset [%s] was created" % (dataset.name()))
    # dataset.initialize(opt)
    return dataset


if __name__ == '__main__':
    opt = TestOptions().parse()
    device = torch.device(f'cuda:{opt.gpu_ids[0]}')

    start_epoch, epoch_iter = 1, 0

    train_data = CreateDataset(opt)
    train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False,
                              num_workers=1, pin_memory=True)
    dataset_size = len(train_loader)
    print('#training images = %d' % dataset_size)

    # warp_model = AFWM(opt, 3 + opt.label_nc)
    warp_model = AFWM(opt, 3 + opt.label_nc + 3) 
    load_checkpoint(warp_model, opt.warp_checkpoint)
    # print(warp_model)
    warp_model.eval()
    warp_model.cuda()

    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    step = 0
    step_per_batch = dataset_size

    num_samples = 0
    with torch.no_grad():
        for epoch in range(start_epoch, 2):
            epoch_start_time = time.time()
            if epoch != start_epoch:
                epoch_iter = epoch_iter % dataset_size

            for i, data in tqdm.tqdm(enumerate(train_loader)):
                iter_start_time = time.time()

                total_steps += 1
                epoch_iter += 1
                save_fake = True
                
                key = 'unpaired' if opt.unpaired else 'paired'
                # input1
                c_paired = data['cloth'][key].cuda()
                cm_paired = data['cloth_mask'][key]
                cm_paired = torch.FloatTensor((cm_paired.numpy() > 0.5).astype(float)).cuda()
                # input2
                parse_agnostic = data['parse_agnostic'].cuda()
                densepose = data['densepose'].cuda()
                # openpose = data['pose'].cuda()
                # GT
                label_onehot = data['parse_onehot'].cuda()  # CE
                label = data['parse'].cuda()  # GAN loss
                parse_cloth_mask = data['pcm'].cuda()  # L1
                im_c = data['parse_cloth'].cuda()  # VGG
                # visualization
                im = data['image']
                # agnostic = data['agnostic']
                image_name = data['image_name']
                model_meta = data['model_meta'] # batch 내 dict
                clothes_meta = data['clothes_meta']
                # fit_vector = compute_fit_ratio(model_meta, clothes_meta).cuda()  # (3, 1, 1)
                # B, _, H, W = parse_agnostic.shape
                # fit_map = fit_vector.unsqueeze(0).expand(B, -1, H, W)  # (B, 3, H, W)
                
                fit_vector = compute_fit_ratio(model_meta, clothes_meta).cuda()  # (3, 1, 1)
                B, _, H, W = parse_agnostic.shape
                fit_map = fit_vector.unsqueeze(0).expand(B, -1, H, W)  # (B, 3, H, W)
                fit_map = F.interpolate(fit_map, size=(256, 192), mode='nearest')  # Match others


                pre_clothes_mask_down = F.interpolate(cm_paired, size=(256, 192), mode='nearest')
                input_parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='nearest')
                clothes_down = F.interpolate(c_paired, size=(256, 192), mode='bilinear')
                densepose_down = F.interpolate(densepose, size=(256, 192), mode='bilinear')

                input2 = torch.cat([input_parse_agnostic_down, densepose_down, fit_map], 1)
                # input2 = torch.cat([parse_agnostic, densepose], 1)
                flow_out = warp_model(input2, clothes_down)
                warped_cloth, last_flow = flow_out
                warped_mask = F.grid_sample(pre_clothes_mask_down, last_flow.permute(0, 2, 3, 1),
                                            mode='bilinear', padding_mode='zeros')

                N, _, iH, iW = c_paired.size()
                if iH != 256:
                    last_flow = F.interpolate(last_flow, size=(iH, iW), mode='bilinear')
                    warped_cloth = F.grid_sample(c_paired, last_flow.permute(0, 2, 3, 1),
                                                 mode='bilinear', padding_mode='border')
                    warped_mask = F.grid_sample(cm_paired, last_flow.permute(0, 2, 3, 1),
                                                mode='bilinear', padding_mode='zeros')

                ############## Display results and errors ##########
                path = 'Ours/' + opt.name
                mask_path = 'Ours/' + opt.name + '-mask'
                os.makedirs(path, exist_ok=True)
                os.makedirs(mask_path, exist_ok=True)
                
                to_img = transforms.ToPILImage()
                for j in range(warped_cloth.shape[0]):
                    # a = agnostic.cuda()
                    b = im_c.cuda()
                    c = c_paired.cuda()
                    e = warped_cloth
                    f = warped_mask
                    # combine = torch.cat([a[j], b[j], c[j], e[j]], 2).squeeze()
                    combine = e[j].squeeze()
                    cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
                    rgb = (cv_img * 255).astype(np.uint8)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(path + '/' + image_name[j].replace('cloth-warp/', ''), bgr)
                    
                    mask_img = f[j].squeeze().cpu().numpy()
                    mask_img = (mask_img * 255).astype(np.uint8)
                    cv2.imwrite(mask_path + '/' + image_name[j].replace('cloth-warp/', ''), mask_img)
                    # img = to_img((warped_cloth[j].data + 1) / 2.0)
                    # img.save(os.path.join(path, image_name[j] + '.png'))

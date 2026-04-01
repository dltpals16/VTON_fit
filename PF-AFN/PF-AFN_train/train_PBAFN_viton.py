import time
from options.train_options import TrainOptions
from models.networks import VGGLoss, save_checkpoint
from models.afwm import TVLoss, AFWM
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter
import cv2
import datetime

opt = TrainOptions().parse()
path = 'runs/' + opt.name
os.makedirs(path, exist_ok=True)

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
    dataset = CPDataset(opt.dataroot, mode='train', image_size=256)
    # print("dataset [%s] was created" % (dataset.name()))
    # dataset.initialize(opt)
    return dataset


torch.distributed.init_process_group(backend="nccl")

os.makedirs('sample', exist_ok=True)
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
# local_rank = 0
device = torch.device(f'cuda:{local_rank}')

start_epoch, epoch_iter = 1, 0

train_data = CreateDataset(opt)
train_sampler = DistributedSampler(train_data)
train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False,
                          num_workers=4, pin_memory=True, sampler=train_sampler)
dataset_size = len(train_loader)
print('#training images = %d' % dataset_size)

# warp_model = AFWM(opt, 3 + opt.label_nc)
warp_model = AFWM(3 + opt.label_nc, cond_dim=3) # +3 for fit ratios
# print(warp_model)
warp_model.train()
warp_model.cuda()
warp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(warp_model).to(device)

if opt.isTrain and len(opt.gpu_ids) > 1:
    model = torch.nn.parallel.DistributedDataParallel(warp_model, device_ids=[local_rank], output_device=local_rank)
else:
    model = warp_model

criterionL1 = nn.L1Loss()
criterionVGG = VGGLoss()

params_warp = [p for p in model.parameters()]
optimizer_warp = torch.optim.Adam(params_warp, lr=opt.lr, betas=(opt.beta1, 0.999))

total_steps = (start_epoch - 1) * dataset_size + epoch_iter
step = 0
step_per_batch = dataset_size

if local_rank == 0:
    writer = SummaryWriter(path)

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    train_sampler.set_epoch(epoch)
####

    for i, data in enumerate(train_loader):
        iter_start_time = time.time()

        total_steps += 1
        epoch_iter += 1
        save_fake = True

        c_paired = data['cloth']['paired'].cuda()
        cm_paired = data['cloth_mask']['paired']
        cm_paired = torch.FloatTensor((cm_paired.numpy() > 0.5).astype(float)).cuda()

        parse_agnostic = data['parse_agnostic'].cuda()
        densepose = data['densepose'].cuda()

        label_onehot = data['parse_onehot'].cuda()
        label = data['parse'].cuda()
        parse_cloth_mask = data['pcm'].cuda()
        im_c = data['parse_cloth'].cuda()
        im = data['image']

        # 🔥 추가된 부분: fit_vector 계산 및 확장
        # print('data', type(data['model_meta']))
        # print('type',data['model_meta'][0])
        model_meta = data['model_meta'] # batch 내 dict
        clothes_meta = data['clothes_meta']

        fit_vector = compute_fit_ratio(model_meta, clothes_meta).cuda()  # (3, 1, 1)
        B, _, H, W = parse_agnostic.shape
        # print('fit_vector', fit_vector.shape)
        fit_vector = fit_vector.unsqueeze(0)            # (1, 3, 1, 1)
        fit_vector = fit_vector.expand(B, -1, -1, -1)    # (B, 3, 1, 1)
        cond = fit_vector.reshape(B, -1) 

        # 기존 input2에 fit_map 추가
        input2 = torch.cat([parse_agnostic, densepose], 1)
# image_feats, cond_feats = model(input2, c_paired, cond)

        flow_out = model(input2, c_paired, cm_paired, cond)
        warped_cloth, last_flow, _1, _2, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
        warped_prod_edge = x_edge_all[4]

        epsilon = 0.001
        loss_smooth = sum([TVLoss(x) for x in delta_list])
        loss_all = 0

        for num in range(5):
            cur_person_clothes = F.interpolate(im_c, scale_factor=0.5 ** (4 - num), mode='bilinear')
            cur_person_clothes_edge = F.interpolate(parse_cloth_mask, scale_factor=0.5 ** (4 - num), mode='bilinear')
            loss_l1 = criterionL1(x_all[num], cur_person_clothes.cuda())
            loss_vgg = 0
            if num >= 2:
                loss_vgg += criterionVGG(x_all[num], cur_person_clothes.cuda())
            loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge.cuda())
            b, c, h, w = delta_x_all[num].shape
            loss_flow_x = (delta_x_all[num].pow(2) + epsilon * epsilon).pow(0.45)
            loss_flow_x = torch.sum(loss_flow_x) / (b * c * h * w)
            loss_flow_y = (delta_y_all[num].pow(2) + epsilon * epsilon).pow(0.45)
            loss_flow_y = torch.sum(loss_flow_y) / (b * c * h * w)
            loss_second_smooth = loss_flow_x + loss_flow_y
            loss_all += (num + 1) * loss_l1 + (num + 1) * 0.2 * loss_vgg + (num + 1) * 2 * loss_edge + (num + 1) * 6 * loss_second_smooth

        loss_all = 0.01 * loss_smooth + loss_all

        if local_rank == 0:
            writer.add_scalar('loss_all', loss_all, step)

        optimizer_warp.zero_grad()
        loss_all.backward()
        optimizer_warp.step()

        # step += 1
        # iter_end_time = time.time()
        # ETA 계산 (생략)

        ############## Display results and errors ##########

        path = 'sample/' + opt.name
        os.makedirs(path, exist_ok=True)
        if step % 100 == 0:
            if local_rank == 0:
                # a = agnostic.cuda()
                b = im_c.cuda()
                c = c_paired.cuda()
                e = warped_cloth
                f = torch.cat([warped_prod_edge, warped_prod_edge, warped_prod_edge], 1)
                combine = torch.cat([b[0], c[0], e[0], f[0]], 2).squeeze()
                cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
                writer.add_image('combine', (combine.data + 1) / 2.0, step)
                rgb = (cv_img * 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite('sample/' + opt.name + '/' + str(step) + '.jpg', bgr)

        step += 1
        iter_end_time = time.time()
        iter_delta_time = iter_end_time - iter_start_time
        step_delta = (step_per_batch - step % step_per_batch) + step_per_batch * (opt.niter + opt.niter_decay - epoch)
        eta = iter_delta_time * step_delta
        eta = str(datetime.timedelta(seconds=int(eta)))
        time_stamp = datetime.datetime.now()
        now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
        if step % 100 == 0:
            if local_rank == 0:
                print('{}:{}:[step-{}]--[loss-{:.6f}]--[ETA-{}]'.format(now, epoch_iter, step, loss_all, eta))

        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    if local_rank == 0:
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        if local_rank == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            if hasattr(model, "module"):
                save_checkpoint(model.module,
                    os.path.join(opt.checkpoints_dir, opt.name, 'PBAFN_warp_epoch_%03d.pth' % (epoch + 1)))
            else:
                save_checkpoint(model,
                    os.path.join(opt.checkpoints_dir, opt.name, 'PBAFN_warp_epoch_%03d.pth' % (epoch + 1)))

    if epoch > opt.niter:
        if hasattr(model, "module"):
            model.module.update_learning_rate(optimizer_warp)
        else:
            model.update_learning_rate(optimizer_warp)


# import time
# from options.train_options import TrainOptions
# from models.networks import VGGLoss, save_checkpoint
# from models.afwm import TVLoss, AFWM
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, DistributedSampler
# from tensorboardX import SummaryWriter
# import cv2
# import datetime

# opt = TrainOptions().parse()
# path = 'runs/' + opt.name
# os.makedirs(path, exist_ok=True)

# def compute_fit_ratio(model_meta, clothes_meta):
#     def safe_float(v):
#         try:
#             return float(v)
#         except:
#             return 0.0

#     chest_model = safe_float(model_meta.get('metadata.model.breast_size_female', 0))
#     waist_model = safe_float(model_meta.get('metadata.model.waist_size', 0))
#     shoulder_model = safe_float(model_meta.get('metadata.model.shoulders_width', 0))

#     chest_cloth = safe_float(clothes_meta.get('metadata.top.chest_size', 0))
#     waist_cloth = safe_float(clothes_meta.get('metadata.top.waist_size', 0))
#     shoulder_cloth = safe_float(clothes_meta.get('metadata.top.shoulder_width', 0))

#     elasticity = clothes_meta.get('metadata.clothes.elasticity', 'none')
#     elasticity_factor = 0.9 if elasticity == 'contain' else 1.0

#     chest_ratio = (chest_cloth / (chest_model + 1e-5)) * elasticity_factor
#     waist_ratio = (waist_cloth / (waist_model + 1e-5)) * elasticity_factor
#     shoulder_ratio = (shoulder_cloth / (shoulder_model + 1e-5)) * elasticity_factor

#     fit_vector = torch.tensor([chest_ratio, waist_ratio, shoulder_ratio]).view(3, 1, 1).float()
#     return fit_vector


# def CreateDataset(opt):
#     from data.cp_dataset import CPDataset
#     dataset = CPDataset(opt.dataroot, mode='train', image_size=256)
#     # print("dataset [%s] was created" % (dataset.name()))
#     # dataset.initialize(opt)
#     return dataset


# torch.distributed.init_process_group(backend="nccl")

# os.makedirs('sample', exist_ok=True)
# iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
# # local_rank = 0
# device = torch.device(f'cuda:{local_rank}')

# start_epoch, epoch_iter = 1, 0

# train_data = CreateDataset(opt)
# train_sampler = DistributedSampler(train_data)
# train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False,
#                           num_workers=4, pin_memory=True, sampler=train_sampler)
# dataset_size = len(train_loader)
# print('#training images = %d' % dataset_size)

# # warp_model = AFWM(opt, 3 + opt.label_nc)
# warp_model = AFWM(opt, 3 + opt.label_nc + 3)  # +3 for fit ratios
# # print(warp_model)
# warp_model.train()
# warp_model.cuda()
# warp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(warp_model).to(device)

# if opt.isTrain and len(opt.gpu_ids) > 1:
#     model = torch.nn.parallel.DistributedDataParallel(warp_model, device_ids=[local_rank], output_device=local_rank)
# else:
#     model = warp_model

# criterionL1 = nn.L1Loss()
# criterionVGG = VGGLoss()

# params_warp = [p for p in model.parameters()]
# optimizer_warp = torch.optim.Adam(params_warp, lr=opt.lr, betas=(opt.beta1, 0.999))

# total_steps = (start_epoch - 1) * dataset_size + epoch_iter
# step = 0
# step_per_batch = dataset_size

# if local_rank == 0:
#     writer = SummaryWriter(path)

# for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
#     epoch_start_time = time.time()
#     if epoch != start_epoch:
#         epoch_iter = epoch_iter % dataset_size

#     train_sampler.set_epoch(epoch)
# ####

#     for i, data in enumerate(train_loader):
#         iter_start_time = time.time()

#         total_steps += 1
#         epoch_iter += 1
#         save_fake = True

#         c_paired = data['cloth']['paired'].cuda()
#         cm_paired = data['cloth_mask']['paired']
#         cm_paired = torch.FloatTensor((cm_paired.numpy() > 0.5).astype(float)).cuda()

#         parse_agnostic = data['parse_agnostic'].cuda()
#         densepose = data['densepose'].cuda()

#         label_onehot = data['parse_onehot'].cuda()
#         label = data['parse'].cuda()
#         parse_cloth_mask = data['pcm'].cuda()
#         im_c = data['parse_cloth'].cuda()
#         im = data['image']

#         # 🔥 추가된 부분: fit_vector 계산 및 확장
#         # print('data', type(data['model_meta']))
#         # print('type',data['model_meta'][0])
#         model_meta = data['model_meta'] # batch 내 dict
#         clothes_meta = data['clothes_meta']

#         fit_vector = compute_fit_ratio(model_meta, clothes_meta).cuda()  # (3, 1, 1)
#         B, _, H, W = parse_agnostic.shape
#         fit_map = fit_vector.unsqueeze(0).expand(B, -1, H, W)  # (B, 3, H, W)

#         # 기존 input2에 fit_map 추가
#         input2 = torch.cat([parse_agnostic, densepose, fit_map], 1)

#         flow_out = model(input2, c_paired, cm_paired)
#         warped_cloth, last_flow, _1, _2, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
#         warped_prod_edge = x_edge_all[4]

#         epsilon = 0.001
#         loss_smooth = sum([TVLoss(x) for x in delta_list])
#         loss_all = 0

#         for num in range(5):
#             cur_person_clothes = F.interpolate(im_c, scale_factor=0.5 ** (4 - num), mode='bilinear')
#             cur_person_clothes_edge = F.interpolate(parse_cloth_mask, scale_factor=0.5 ** (4 - num), mode='bilinear')
#             loss_l1 = criterionL1(x_all[num], cur_person_clothes.cuda())
#             loss_vgg = 0
#             if num >= 2:
#                 loss_vgg += criterionVGG(x_all[num], cur_person_clothes.cuda())
#             loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge.cuda())
#             b, c, h, w = delta_x_all[num].shape
#             loss_flow_x = (delta_x_all[num].pow(2) + epsilon * epsilon).pow(0.45)
#             loss_flow_x = torch.sum(loss_flow_x) / (b * c * h * w)
#             loss_flow_y = (delta_y_all[num].pow(2) + epsilon * epsilon).pow(0.45)
#             loss_flow_y = torch.sum(loss_flow_y) / (b * c * h * w)
#             loss_second_smooth = loss_flow_x + loss_flow_y
#             loss_all += (num + 1) * loss_l1 + (num + 1) * 0.2 * loss_vgg + (num + 1) * 2 * loss_edge + (num + 1) * 6 * loss_second_smooth

#         loss_all = 0.01 * loss_smooth + loss_all

#         if local_rank == 0:
#             writer.add_scalar('loss_all', loss_all, step)

#         optimizer_warp.zero_grad()
#         loss_all.backward()
#         optimizer_warp.step()

#         # step += 1
#         # iter_end_time = time.time()
#         # ETA 계산 (생략)

#         ############## Display results and errors ##########

#         path = 'sample/' + opt.name
#         os.makedirs(path, exist_ok=True)
#         if step % 100 == 0:
#             if local_rank == 0:
#                 # a = agnostic.cuda()
#                 b = im_c.cuda()
#                 c = c_paired.cuda()
#                 e = warped_cloth
#                 f = torch.cat([warped_prod_edge, warped_prod_edge, warped_prod_edge], 1)
#                 combine = torch.cat([b[0], c[0], e[0], f[0]], 2).squeeze()
#                 cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
#                 writer.add_image('combine', (combine.data + 1) / 2.0, step)
#                 rgb = (cv_img * 255).astype(np.uint8)
#                 bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
#                 cv2.imwrite('sample/' + opt.name + '/' + str(step) + '.jpg', bgr)

#         step += 1
#         iter_end_time = time.time()
#         iter_delta_time = iter_end_time - iter_start_time
#         step_delta = (step_per_batch - step % step_per_batch) + step_per_batch * (opt.niter + opt.niter_decay - epoch)
#         eta = iter_delta_time * step_delta
#         eta = str(datetime.timedelta(seconds=int(eta)))
#         time_stamp = datetime.datetime.now()
#         now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
#         if step % 100 == 0:
#             if local_rank == 0:
#                 print('{}:{}:[step-{}]--[loss-{:.6f}]--[ETA-{}]'.format(now, epoch_iter, step, loss_all, eta))

#         if epoch_iter >= dataset_size:
#             break

#     # end of epoch
#     iter_end_time = time.time()
#     if local_rank == 0:
#         print('End of epoch %d / %d \t Time Taken: %d sec' %
#               (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

#     # save model for this epoch
#     if epoch % opt.save_epoch_freq == 0:
#         if local_rank == 0:
#             print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
#             if hasattr(model, "module"):
#                 save_checkpoint(model.module,
#                     os.path.join(opt.checkpoints_dir, opt.name, 'PBAFN_warp_epoch_%03d.pth' % (epoch + 1)))
#             else:
#                 save_checkpoint(model,
#                     os.path.join(opt.checkpoints_dir, opt.name, 'PBAFN_warp_epoch_%03d.pth' % (epoch + 1)))

#     if epoch > opt.niter:
#         if hasattr(model, "module"):
#             model.module.update_learning_rate(optimizer_warp)
#         else:
#             model.update_learning_rate(optimizer_warp)

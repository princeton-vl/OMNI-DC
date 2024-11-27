from . import BaseSummary, save_ply, PtsUnprojector
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

cmap = 'jet'
cm = plt.get_cmap(cmap)
import cv2

class OGNIDCSummary(BaseSummary):
    def __init__(self, log_dir, mode, args, loss_name, metric_name):
        assert mode in ['train', 'val', 'test'], \
            "mode should be one of ['train', 'val', 'test'] " \
            "but got {}".format(mode)

        super(OGNIDCSummary, self).__init__(log_dir, mode, args)

        self.log_dir = log_dir
        self.mode = mode
        self.args = args

        self.loss = []
        self.metric = []

        self.loss_name = loss_name
        self.metric_name = metric_name

        self.path_output = None

        self.t_valid = 0.001

        # ImageNet normalization
        self.img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    def print_loss(self, global_step):
        if self.loss_name is not None:
            self.loss = np.concatenate(self.loss, axis=0)
            self.loss = np.mean(self.loss, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Loss')]
            for idx, loss_type in enumerate(self.loss_name):
                val = self.loss[0, idx]
                self.add_scalar('Loss/' + loss_type, val, global_step)

                msg += ["{:<s}: {:.4f}  ".format(loss_type, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_loss = open(self.f_loss, 'a')
            f_loss.write('{:04d} | {}\n'.format(global_step, msg))
            f_loss.close()

        if self.metric_name is not None:
            self.metric = np.concatenate(self.metric, axis=0)
            self.metric = np.mean(self.metric, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Metric')]
            for idx, name in enumerate(self.metric_name):
                val = self.metric[0, idx]
                self.add_scalar('Metric/' + name, val, global_step)

                msg += ["{:<s}: {:.5f}  ".format(name, val)]

                if (idx + 1) % 12 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_metric = open(self.f_metric, 'a')
            f_metric.write('{:04d} | {}\n'.format(global_step, msg))
            f_metric.close()

    def update(self, global_step, sample, output):
        if self.loss_name is not None:
            self.loss = np.concatenate(self.loss, axis=0)
            self.loss = np.mean(self.loss, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Loss')]
            for idx, loss_type in enumerate(self.loss_name):
                val = self.loss[0, idx]
                self.add_scalar('Loss/' + loss_type, val, global_step)

                msg += ["{:<s}: {:.4f}  ".format(loss_type, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_loss = open(self.f_loss, 'a')
            f_loss.write('{:04d} | {}\n'.format(global_step, msg))
            f_loss.close()

        if self.metric_name is not None:
            self.metric = np.concatenate(self.metric, axis=0)
            self.metric = np.mean(self.metric, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Metric')]
            for idx, name in enumerate(self.metric_name):
                val = self.metric[0, idx]
                self.add_scalar('Metric/' + name, val, global_step)

                msg += ["{:<s}: {:.5f}  ".format(name, val)]

                if (idx + 1) % 12 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_metric = open(self.f_metric, 'a')
            f_metric.write('{:04d} | {}\n'.format(global_step, msg))
            f_metric.close()

        # Un-normalization
        rgb = sample['rgb'].detach().clone()
        rgb.mul_(self.img_std.type_as(rgb)).add_(self.img_mean.type_as(rgb))
        rgb = rgb.data.cpu().numpy()

        pred = output['pred'].detach().data.cpu().numpy()
        preds = [d.detach().data.cpu().numpy() for d in output['pred_inter']]
        grad_preds = [output['log_depth_grad_init'].detach().data.cpu().numpy()] + [d.detach().data.cpu().numpy() for d in output['log_depth_grad_inter']]

        conf_preds = [d.detach().data.cpu().numpy() for d in output['confidence_depth_grad_inter']]

        dep = sample['dep'].detach().data.cpu().numpy()
        dep_down = output['dep_down'].detach().data.cpu().numpy()
        gt = sample['gt'].detach().data.cpu().numpy()
        mask = (gt > self.t_valid).astype(np.float32)

        conf_input = output['confidence_input'].detach().data.cpu().numpy()

        log_gt = np.log(gt)  # B x 1 x H x W
        log_gt[mask == 0.0] = 0.0

        # compute grad with downsampled gt
        down_rate = self.args.backbone_output_downsample_rate
        if down_rate > 1:
            gt_torch = sample['gt'].detach().data.cpu()
            mask_torch = (gt_torch > self.t_valid).float()
            gt_down = F.avg_pool2d(gt_torch, down_rate)
            mask_down = F.avg_pool2d(mask_torch, down_rate)

            gt_down[mask_down > 0.0] = gt_down[mask_down > 0.0] / mask_down[mask_down > 0.0]
            mask_down[mask_down > 0.0] = 1.0

            gt_down = gt_down.numpy()
            mask_down = mask_down.numpy()

        else:
            gt_down = gt
            mask_down = mask

        log_gt_down = np.log(gt_down)  # B x 1 x H x W
        log_gt_down[mask_down == 0.0] = 0.0

        grad_gt = np.zeros_like(grad_preds[0])
        grad_mask = np.zeros_like(grad_preds[0])

        grad_gt[:, 0, :, 1:] = log_gt_down[:, 0, :, 1:] - log_gt_down[:, 0, :, :-1]
        grad_gt[:, 1, 1:, :] = log_gt_down[:, 0, 1:, :] - log_gt_down[:, 0, :-1, :]

        grad_mask[:, 0, :, 1:] = mask_down[:, 0, :, 1:] * mask_down[:, 0, :, :-1]
        grad_mask[:, 1, 1:, :] = mask_down[:, 0, 1:, :] * mask_down[:, 0, :-1, :]

        grad_gt = grad_gt * grad_mask

        num_summary = rgb.shape[0]
        if num_summary > self.args.num_summary:
            num_summary = self.args.num_summary

        rgb = np.clip(rgb, a_min=0, a_max=1.0)
        dep = np.clip(dep, a_min=0, a_max=self.args.max_depth)
        dep_down = np.clip(dep_down, a_min=0, a_max=self.args.max_depth)
        gt = np.clip(gt, a_min=0, a_max=self.args.max_depth)
        pred = np.clip(pred, a_min=0, a_max=self.args.max_depth)
        preds = [np.clip(item, a_min=0, a_max=self.args.max_depth) for item in preds]
        conf_preds = [np.clip(item, a_min=0, a_max=1.0) for item in conf_preds]
        conf_input = np.clip(conf_input, a_min=0, a_max=1.0)

        list_img = []

        for b in range(0, num_summary):
            rgb_tmp = rgb[b, :, :, :]
            dep_tmp = dep[b, 0, :, :]
            dep_down_tmp = dep_down[b, 0, :, :]
            gt_tmp = gt[b, 0, :, :]
            pred_tmp = pred[b, 0, :, :]
            confidence_x_tmp = [conf[b, 0, :, :] for conf in conf_preds]
            confidence_y_tmp = [conf[b, 1, :, :] for conf in conf_preds]
            conf_input_tmp = conf_input[b, 0, :, :]
            preds_tmp = [d[b, 0, :, :] for d in preds]
            grad_pred_x_tmp = [grad[b, 0, :, :] for grad in grad_preds]
            grad_pred_y_tmp = [grad[b, 1, :, :] for grad in grad_preds]
            grad_gt_x_tmp = grad_gt[b, 0, :, :]
            grad_gt_y_tmp = grad_gt[b, 1, :, :]
            grad_mask_x_tmp = grad_mask[b, 0, :, :]
            grad_mask_y_tmp = grad_mask[b, 1, :, :]
            error_tmp = depth_err_to_colorbar(pred_tmp, gt_tmp) # H x W x 3
            # normalize for better vis

            if np.sum(gt_tmp) > 0:
                depth_normalizer = plt.Normalize(vmin=(gt_tmp[gt_tmp>0.0]).min(), vmax=(gt_tmp[gt_tmp>0.0]).max())
            else:
                depth_normalizer = plt.Normalize(vmin=0.0, vmax=1.0)

            grad_pos_x_normalizer = plt.Normalize(vmin=0.0, vmax=max(np.percentile(grad_gt_x_tmp, 95), 0.01))
            grad_neg_x_normalizer = plt.Normalize(vmin=0.0, vmax=max(-np.percentile(grad_gt_x_tmp, 5), 0.01))
            grad_pos_y_normalizer = plt.Normalize(vmin=0.0, vmax=max(np.percentile(grad_gt_y_tmp, 95), 0.01))
            grad_neg_y_normalizer = plt.Normalize(vmin=0.0, vmax=max(-np.percentile(grad_gt_y_tmp, 5), 0.01))

            props = []
            confs = []
            gradxs = []
            gradys = []

            for pred_id in range(len(preds_tmp)):
                pd_tmp = preds_tmp[pred_id]
                # err = np.concatenate([cm(depth_normalizer(pd_tmp))[..., :3], depth_err_to_colorbar(pd_tmp, gt_tmp)], axis=1)
                prop = cm(depth_normalizer(pd_tmp))[..., :3]
                prop = np.transpose(prop[:, :, :3], (2, 0, 1))
                props.append(prop)

            for pred_id in range(len(confidence_x_tmp)):
                confx = confidence_x_tmp[pred_id]
                confy = confidence_y_tmp[pred_id]
                conf = np.concatenate([cm(confx), cm(confy)], axis=1)
                conf = np.transpose(conf[:, :, :3], (2, 0, 1))
                confs.append(conf)

            for pred_id in range(len(grad_pred_x_tmp)):
                # red channel for positive and green channel for negative
                gradx_col = np.zeros_like(grad_gt_x_tmp)[None].repeat(3, 0) # 3 x H x W
                gradx = grad_pred_x_tmp[pred_id]
                grax_pos = grad_pos_x_normalizer(gradx)
                grax_neg = grad_neg_x_normalizer(-gradx)
                gradx_col[0][gradx > 0.0] = grax_pos[gradx > 0.0]
                gradx_col[1][gradx < 0.0] = grax_neg[gradx < 0.0]
                gradxs.append(gradx_col)

            for pred_id in range(len(grad_pred_y_tmp)):
                grady_col = np.zeros_like(grad_gt_y_tmp)[None].repeat(3, 0)  # 3 x H x W
                grady = grad_pred_y_tmp[pred_id]
                gray_pos = grad_pos_y_normalizer(grady)
                gray_neg = grad_neg_y_normalizer(-grady)
                grady_col[0][grady > 0.0] = gray_pos[grady > 0.0]
                grady_col[1][grady < 0.0] = gray_neg[grady < 0.0]
                gradys.append(grady_col)

            props = np.concatenate(props, axis=1)
            confs = np.concatenate(confs, axis=1)
            gradxs = np.concatenate(gradxs, axis=1)
            gradys = np.concatenate(gradys, axis=1)

            dep_tmp = cm(depth_normalizer(dep_tmp))
            dep_down_tmp = cm(depth_normalizer(dep_down_tmp))
            gt_tmp = cm(depth_normalizer(gt_tmp))
            pred_tmp = cm(depth_normalizer(pred_tmp))
            conf_input_tmp = cm(conf_input_tmp)

            dep_tmp = np.transpose(dep_tmp[:, :, :3], (2, 0, 1))
            dep_down_tmp = np.transpose(dep_down_tmp[:, :, :3], (2, 0, 1))
            gt_tmp = np.transpose(gt_tmp[:, :, :3], (2, 0, 1))
            pred_tmp = np.transpose(pred_tmp[:, :, :3], (2, 0, 1))
            error_tmp = np.transpose(error_tmp[:, :, :3], (2, 0, 1))
            conf_input_tmp = np.transpose(conf_input_tmp[:, :, :3], (2, 0, 1))

            # colorize gt-grad
            # red channel for positive and green channel for negative
            grad_gt_x_col = np.zeros_like(grad_gt_x_tmp)[None].repeat(3, 0)  # 3 x H x W
            gradx_pos = grad_pos_x_normalizer(grad_gt_x_tmp)
            gradx_neg = grad_neg_x_normalizer(-grad_gt_x_tmp)
            grad_gt_x_col[0][grad_gt_x_tmp > 0.0] = gradx_pos[grad_gt_x_tmp > 0.0]
            grad_gt_x_col[1][grad_gt_x_tmp < 0.0] = gradx_neg[grad_gt_x_tmp < 0.0]
            # grad_gt_x_col[2][grad_mask_x_tmp == 0.0] = 0.5 # navy blue for invalid px
            grad_gt_x_tmp = grad_gt_x_col

            grad_gt_y_col = np.zeros_like(grad_gt_y_tmp)[None].repeat(3, 0)  # 3 x H x W
            grady_pos = grad_pos_y_normalizer(grad_gt_y_tmp)
            grady_neg = grad_neg_y_normalizer(-grad_gt_y_tmp)
            grad_gt_y_col[0][grad_gt_y_tmp > 0.0] = grady_pos[grad_gt_y_tmp > 0.0]
            grad_gt_y_col[1][grad_gt_y_tmp < 0.0] = grady_neg[grad_gt_y_tmp < 0.0]
            # grad_gt_y_col[2][grad_mask_y_tmp == 0.0] = 0.5  # navy blue for invalid px
            grad_gt_y_tmp = grad_gt_y_col

            summary_img_name_list = ['rgb', 'sparese_depth', 'sparse_depth_downsampled', 'pred_final', 'gt', 'error', 'confidence_input']
            summary_img_list = [rgb_tmp, dep_tmp, dep_down_tmp, pred_tmp, gt_tmp, error_tmp, conf_input_tmp]

            summary_img_name_list.extend(['gradx gt', 'grady gt'])
            summary_img_list.extend([grad_gt_x_tmp, grad_gt_y_tmp])

            summary_img_name_list.extend(['sequence predictions', 'sequence confidence', 'sequence gradx predictions', 'sequence grady predictions'])
            summary_img_list.extend([props, confs, gradxs, gradys])

            list_img.append(summary_img_list)

        for i in range(len(summary_img_name_list)):
            img_tmp = []
            for j in range(len(list_img)):
                img_tmp.append(list_img[j][i])

            img_tmp = np.concatenate(img_tmp, axis=2) # W
            img_tmp = torch.from_numpy(img_tmp)

            self.add_image(self.mode + f'/{summary_img_name_list[i]}', img_tmp, global_step)

        self.flush()

        rmse = self.metric[0, 0]

        # Reset
        self.loss = []
        self.metric = []

        return rmse

    def save(self, epoch, idx, sample, output, id_in_batch=0):
        with torch.no_grad():
            if self.args.save_result_only:
                self.path_output = '{}/{}/epoch{:04d}'.format(self.log_dir,
                                                              self.mode, epoch)
                os.makedirs(self.path_output, exist_ok=True)

                path_save_pred = '{}/{:010d}.png'.format(self.path_output, idx)

                pred = output['pred'].detach()

                pred = torch.clamp(pred, min=0)

                pred = pred[id_in_batch, 0, :, :].data.cpu().numpy()

                pred = (pred*256.0).astype(np.uint16)
                pred = Image.fromarray(pred)
                pred.save(path_save_pred)
            else:
                # Parse data
                list_feat = []

                rgb_torch = sample['rgb'].detach().clone()
                dep = sample['dep'].detach()
                pred_torch = output['pred'].detach()
                gt_torch = sample['gt'].detach()
                K = sample['K'].detach()

                preds_torch = [d.detach() for d in output['pred_inter']]

                pred_torch = torch.clamp(pred_torch, min=0, max=self.args.max_depth)
                preds_torch = [torch.clamp(d, min=0, max=self.args.max_depth) for d in preds_torch]

                # Un-normalization
                rgb_torch.mul_(self.img_std.type_as(rgb_torch)).add_(
                    self.img_mean.type_as(rgb_torch))

                rgb = rgb_torch[id_in_batch, :, :, :].data.cpu().numpy()
                dep = dep[id_in_batch, 0, :, :].data.cpu().numpy()

                kernel = np.ones((3, 3))
                dep_dialated = cv2.dilate(dep, kernel, iterations=1)

                pred = pred_torch[id_in_batch, 0, :, :].data.cpu().numpy()
                pred_gray = pred
                gt = gt_torch[id_in_batch, 0, :, :].data.cpu().numpy()
                gt_dialated = cv2.dilate(gt, kernel, iterations=1)

                max_depth = max(gt.max(), pred.max())
                norm = plt.Normalize(vmin=np.percentile((gt[gt>0.0]), 5), vmax=np.percentile((gt[gt>0.0]), 95))
                # norm = plt.Normalize(vmin=gt.min(), vmax=self.args.max_depth)

                rgb = np.transpose(rgb, (1, 2, 0))
                for k in range(0, len(preds_torch)):
                    feat_inter = preds_torch[k]
                    feat_inter = feat_inter[id_in_batch, 0, :, :].data.cpu().numpy()
                    feat_inter = np.concatenate((rgb, cm(norm(pred))[...,:3], cm(norm(gt))[...,:3], depth_err_to_colorbar(feat_inter, gt)), axis=0)

                    list_feat.append(feat_inter)

                self.path_output = '{}/{}/epoch{:04d}/{:08d}'.format(
                    self.log_dir, self.mode, epoch, idx)
                os.makedirs(self.path_output, exist_ok=True)

                path_save_rgb = '{}/01_rgb.jpg'.format(self.path_output)
                path_save_dep = '{}/02_dep.png'.format(self.path_output)
                path_save_dep_dialated = '{}/02.1_dep_dialated.png'.format(self.path_output)
                path_save_pred = '{}/05_pred_final.jpg'.format(self.path_output)
                path_save_pred_gray = '{}/05_pred_final_gray.jpg'.format(self.path_output)
                path_save_gt = '{}/06_gt.jpg'.format(self.path_output)
                path_save_gt_dialated = '{}/06.1_gt_dialated.jpg'.format(self.path_output)
                path_save_error = '{}/07_error.jpg'.format(self.path_output)
                path_save_error_dialated = '{}/07.1_error_dialated.jpg'.format(self.path_output)

                plt.imsave(path_save_rgb, rgb, cmap=cmap)
                plt.imsave(path_save_gt, cm(norm(gt)))
                plt.imsave(path_save_gt_dialated, cm(norm(gt_dialated)))
                plt.imsave(path_save_pred, cm(norm(pred)))
                plt.imsave(path_save_pred_gray, pred_gray, cmap='gray')
                plt.imsave(path_save_dep, cm(norm(dep)))
                plt.imsave(path_save_dep_dialated, cm(norm(dep_dialated)))
                plt.imsave(path_save_error, depth_err_to_colorbar(pred, gt, with_bar=False))
                plt.imsave(path_save_error_dialated, depth_err_to_colorbar(pred, gt_dialated, with_bar=False))

                for k in range(0, len(list_feat)):
                    path_save_inter = '{}/04_pred_prop_{:02d}.jpg'.format(
                        self.path_output, k)
                    plt.imsave(path_save_inter, list_feat[k])

                if self.args.save_pointcloud_visualization:
                    unprojector = PtsUnprojector()
                    # pred_torch[gt_torch==0.0] = 0.0
                    pred_torch[pred_torch > self.args.max_depth] = 0.0
                    xyz_gt = unprojector(gt_torch[id_in_batch:id_in_batch+1], K[id_in_batch:id_in_batch+1])  # N x 3
                    xyz_pred = unprojector(pred_torch[id_in_batch:id_in_batch + 1], K[id_in_batch:id_in_batch + 1])  # N x 3

                    colors = unprojector.apply_mask(rgb_torch[id_in_batch:id_in_batch + 1])

                    path_save_pointcloud_gt = '{}/10_pointcloud_gt.ply'.format(self.path_output)
                    path_save_pointcloud_pred = '{}/10_pointcloud_pred.ply'.format(self.path_output)

                    save_ply(path_save_pointcloud_gt, xyz_gt, colors)
                    save_ply(path_save_pointcloud_pred, xyz_pred, colors)

def depth_err_to_colorbar(est, gt=None, with_bar=False, cmap='jet'):
    error_bar_height = 50
    if gt is None:
        gt = np.zeros_like(est)
        valid = est > 0
        max_depth = est.max()
    else:
        valid = gt > 0
        max_depth = gt.max()
    error_map = np.abs(est - gt) * valid
    h, w= error_map.shape

    maxvalue = error_map.max()
    if max_depth < 30:
        breakpoints = np.array([0,      0.1,      0.5,      1.25,     2,    4,       max(10, maxvalue)])
    else:
        breakpoints = np.array([0,      0.1,      0.5,      1.25,     2,    4,     max(90, maxvalue)])
    points      = np.array([0,      0.25,   0.38,   0.66,  0.83,  0.95,     1])
    num_bins    = np.array([0,      w//8,   w//8,   w//4,  w//4,  w//8,     w - (w//4 + w//4 + w//8 + w//8 + w//8)])
    acc_num_bins = np.cumsum(num_bins)

    for i in range(1, len(breakpoints)):
        scale = points[i] - points[i-1]
        start = points[i-1]
        lower = breakpoints[i-1]
        upper = breakpoints[i]
        error_map = revalue(error_map, lower, upper, start, scale)

    # [0, 1], [H, W, 3]
    error_map = plt.cm.get_cmap(cmap)(error_map)[:, :, :3]

    # mark invalid px as black
    error_map = error_map * valid[:, :, None]

    if not with_bar:
        return error_map

    error_bar = np.array([])
    for i in range(1, len(num_bins)):
        error_bar = np.concatenate((error_bar, np.linspace(points[i-1], points[i], num_bins[i])))

    error_bar = np.repeat(error_bar, error_bar_height).reshape(w, error_bar_height).transpose(1, 0) # [error_bar_height, w]
    error_bar_map = plt.cm.get_cmap(cmap)(error_bar)[:, :, :3]
    plt.xticks(ticks=acc_num_bins, labels=[str(f) for f in breakpoints])
    plt.axis('on')

    # [0, 1], [H, W, 3]
    error_map = np.concatenate((error_map, error_bar_map[..., :3]), axis=0)[..., :3]

    return error_map

def revalue(map, lower, upper, start, scale):
    mask = (map > lower) & (map <= upper)
    if np.sum(mask) >= 1.0:
        mn, mx = map[mask].min(), map[mask].max()
        map[mask] = ((map[mask] - mn) / (mx -mn + 1e-7)) * scale + start

    return map


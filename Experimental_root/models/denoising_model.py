import torch
import torch.nn.functional as F
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel
from Experimental_root.models.validation_seq_infer import denoise_seq

@MODEL_REGISTRY.register()
class DenoisingModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(DenoisingModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.center_frame_only = self.opt.get('center_frame_only', False)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            if param_key == 'None':
                param_key = None
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        # Feedin clean image, synthesize noisy image
        self.lq = data['lq'].to(self.device)
        self.noise_map = None
        if 'noise_map' in data:
            self.noise_map = data['noise_map'].to(self.device)

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        if self.noise_map is not None:
            self.output = self.net_g(self.lq, self.noise_map)
        else:
            self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def padding_input(self, padded_lq):
        # make size a multiple of four (we have two scales in the denoiser)
        window_size = 4 
        temp_pad = 0
        ref_pad = 0
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        
        padding_list = [ref_pad, ref_pad+mod_pad_w, ref_pad, ref_pad+mod_pad_h,temp_pad,0]
        padded_lq = F.pad(padded_lq, padding_list[0:4], 'reflect')
        padded_lq_permute = padded_lq.permute(1,2,3,0)
        padded_lq = F.pad(padded_lq_permute, [padding_list[4],padding_list[5], 0,0], 'reflect')
        padded_lq = padded_lq.permute(3,0,1,2)
        # (0, 2, 0, 0, 0, 0)
        # padded_lq = F.pad(lq_permute, **padding_list, 'reflect')
        # padded_lq = F.pad(lq_permute, (0, 2, 0, 0, 0, 0), 'reflect')
        
        # if self.noise_map is not None:
            # padded_noise_map = F.pad(self.noise_map, padding_list, 'reflect')
        # else:
            # padded_noise_map = None
        
        return padded_lq, padding_list
    
    def crop_output(self, padding_list):
        # scale = self.opt.get('scale', 1)
        # _, _, h, w = self.scaled_lr.shape
        # self.scaled_lr = self.scaled_lr[:, :, 0:h - int(mod_pad_h / scale), 0:w - int(mod_pad_w / scale)]
        # self.scaled_degrade_lr = self.scaled_degrade_lr[:, :, 0:h - int(mod_pad_h / scale), 0:w - int(mod_pad_w / scale)]
        pad_w1, pad_w2, pad_h1, pad_h2, temp_pad1,temp_pad2, = padding_list
        _, F, _, h, w = self.output.shape
        self.output = self.output[:, temp_pad1:F-temp_pad2, :, pad_h1:h - pad_h2, pad_w1:w - pad_w2]
        
    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                # self.output = self.net_g_ema(self.lq)
                if self.noise_map is not None:
                    self.output = self.net_g_ema(self.lq, self.noise_map)
                else:
                    self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            padded_lq, padding_list = self.padding_input(self.lq)
            if self.noise_map is not None:
                padded_noise_map, padding_list = self.padding_input(self.noise_map)
            else:
                padded_noise_map = None
            self.output = denoise_seq(padded_lq, padded_noise_map, self.opt['val']['temp_psz'],
                        self.net_g, future_buffer_len=self.opt['val'].get('future_buffer_len', 0),)[None, ...]
            self.crop_output(padding_list)

            self.net_g.train()
            
    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """
        if self.opt['dist']:
            return self.dist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            with torch.cuda.amp.autocast(self.opt['val'].get('fp16', False)):
                logger = get_root_logger()
                if self.opt['val'].get('fp16', False):
                    logger.info("Warning: validate at fp16, performance may be lower")
            # have been validate by same ckpt, 0.002 difference in PSNR
                return self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None
        # initialize self.metric_results
        # It is a dict: {
        #    'folder1': tensor (num_frame x len(metrics)),
        #    'folder2': tensor (num_frame x len(metrics))
        # }
        # if with_metrics and not hasattr(self, 'metric_results'):
        if with_metrics:
            self.metric_results = {}
            # num_frame = self.opt['datasets']['val']['num_validation_frames']
            # seqs_dirs = dataset.seqs_dirs
            for index, folder in enumerate(dataset.base_folder):
                self.metric_results[folder] = torch.zeros(
                    dataset.num_frames[index], len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')

            # rank, world_size = get_dist_info()
            # if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()
            metric_data = dict()
            metric_data_float = dict()
        num_folders = len(dataset)
        # num_pad = (world_size - (num_folders % world_size)) % world_size
        # if rank == 0:
        pbar = tqdm(total=len(dataset), unit='folder')
        # Will evaluate (num_folders + num_pad) times, but only the first
        # num_folders results will be recorded. (To avoid wait-dead)
        for i in range(0, num_folders, 1):
            idx = min(i, num_folders - 1)
            val_data = dataset[idx]
            folder = val_data['folder']

            # compute outputs
            # val_data['lq'].unsqueeze_(0)
            # val_data['gt'].unsqueeze_(0)
            self.feed_data(val_data)
            val_data['lq'].squeeze_(0)
            val_data['gt'].squeeze_(0)
            if 'noise_map' in val_data.keys():
                val_data['noise_map'].squeeze_(0)
            with torch.no_grad(): self.test()
            visuals = self.get_current_visuals()

            # tentative for out of GPU memory
            del self.lq
            del self.output
            del self.noise_map
            if 'gt' in visuals:
                del self.gt
            torch.cuda.empty_cache()

            if self.center_frame_only:
                visuals['result'] = visuals['result'].unsqueeze(1)
                if 'gt' in visuals:
                    visuals['gt'] = visuals['gt'].unsqueeze(1)

            # evaluate
            if i < num_folders:
                for idx in range(visuals['result'].size(1)):
                    result = visuals['result'][0, idx, :, :, :]
                    result_img = tensor2img([result])  # uint8, bgr
                    metric_data['img'] = result_img
                    metric_data_float['img_float'] = result
                    if 'gt' in visuals:
                        gt = visuals['gt'][0, idx, :, :, :]
                        gt_img = tensor2img([gt])  # uint8, bgr
                        metric_data['img2'] = gt_img
                        metric_data_float['img2_float'] = gt

                    if save_img:
                        # if self.opt['is_train']:
                        #     raise NotImplementedError('saving image is not supported during training.')
                        # else:
                        if self.center_frame_only:  # vimeo-90k
                            clip_ = val_data['lq_path'].split('/')[-3]
                            seq_ = val_data['lq_path'].split('/')[-2]
                            name_ = f'{clip_}_{seq_}'
                            img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                f"{name_}_{self.opt['name']}.png")
                        else:  # others
                            img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                f"{idx:08d}_{self.opt['name']}.png")
                            # image name only for REDS dataset
                        imwrite(result_img, img_path)

                    # calculate metrics
                    if with_metrics:
                        for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                            if 'float' in opt_['type']:
                                result = calculate_metric(metric_data_float, opt_)
                            else:    
                                result = calculate_metric(metric_data, opt_)
                            self.metric_results[folder][idx, metric_idx] += result
                pbar.update(1)
                pbar.set_description(f'Test {folder}')
            total_avg_results = self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        pbar.close()
        # if rank == 0:
        return total_avg_results

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        
        logger = get_root_logger()
        
        # average all frames for each sub-folder
        # metric_results_avg is a dict:{
        #    'folder1': tensor (len(metrics)),
        #    'folder2': tensor (len(metrics))
        # }
        metric_results_avg = {
            folder: torch.mean(tensor, dim=0).cpu()
            for (folder, tensor) in self.metric_results.items()
        }
        # total_avg_results is a dict: {
        #    'metric1': float,
        #    'metric2': float
        # }
        import pandas as pd
        
        # df_dict = {'name': self.metric_all_dict['image_name']}
        for sub_img in self.metric_results:
            df_dict = {}
            # if isinstance(self.metric_results[sub_img], dict):
            for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                df_dict[f'{sub_img}_{metric_idx}'] = self.metric_results[sub_img][:, metric_idx].cpu()
            df = pd.DataFrame.from_dict(df_dict)
            csv_path = logger.handlers[1].baseFilename.replace(".log", f"{sub_img}.csv")
            df.to_csv(csv_path)
        
        total_avg_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        for folder, tensor in metric_results_avg.items():
            for idx, metric in enumerate(total_avg_results.keys()):
                total_avg_results[metric] += metric_results_avg[folder][idx].item()
        # average among folders
        for metric in total_avg_results.keys():
            total_avg_results[metric] /= len(metric_results_avg)

        log_str = f'Validation {dataset_name}\n'
        for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
            log_str += f'\t # {metric}: {value:.4f}'
            for folder, tensor in metric_results_avg.items():
                log_str += f'\t # {folder}: {tensor[metric_idx].item():.4f}'
            log_str += '\n'
        logger.info(log_str)
        if tb_logger:
            for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
                for folder, tensor in metric_results_avg.items():
                    tb_logger.add_scalar(f'metrics/{metric}/{folder}', tensor[metric_idx].item(), current_iter)
        return total_avg_results

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state['optimizers']
        # resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        # assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        # for i, s in enumerate(resume_schedulers):
            # self.schedulers[i].load_state_dict(s)

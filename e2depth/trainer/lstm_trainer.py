import numpy as np
import torch
from base import BaseTrainer
from torchvision import utils
from model.loss import temporal_consistency_loss, mse_loss, l1_loss, multi_scale_grad_loss, depth_smoothness_loss, ordinal_depth_loss
from utils.training_utils import select_evenly_spaced_elements, plot_grad_flow, plot_grad_flow_bars
import torch.nn.functional as f


def quick_norm(img):
    return (img - torch.min(img))/(torch.max(img) - torch.min(img) + 1e-5)


class LSTMTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """

    def __init__(self, model, loss, loss_params, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(LSTMTrainer, self).__init__(model, loss,
                                          loss_params, metrics, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        self.log_step = int(np.sqrt(self.batch_size))
        self.num_previews = config['trainer']['num_previews']
        self.num_val_previews = config['trainer']['num_val_previews']
        self.record_every_N_sample = 5
        self.movie = bool(config['trainer'].get('movie', True))
        self.still_previews = bool(config['trainer'].get('still_previews', False))
        self.grid_loss = bool(config['trainer'].get('grid_loss', False))

        # Parameters for temporal consistency loss
        if 'temporal_consistency_loss' in config:
            self.use_temporal_consistency_loss = True
            try:
                self.L0 = config['temporal_consistency_loss']['L0']
            except KeyError:
                self.L0 = 1

            try:
                self.weight_temporal_consistency = config['temporal_consistency_loss']['weight']
            except KeyError:
                self.weight_temporal_consistency = 5.0

            print('Using temporal consistency loss with L0={} and weight={:.2f}'
                  .format(self.L0, self.weight_temporal_consistency))
        else:
            print('Will not use temporal consistency loss')
            self.use_temporal_consistency_loss = False

        # Parameters for multi scale gradiant loss
        if 'grad_loss' in config:
            self.use_grad_loss = True
            try:
                self.weight_grad_loss = config['grad_loss']['weight']
            except KeyError:
                self.weight_grad_loss = 1.0

            print('Using Multi Scale Gradient loss with weight={:.2f}'.format(
                self.weight_grad_loss))
        else:
            print('Will not use Multi Scale Gradiant loss')
            self.use_grad_loss = False

        # Parameters for smooth gradiant loss
        if 'smooth_loss' in config:
            self.use_smooth_loss = True
            try:
                self.weight_smooth_loss = config['smooth_loss']['weight']
            except KeyError:
                self.weight_smooth_loss = 1.0

            print('Using Multi Scale Depth Smoothness loss with weight={:.2f}'.format(
                self.weight_smooth_loss))
        else:
            print('Will not use Multi Scale Depth Smoothness loss')
            self.use_smooth_loss = False

        # Parameters for ordinal depth loss
        if 'ordinal_loss' in config:
            self.use_ordinal_loss = True
            try:
                self.weight_ordinal_loss = config['ordinal_loss']['weight']
            except KeyError:
                self.weight_ordinal_loss = 1.0
            try:
                self.percent_ordinal_loss = config['ordinal_loss']['percent']
            except KeyError:
                self.percent_ordinal_loss = 1.0
            try:
                self.method_ordinal_loss = config['ordinal_loss']['method']
            except KeyError:
                self.method_ordinal_loss = "classical"


            print('Using Ordinal Depth loss with weight={:.2f}, percent={:.2f} and method'.format(
                self.weight_ordinal_loss, self.percent_ordinal_loss), (self.method_ordinal_loss))
        else:
            print('Will not use Ordinal Depth loss')
            self.use_ordinal_loss = False

        # Semantic loss (TO-DO)
        self.use_semantic_loss = False

        # Parameters for mse loss
        if 'mse_loss' in config:
            self.use_mse_loss = True
            try:
                self.weight_mse_loss = config['mse_loss']['weight']
            except KeyError:
                self.weight_mse_loss = 1.0

            try:
                self.mse_loss_downsampling_factor = config['mse_loss']['downsampling_factor']
            except KeyError:
                self.mse_loss_downsampling_factor = 0.5

            print('Using MSE loss with weight={:.2f} and downsampling factor={:.2f}'.format(
                self.weight_mse_loss, self.mse_loss_downsampling_factor))
        else:
            print('Will not use MSE loss')
            self.use_mse_loss = False

        # Parameters for l1 loss
        if 'l1_loss' in config:

            self.use_l1_loss = True
            try:
                self.weight_l1_loss = config['l1_loss']['weight']
            except KeyError:
                self.weight_l1_loss = 1.0

            try:
                self.l1_loss_downsampling_factor = config['l1_loss']['downsampling_factor']
            except KeyError:
                self.l1_loss_downsampling_factor = 0.5

            print('Using L1 loss with weight={:.2f} and downsampling factor={:.2f}'.format(
                self.weight_l1_loss, self.l1_loss_downsampling_factor))
        else:
            print('Will not use L1 loss')
            self.use_l1_loss = False

        # To visualize the progress of the network on the training and validation data,
        # we plot N training / validation samples, spread uniformly over all the samples
        self.preview_indices = select_evenly_spaced_elements(self.num_previews, len(self.data_loader))
        if valid_data_loader:
            self.val_preview_indices = select_evenly_spaced_elements(self.num_val_previews, len(self.valid_data_loader))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        output = output.cpu().data.numpy()
        target = target.cpu().data.numpy()
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
        return acc_metrics

    def _to_input_and_target(self, item):
        network_input = item['events'].to(self.gpu)
        target = item['frame'].to(self.gpu)
        flow = item['flow'].to(self.gpu) if self.use_temporal_consistency_loss else None
        semantic = item['semantic'].to(self.gpu) if self.use_semantic_loss else None
        return network_input, target, flow, semantic

    @staticmethod
    def make_preview(event_previews, predicted_frames, groundtruth_frames):
        # event_previews: a list of [1 x 1 x H x W] event previews
        # predicted_frames: a list of [1 x 1 x H x W] predicted frames
        # for make_grid, we need to pass [N x 1 x H x W] where N is the number of images in the grid
        for f in groundtruth_frames:
            if torch.isnan(f).sum()>0:
                f[f!=f] = 0
        return utils.make_grid(torch.cat(event_previews + predicted_frames + groundtruth_frames, dim=0),
                               normalize=True, scale_each=True,
                               nrow=len(predicted_frames))
    @staticmethod
    def make_grad_loss_preview(grad_loss_frames):
        # grad_loss_frames is a list of N multi scale grad losses of size [1 x 1 x H x W]
        return utils.make_grid(torch.cat(grad_loss_frames, dim=0),
                               normalize=True, scale_each=True,
                               nrow=len(grad_loss_frames))

    @staticmethod
    def make_movie(event_previews, predicted_frames, groundtruth_frames):
        # event_previews: a list of [1 x 1 x H x W] event previews
        # predicted_frames: a list of [1 x 1 x H x W] predicted frames
        # for movie, we need to pass [1 x T x 1 x H x W] where T is the time dimension

        video_tensor = None
        for i in torch.arange(len(event_previews)):
            voxel = event_previews[i] #quick_norm(event_previews[i])
            predicted_frame = predicted_frames[i] #quick_norm(predicted_frames[i])
            movie_frame = torch.cat([voxel,
                                     predicted_frame,
                                     groundtruth_frames[i]],
                                     dim=-1)
            movie_frame.unsqueeze_(dim=0)
            video_tensor = movie_frame if video_tensor is None else \
                torch.cat((video_tensor, movie_frame), dim=1)
        return video_tensor

    def forward_pass_sequence(self, sequence, record=False):
        # 'sequence' is a list containing L successive events <-> frames pairs
        # each element in 'sequence' is a dictionary containing the keys 'events' and 'frame'
        L = len(sequence)
        assert(L > 0)

        # list of per-iteration losses (summed after the loop)
        iter_losses = [] # main defined loss
        iter_grad_losses = []
        iter_smooth_losses = []
        iter_ordinal_losses = []
        iter_temporal_losses = []
        iter_mse_losses = []
        iter_l1_losses = []

        if record:
            event_previews = []
            predicted_frames = []  # list of intermediate predicted frames
            groundtruth_frames = []
            grad_loss_frames = [] # list of loss visualization frames

        if self.use_temporal_consistency_loss:
            assert(self.L0 >= 1)
            assert(self.L0 < L)

        # initialize the K last predicted frames with -1
        N, _, H, W = sequence[0]['frame'].shape
        prev_states = None
        prev_frame, prev_predicted_frame = None, None
        for l in range(L):
            item = sequence[l]
            new_events, new_frame, flow01, semantic = self._to_input_and_target(item)
            # the output of the network is a [N x 1 x H x W] tensor containing the image prediction
            new_predicted_frame, states = self.model(new_events, prev_states)

            # with torch.no_grad():
            #     print('gt. std: {:.3f}'.format(new_frame.std()))
            #     print('rec. std: {:.3f}'.format(new_predicted_frame.std()))

            prev_states = states

            if record:
                with torch.no_grad():
                    event_previews.append(torch.sum(new_events, dim=1).unsqueeze(0))
                    predicted_frames.append(new_predicted_frame.clone())
                    groundtruth_frames.append(new_frame.clone())

            # Compute the nominal loss
            if self.loss_params is not None:
                iter_losses.append(
                    self.loss(new_predicted_frame, new_frame, **self.loss_params))
            else:
                iter_losses.append(self.loss(new_predicted_frame, new_frame))

            # Compute the temporal consistency loss
            if self.use_temporal_consistency_loss:
                if l >= self.L0:
                    assert(prev_frame is not None)
                    assert(prev_predicted_frame is not None)
                    iter_temporal_losses.append(
                        temporal_consistency_loss(prev_frame, new_frame,
                                                  prev_predicted_frame, new_predicted_frame,
                                                  flow01))

            # Compute the multi scale gradient loss
            if self.use_grad_loss:
                if record:
                    with torch.no_grad():
                        grad_loss_frames.append( multi_scale_grad_loss(new_predicted_frame, new_frame, preview = record))
                else:
                    grad_loss = multi_scale_grad_loss(new_predicted_frame, new_frame)
                    iter_grad_losses.append(grad_loss)

            # Compute the smooth loss
            if self.use_smooth_loss:
                smooth_loss = depth_smoothness_loss(new_predicted_frame, new_events)
                iter_smooth_losses.append(smooth_loss)
            
            # Compute the ordinal loss
            if self.use_ordinal_loss:
                ordinal_loss = ordinal_depth_loss(new_predicted_frame, new_frame, new_events,
                                                    self.percent_ordinal_loss, self.method_ordinal_loss)
                iter_ordinal_losses.append(ordinal_loss)

            # Compute the mse loss
            if self.use_mse_loss:
                # compute the MSE loss at a lower resolution
                downsampling_factor = self.mse_loss_downsampling_factor

                if downsampling_factor != 1.0:
                    new_frame_downsampled = f.interpolate(
                        new_frame, scale_factor=downsampling_factor, mode='bilinear', align_corners=False)
                    new_predicted_frame_downsampled = f.interpolate(
                        new_predicted_frame, scale_factor=downsampling_factor, mode='bilinear', align_corners=False)
                    mse = mse_loss(new_predicted_frame_downsampled, new_frame_downsampled)
                else:
                    mse = mse_loss(new_predicted_frame, new_frame)
                iter_mse_losses.append(mse)
            
            # Compute the l1 loss
            if self.use_l1_loss:
                # compute the L1 loss at a lower resolution
                downsampling_factor = self.l1_loss_downsampling_factor

                if downsampling_factor != 1.0:
                    new_frame_downsampled = f.interpolate(
                        new_frame, scale_factor=downsampling_factor, mode='bilinear', align_corners=False)
                    new_predicted_frame_downsampled = f.interpolate(
                        new_predicted_frame, scale_factor=downsampling_factor, mode='bilinear', align_corners=False)
                    l1 = l1_loss(new_predicted_frame_downsampled, new_frame_downsampled)
                else:
                    l1 = l1_loss(new_predicted_frame, new_frame)
                iter_l1_losses.append(l1)
 
            prev_frame = new_frame
            prev_predicted_frame = new_predicted_frame

        nominal_loss = sum(iter_losses) / float(L)

        losses = []
        losses.append(nominal_loss)

        # Add temporal consistenvy loss to the losses
        if self.use_temporal_consistency_loss:
            temporal_loss = self.weight_temporal_consistency * sum(iter_temporal_losses) / float(L - self.L0)
            losses.append(temporal_loss)

        # Add multi scale gradient loss
        if self.use_grad_loss:
            grad_loss = self.weight_grad_loss * sum(iter_grad_losses)/float(L)
            losses.append(grad_loss)

        # Add multi scale smooth loss
        if self.use_smooth_loss:
            smooth_loss = self.weight_smooth_loss * sum(iter_smooth_losses)/float(L)
            losses.append(smooth_loss)

        # Add ordinal depth loss
        if self.use_ordinal_loss:
            ordinal_loss = self.weight_ordinal_loss * sum(iter_ordinal_losses)/float(L)
            losses.append(ordinal_loss)

        # Add mse loss to the losses
        if self.use_mse_loss:
            mse = self.weight_mse_loss * sum(iter_mse_losses) / float(L)
            losses.append(mse)

        # Add L1 loss to the losses
        if self.use_l1_loss:
            l1 = self.weight_l1_loss * sum(iter_l1_losses) / float(L)
            losses.append(l1)

        loss = sum(losses)

        # add all losses in a dict for logging
        with torch.no_grad():
            loss_dict = {'loss': loss, 'L_si': nominal_loss}
            if self.use_temporal_consistency_loss:
                loss_dict['L_tc'] = temporal_loss
            if self.use_grad_loss:
                loss_dict['L_grad'] = grad_loss
            if self.use_smooth_loss:
                loss_dict['L_smooth'] = smooth_loss
            if self.use_ordinal_loss:
                loss_dict['L_ord'] = ordinal_loss
            if self.use_mse_loss:
                loss_dict['L_mse'] = mse
            if self.use_l1_loss:
                loss_dict['L_l1'] = l1

        return loss_dict, \
            predicted_frames if record else None, \
            groundtruth_frames if record else None, \
            event_previews if record else None, \
            grad_loss_frames if record else None

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        all_losses_in_batch = {}
        for batch_idx, sequence in enumerate(self.data_loader):

            self.optimizer.zero_grad()
            losses, _, _, _ , _= self.forward_pass_sequence(sequence)
            loss = losses['loss']
            loss.backward()
            if batch_idx % 25 == 0:
                plot_grad_flow(self.model.named_parameters())
            self.optimizer.step()

            with torch.no_grad():
                for loss_name, loss_value in losses.items():
                    if loss_name not in all_losses_in_batch:
                        all_losses_in_batch[loss_name] = []
                    all_losses_in_batch[loss_name].append(loss_value.item())

                if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                    loss_str = ''
                    for loss_name, loss_value in losses.items():
                        loss_str += '{}: {:.4f} '.format(loss_name, loss_value.item())
                    self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] {}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        len(self.data_loader) * self.data_loader.batch_size,
                        100.0 * batch_idx / len(self.data_loader),
                        loss_str))

        with torch.no_grad():
            # create a set of previews and log them
            previews = []
            total_metrics = np.zeros(len(self.metrics))
            self.preview_count = 0
            for preview_idx in self.preview_indices:
                # data is a sequence containing L successive events <-> frames pairs
                sequence = self.data_loader.dataset[preview_idx]

                # every element in sequence is a [C x H x W] tensor
                # but the model requires [1 x C x H x W] tensor, so
                # we preprocess the data here to adjust to this expected format
                for data_items in sequence:
                    for item in data_items.values():
                        item.unsqueeze_(dim=0)

                _, predicted_frames, groundtruth_frames, event_previews, grad_loss_frames = self.forward_pass_sequence(
                    sequence, record=True)
                hist_idx = len(predicted_frames) - 1  # choose an idx to plot
                self.writer.add_histogram(f'{self.preview_count}_prediction',
                                          predicted_frames[hist_idx],
                                          global_step=epoch)
                self.writer.add_histogram(f'{self.preview_count}_groundtruth',
                                          groundtruth_frames[hist_idx],
                                          global_step=epoch)

                fig = plot_grad_flow_bars(self.model.named_parameters())
                self.writer.add_figure('grad_figure', fig, global_step=epoch)

                total_metrics += self._eval_metrics(predicted_frames[0], groundtruth_frames[0])

                for tag, value in self.model.named_parameters():
                    self.writer.add_histogram(tag + '/grad', value.grad, global_step=epoch)
                    self.writer.add_histogram(tag + '/weights', value.data, global_step=epoch)
                if self.movie:
                    video_tensor = self.make_movie(event_previews, predicted_frames, groundtruth_frames)
                    self.writer.add_video(
                        f'movie_{self.preview_count}__events__prediction__groundtruth',
                        video_tensor, global_step=epoch, fps=5)
                if self.still_previews:
                    step = self.record_every_N_sample
                    previews.append(self.make_preview(
                        event_previews[::step], predicted_frames[::step], groundtruth_frames[::step]))
                if self.grid_loss and len(grad_loss_frames) != 0:
                    #print ("train len(grad_loss_frames[0]): ", len(grad_loss_frames[::step][0]))
                    previews.append(self.make_grad_loss_preview(grad_loss_frames[::step][0]))
                self.preview_count += 1

        # compute average losses over the batch
        total_losses = {loss_name: sum(loss_values) / len(self.data_loader)
                        for loss_name, loss_values in all_losses_in_batch.items()}
        log = {
            'loss': total_losses['loss'],
            'losses': total_losses,
            'metrics': (total_metrics / len(self.data_loader)).tolist(),
            'previews': previews
        }

        if self.valid:
            val_log = self._valid_epoch(epoch=epoch)
            log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch=0):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        all_losses_in_batch = {}
        with torch.no_grad():
            for batch_idx, sequence in enumerate(self.valid_data_loader):
                losses, _, _, _, _ = self.forward_pass_sequence(sequence)
                for loss_name, loss_value in losses.items():
                    if loss_name not in all_losses_in_batch:
                        all_losses_in_batch[loss_name] = []
                    all_losses_in_batch[loss_name].append(loss_value.item())

                if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                    self.logger.info('Validation: [{}/{} ({:.0f}%)]'.format(
                        batch_idx * self.valid_data_loader.batch_size,
                        len(self.valid_data_loader) * self.valid_data_loader.batch_size,
                        100.0 * batch_idx / len(self.valid_data_loader)))

            # create a set of previews and log then
            val_previews = []
            total_metrics = np.zeros(len(self.metrics))
            self.preview_count = 0
            for val_preview_idx in self.val_preview_indices:
                # data is a sequence containing L successive events <-> frames pairs
                sequence = self.valid_data_loader.dataset[val_preview_idx]

                # every element in sequence is a [C x H x W] tensor
                # but the model requires [1 x C x H x W] tensor, so
                # we preprocess the data here to adjust to this expected format
                for data_items in sequence:
                    for item in data_items.values():
                        item.unsqueeze_(dim=0)

                _, predicted_frames, groundtruth_frames, event_previews, grad_loss_frames = self.forward_pass_sequence(
                    sequence, record=True)

                total_metrics += self._eval_metrics(predicted_frames[0], groundtruth_frames[0])

                if self.movie:
                    video_tensor = self.make_movie(event_previews, predicted_frames, groundtruth_frames)
                    self.writer.add_video(
                        f"val_movie_{self.preview_count}__events__prediction__groundtruth",
                        video_tensor, global_step=epoch, fps=5)
                    self.preview_count += 1
                if self.still_previews:
                    step = self.record_every_N_sample
                    val_previews.append(self.make_preview(
                        event_previews[::step], predicted_frames[::step], groundtruth_frames[::step]))
                if self.grid_loss and len(grad_loss_frames) != 0:
                    val_previews.append(self.make_grad_loss_preview(grad_loss_frames[::step][0]))

        total_losses = {loss_name: sum(loss_values) / len(self.valid_data_loader)
                        for loss_name, loss_values in all_losses_in_batch.items()}
        return {
            'val_loss': total_losses['loss'],
            'val_losses': total_losses,
            'val_metrics': (total_metrics / len(self.data_loader)).tolist(),
            'val_previews': val_previews
        }

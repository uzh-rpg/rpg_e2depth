import numpy as np
import torch
from base import BaseTrainer
from torchvision import utils
from utils.training_utils import plot_grad_flow


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """

    def __init__(self, model, loss, loss_params, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, loss_params, metrics, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        self.log_step = int(np.sqrt(self.batch_size))
        self.num_previews = config['trainer']['num_previews']
        self.num_val_previews = config['trainer']['num_val_previews']

        try:
            self.weight_contrast_loss = config['weight_contrast_loss']
            print('Will use contrast loss with weight={:.2f}'.format(self.weight_contrast_loss))
        except KeyError:
            print('Will not use contrast loss')
            self.weight_contrast_loss = 0

        # To visualize the progress of the network on the training and validation data,
        # we select a random sample in the training and validation set, which we will plot at each epoch
        self.preview_indices = np.random.choice(range(len(data_loader.dataset)), self.num_previews, replace=False)
        if valid_data_loader:
            self.val_preview_indices = np.random.choice(range(len(valid_data_loader.dataset)),
                                                        self.num_val_previews,
                                                        replace=False)

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        output = output.cpu().data.numpy()
        target = target.cpu().data.numpy()
        # output = np.argmax(output, axis=1)
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
        return acc_metrics

    def _to_input_and_target(self, item):
        network_input = item['events'].to(self.gpu)
        target = item['frame'].to(self.gpu)
        return network_input, target

    @staticmethod
    def make_preview(network_input, target, target_pred):
        # the visualization consists of three images: [events, predicted image, ground truth image]
        # network_input: [C x H x W]
        # target: [1 x H x W]
        # target_pred: [1 x H x W]
        # for make_grid, we need to pass [N x 1 x H x W] where N is the number of images in the grid
        event_preview = torch.sum(network_input, dim=0).unsqueeze(0)
        return utils.make_grid(torch.cat([event_preview, target, target_pred], dim=0).unsqueeze(dim=1),
                               normalize=True, scale_each=True,
                               nrow=3)

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

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, data in enumerate(self.data_loader):
            network_input, target = self._to_input_and_target(data)
            self.optimizer.zero_grad()
            target_pred, _ = self.model(network_input)

            if self.loss_params is not None:
                reconstruction_loss = self.loss(target_pred, target, **self.loss_params)
            else:
                reconstruction_loss = self.loss(target_pred, target)

            # contrast loss: tries to push the image to have reasonable contrast

            # with torch.no_grad():
            #     print('gt. std: {:.3f}'.format(target.std()))
            #     print('rec. std: {:.3f}'.format(target_pred.std()))

            contrast_loss = self.weight_contrast_loss * torch.pow(target_pred.std() - target.std(), 2)
            loss = reconstruction_loss + contrast_loss

            loss.backward()
            if batch_idx % 25 == 0:
                plot_grad_flow(self.model.named_parameters())
            self.optimizer.step()

            total_loss += loss.item()
            total_metrics += self._eval_metrics(target_pred, target)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.3f}, L_r: {:.3f}, L_contrast: {:.3f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    len(self.data_loader) * self.data_loader.batch_size,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item(),
                    reconstruction_loss.item(),
                    contrast_loss.item()))

        with torch.no_grad():
            # create a set of previews and log them
            previews = []
            for preview_idx in self.preview_indices:
                data = self.data_loader.dataset[preview_idx]
                network_input, target = self._to_input_and_target(data)
                target_pred, _ = self.model(network_input.unsqueeze(dim=0))
                previews.append(self.make_preview(network_input, target, target_pred.squeeze(dim=0)))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist(),
            'previews': previews
        }

        if self.valid:
            val_log = self._valid_epoch()
            log = {**log, **val_log}

        return log

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                network_input, target = self._to_input_and_target(data)
                target_pred, _ = self.model(network_input)

                if self.loss_params is not None:
                    loss = self.loss(target_pred, target, **self.loss_params)
                else:
                    loss = self.loss(target_pred, target)

                contrast_loss = self.weight_contrast_loss * torch.pow(target_pred.std() - target.std(), 2)
                loss += contrast_loss

                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(target_pred, target)

            # create a set of previews and log then
            val_previews = []
            for val_preview_idx in self.val_preview_indices:
                data = self.valid_data_loader.dataset[val_preview_idx]
                network_input, target = self._to_input_and_target(data)
                target_pred, _ = self.model(network_input.unsqueeze(dim=0))
                val_preview = self.make_preview(network_input, target, target_pred.squeeze(dim=0))
                val_previews.append(val_preview)

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist(),
            'val_previews': val_previews
        }

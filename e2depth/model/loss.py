import torch.nn.functional as F
import torch
import numpy as np
from LPIPS.models import dist_model as dm
import kornia
from kornia.filters.sobel import spatial_gradient, sobel

def scale_invariant_loss(y_input, y_target, weight = 1.0, n_lambda = 1.0):
    log_diff = y_input - y_target
    is_nan = torch.isnan(log_diff)
    return weight * ((log_diff[~is_nan]**2).mean()-(n_lambda*(log_diff[~is_nan].mean())**2))

def scale_invariant_log_loss(y_input, y_target, n_lambda = 1.0):
    log_diff = torch.log(y_input)-torch.log(y_target)
    is_nan = torch.isnan(log_diff)
    return (log_diff[~is_nan]**2).mean()-(n_lambda*(log_diff[~is_nan].mean())**2)

def mse_loss(y_input, y_target):
    return F.mse_loss(y_input[~torch.isnan(y_target)], y_target[~torch.isnan(y_target)])

def l1_loss(y_input, y_target):
    return F.l1_loss(y_input[~torch.isnan(y_target)], y_target[~torch.isnan(y_target)])

class DepthSmoothnessLoss(torch.nn.Module):
    r"""Criterion that computes image-aware inverse depth smoothness loss.
    based on kornia.losses.depth_smooth

    .. math::

        \text{loss} = \left | \partial_x d_{ij} \right | e^{-\left \|
        \partial_x I_{ij} \right \|} + \left |
        \partial_y d_{ij} \right | e^{-\left \| \partial_y I_{ij} \right \|}


    Shape:
        - Inverse Depth: :math:`(N, 1, H, W)`
        - Voxel: :math:`(N, B, H, W)` where B is the number of voxel bins
        - Output: scalar

    Examples::

        >>> prediction = torch.rand(1, 1, 4, 5)
        >>> voxel = torch.rand(1, 5, 4, 5)
        >>> smooth = DepthSmoothnessLoss()
        >>> loss = smooth(prediction, voxel)
    """

    def __init__(self, start_scale = 1, num_scales = 4):
        super(DepthSmoothnessLoss, self).__init__()
        print('Setting up Multi Scale Depth Smoothness loss...')

        self.start_scale = start_scale
        self.num_scales = num_scales

        self.multi_scales = [torch.nn.AvgPool2d(self.start_scale * (2**scale), self.start_scale * (2**scale)) for scale in range(self.num_scales)]
        print('Done')

    @staticmethod
    def gradient_x(img: torch.Tensor) -> torch.Tensor:
        assert len(img.shape) == 4, img.shape
        return torch.roll(img, shifts=(0,0,0,1), dims=(0,1,2,3)) - img
        #return img[:, :, :, :-1] - img[:, :, :, 1:]

    @staticmethod
    def gradient_y(img: torch.Tensor) -> torch.Tensor:
        assert len(img.shape) == 4, img.shape
        return torch.roll(img, shifts=(0,0,1,0), dims=(0,1,2,3)) - img
        #return img[:, :, :-1, :] - img[:, :, 1:, :]

    def forward(  # type: ignore
            self,
            prediction: torch.Tensor,
            voxel: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(prediction):
            raise TypeError("Input prediction type is not a torch.Tensor. Got {}"
                            .format(type(prediction)))
        if not torch.is_tensor(voxel):
            raise TypeError("Input voxel type is not a torch.Tensor. Got {}"
                            .format(type(voxel)))
        if not len(prediction.shape) == 4:
            raise ValueError("Invalid prediction shape, we expect BxCxHxW. Got: {}"
                             .format(prediction.shape))
        if not len(voxel.shape) == 4:
            raise ValueError("Invalid voxel shape, we expect BxCxHxW. Got: {}"
                             .format(voxel.shape))
        if not prediction.shape[-2:] == voxel.shape[-2:]:
            raise ValueError("Prediction and voxel shapes must be the same. Got: {}"
                             .format(prediction.shape, voxel.shape))
        if not prediction.device == voxel.device:
            raise ValueError(
                "Prediction and voxel must be in the same device. Got: {}" .format(
                    prediction.device, voxel.device))
        if not prediction.dtype == voxel.dtype:
            raise ValueError(
                "Prediction and voxel must be in the same dtype. Got: {}" .format(
                    prediction.dtype, voxel.dtype))
        # Loss
        loss_value = 0

        # compute the gradients
        prediction_dx: torch.Tensor = self.gradient_x(prediction)
        prediction_dy: torch.Tensor = self.gradient_y(prediction)
        events: torch.Tensor = torch.sum(voxel, dim=1).unsqueeze(0)

        for m in self.multi_scales:
            # compute image weights
            weights: torch.Tensor = torch.exp(
                -torch.mean(torch.abs(m(events)), dim=1, keepdim=True))

            # apply image weights to depth
            smoothness_x: torch.Tensor = torch.abs(m(prediction_dx) * weights)
            smoothness_y: torch.Tensor = torch.abs(m(prediction_dy) * weights)
            loss_value += torch.mean(smoothness_x) + torch.mean(smoothness_y)
            
        return (loss_value/self.num_scales)

depth_smoothness_loss_fn = DepthSmoothnessLoss()

def depth_smoothness_loss(
        prediction: torch.Tensor,
        voxel: torch.Tensor) -> torch.Tensor:
    r"""Computes image-aware depth smoothness loss.
    """
    return depth_smoothness_loss_fn.forward(prediction, voxel)

def ordinal_depth_loss(prediction: torch.Tensor, target: torch.Tensor, voxel: torch.Tensor, percent = 1.0, method = "classical"):
    def cost (z_i, z_j, r):
        if r == 1:
            return torch.log(1.0+torch.exp(-z_i + z_j))
        elif r == -1:
            return torch.log(1.0+torch.exp(z_i - z_j))
        else:
            return (z_i - z_j)**2

    # Voxel is (1, B, H, W) where B is the number of voxel bins
    # Get the events from voxel
    events: torch.Tensor = torch.sum(voxel, dim=1).unsqueeze(1)
    mprediction = prediction[events !=0]
    mtarget = target[events != 0]

    # Reduce the number of events (by default use all events points)
    if percent != 1.0:
        factor = int(1.0/percent)
        mprediction = mprediction[::factor]
        mtarget = mprediction[::factor]

    if method == "classical":
        # Compute the ordinal relation of events points (+1, -1 , 0)
        ordinal = torch.tensor( [1 if (a>b) else -1 if (a<b) else 0 for a,b in zip(mtarget, mtarget.flip(dims=[0]))], dtype=torch.int8)
        # Compute the ordinal depth loss based in the cost function
        loss_tensor = torch.Tensor([cost(i, j, r) for i, j, r in zip (mprediction, mprediction.flip(dims=[0]), ordinal)])
    else:
        # Alternative ordinal relation of events points (+1, -1 , 0)
        ordinal = torch.tensor( [1 if (a>b) else -1 if (a<b) else 0 for a,b in zip(mtarget, mprediction)], dtype=torch.int8)
        # Compute the alternative ordinal depth loss based in the cost function
        loss_tensor = torch.Tensor([cost(i, j, r) for i, j, r in zip (mtarget, mprediction, ordinal)])

    return loss_tensor.mean()

class MultiScaleGradient(torch.nn.Module):
    def __init__(self, start_scale = 1, num_scales = 4):
        super(MultiScaleGradient,self).__init__()
        print('Setting up Multi Scale Gradient loss...')

        self.start_scale = start_scale
        self.num_scales = num_scales

        self.multi_scales = [torch.nn.AvgPool2d(self.start_scale * (2**scale), self.start_scale * (2**scale)) for scale in range(self.num_scales)]
        print('Done')
    

    def forward(self, prediction, target, preview = False):
        # helper to remove potential nan in labels
        def interpolate_nans(y):
            #B, C, H, W = y.shape
            copy = y.cpu().numpy()
            [nans, x] = np.isnan(copy), lambda z: z.nonzero()[0]
            copy[nans]= np.interp(x(nans), x(~nans), copy[~nans])
            return torch.from_numpy(copy).cuda()

        def grid_sample_nans(y):
            B, _, H, W = y.shape
            D_H = torch.linspace(1, H, H)
            D_W = torch.linspace(1, W, W)
            meshy, meshx = torch.meshgrid((D_H, D_W))
            grid = torch.stack((meshy, meshx), 2).cuda()
            grid = grid.expand(B, H, W, 2)
            return  torch.nn.functional.grid_sample(y, grid, padding_mode='border', align_corners=True)

        #if torch.isnan(target).sum()>0:
        #    new_target = grid_sample_nans(target)
        #    diff = prediction - new_target
        #else:
        diff = prediction - target
        diff[target!=target] = 0 # clean values where are nans
        _,_,H,W = target.shape
        upsample = torch.nn.Upsample(size=(2*H,2*W), mode='bicubic', align_corners=True)
        record = []
        loss_value = 0

        for m in self.multi_scales:
            # input and type are of the type [B x C x H x W]
            if preview:
                record.append(upsample(sobel(m(diff))))
            else:
                # Use kornia spatial gradient computation
                delta_diff = spatial_gradient(m(diff))
                # output of kornia spatial gradient is [B x C x 2 x H x W]
                loss_value += torch.abs(delta_diff).mean(dim=(3,4)).sum()

        if preview:
            return record
        else:
            return (loss_value/self.num_scales)

multi_scale_grad_loss_fn = MultiScaleGradient()

def multi_scale_grad_loss(prediction, target, preview = False):
    return multi_scale_grad_loss_fn.forward(prediction, target, preview)

class PerceptualLoss(torch.nn.Module):
    # VGG using our perceptually-learned weights (LPIPS metric)
    def __init__(self, model='net-lin', net='vgg', use_gpu=True):
        print('Setting up Perceptual loss..')
        self.model = dm.DistModel()
        self.model.initialize(model=model, net=net, use_gpu=True)
        print('Done')

    def forward(self, pred, target, normalize=False):
        """
        Pred and target are Variables.
        If normalize is on, scales images between [-1, 1]
        Assumes the inputs are in range [0, 1].
        """
        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1

        dist = self.model.forward_pair(target, pred)

        return dist


perceptual_loss_fn = PerceptualLoss()


def perceptual_loss(pred, target, normalize=True):

    # pred and target are N x C x H x W in the range [0,1]
    if pred.shape[1] == 1:
        pred_c3 = torch.cat([pred, pred, pred], dim=1)
        target_c3 = torch.cat([target, target, target], dim=1)
    else:
        pred_c3 = pred
        target_c3 = target

    dist = perceptual_loss_fn.forward(pred_c3, target_c3, normalize=True)

    return dist.mean()


def temporal_consistency_loss(image0, image1, processed0, processed1, flow01, alpha=50.0, output_images=False):
    """ Temporal loss, as described in Eq. (2) of the paper 'Learning Blind Video Temporal Consistency',
        Lai et al., ECCV'18.

        The temporal loss is the warping error between two processed frames (image reconstructions in E2VID),
        after the images have been aligned using the flow `flow01`.
        The input (ground truth) images `image0` and `image1` are used to estimate a visibility mask.

        :param image0: [N x C x H x W] input image 0
        :param image1: [N x C x H x W] input image 1
        :param processed0: [N x C x H x W] processed image (reconstruction) 0
        :param processed1: [N x C x H x W] processed image (reconstruction) 1
        :param flow01: [N x 2 x H x W] displacement map from image1 to image0
        :param alpha: used for computation of the visibility mask (default: 50.0)
    """
    t_width, t_height = image0.shape[3], image0.shape[2]
    xx, yy = torch.meshgrid(torch.arange(t_width), torch.arange(t_height))  # xx, yy -> WxH
    xx, yy = xx.to(image0.device), yy.to(image0.device)
    xx.transpose_(0, 1)
    yy.transpose_(0, 1)
    xx, yy = xx.float(), yy.float()

    flow01_x = flow01[:, 0, :, :]  # N x H x W
    flow01_y = flow01[:, 1, :, :]  # N x H x W

    warping_grid_x = xx + flow01_x  # N x H x W
    warping_grid_y = yy + flow01_y  # N x H x W

    # normalize warping grid to [-1,1]
    warping_grid_x = (2 * warping_grid_x / (t_width - 1)) - 1
    warping_grid_y = (2 * warping_grid_y / (t_height - 1)) - 1

    warping_grid = torch.stack(
        [warping_grid_x, warping_grid_y], dim=3)  # 1 x H x W x 2

    image0_warped_to1 = F.grid_sample(image0, warping_grid)
    visibility_mask = torch.exp(-alpha * (image1 - image0_warped_to1) ** 2)
    processed0_warped_to1 = F.grid_sample(processed0, warping_grid)

    tc_map = visibility_mask * torch.abs(processed1 - processed0_warped_to1) \
             / (torch.abs(processed1) + torch.abs(processed0_warped_to1) + 1e-5)

    tc_loss = tc_map.mean()

    if output_images:
        additional_output = {'image0_warped_to1': image0_warped_to1,
                             'processed0_warped_to1': processed0_warped_to1,
                             'visibility_mask': visibility_mask,
                             'error_map': tc_map}
        return tc_loss, additional_output

    else:
        return tc_loss

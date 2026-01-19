from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta
import numpy as np


def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    #print(torch.full((x.size()[0], num_classes), off_value, device=device))
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


class BaseMiniBatchBlending(metaclass=ABCMeta):
    """Base class for Image Aliasing."""

    def __init__(self, num_classes, smoothing=0.):
        self.num_classes = num_classes
        self.off_value = smoothing / self.num_classes
        self.on_value = 1. - smoothing + self.off_value

    @abstractmethod
    def do_blending(self, imgs, label, **kwargs):
        pass

    def __call__(self, imgs, label, **kwargs):
        """Blending data in a mini-batch.

        Images are float tensors with the shape of (B, N, C, H, W) for 2D
        recognizers or (B, N, C, T, H, W) for 3D recognizers.

        Besides, labels are converted from hard labels to soft labels.
        Hard labels are integer tensors with the shape of (B, 1) and all of the
        elements are in the range [0, num_classes - 1].
        Soft labels (probablity distribution over classes) are float tensors
        with the shape of (B, 1, num_classes) and all of the elements are in
        the range [0, 1].

        Args:
            imgs (torch.Tensor): Model input images, float tensor with the
                shape of (B, N, C, H, W) or (B, N, C, T, H, W).
            label (torch.Tensor): Hard labels, integer tensor with the shape
                of (B, 1) and all elements are in range [0, num_classes).
            kwargs (dict, optional): Other keyword argument to be used to
                blending imgs and labels in a mini-batch.

        Returns:
            mixed_imgs (torch.Tensor): Blending images, float tensor with the
                same shape of the input imgs.
            mixed_label (torch.Tensor): Blended soft labels, float tensor with
                the shape of (B, 1, num_classes) and all elements are in range
                [0, 1].
        """
        one_hot_label = one_hot(label, num_classes=self.num_classes, on_value=self.on_value, off_value=self.off_value, device=label.device)

        mixed_imgs, mixed_label = self.do_blending(imgs, one_hot_label,
                                                   **kwargs)

        return mixed_imgs, mixed_label
    
class BaseMiniBatchBlending2(metaclass=ABCMeta):
    """Base class for Image Aliasing."""

    def __init__(self, num_classes1, num_classes2, smoothing=0.):
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.off_value1 = smoothing / self.num_classes1
        self.on_value1 = 1. - smoothing + self.off_value1
        self.off_value2 = smoothing / self.num_classes2
        self.on_value2 = 1. - smoothing + self.off_value2

    @abstractmethod
    def do_blending(self, imgs, label1, label2, **kwargs):
        pass

    def __call__(self, imgs, label1, label2, **kwargs):
        """Blending data in a mini-batch.

        Images are float tensors with the shape of (B, N, C, H, W) for 2D
        recognizers or (B, N, C, T, H, W) for 3D recognizers.

        Besides, labels are converted from hard labels to soft labels.
        Hard labels are integer tensors with the shape of (B, 1) and all of the
        elements are in the range [0, num_classes - 1].
        Soft labels (probability distribution over classes) are float tensors
        with the shape of (B, num_classes) and all of the elements are in
        the range [0, 1].

        Args:
            imgs (torch.Tensor): Model input images, float tensor with the
                shape of (B, N, C, H, W) or (B, N, C, T, H, W).
            label1 (torch.Tensor): Hard labels for the first task, integer tensor
                with the shape of (B, 1) and all elements are in range [0, num_classes1).
            label2 (torch.Tensor): Hard labels for the second task, integer tensor
                with the shape of (B, 1) and all elements are in range [0, num_classes2).
            kwargs (dict, optional): Other keyword argument to be used to
                blending imgs and labels in a mini-batch.

        Returns:
            mixed_imgs (torch.Tensor): Blended images, float tensor with the
                same shape as the input imgs.
            mixed_label1 (torch.Tensor): Blended soft labels for the first task,
                float tensor with the shape of (B, num_classes1) and all elements are in range [0, 1].
            mixed_label2 (torch.Tensor): Blended soft labels for the second task,
                float tensor with the shape of (B, num_classes2) and all elements are in range [0, 1].
        """
        one_hot_label1 = one_hot(label1, num_classes=self.num_classes1, on_value=self.on_value1, off_value=self.off_value1, device=label1.device)
        one_hot_label2 = one_hot(label2, num_classes=self.num_classes2, on_value=self.on_value2, off_value=self.off_value2, device=label2.device)

        mixed_imgs, mixed_label1, mixed_label2 = self.do_blending(imgs, one_hot_label1, one_hot_label2, **kwargs)

        return mixed_imgs, mixed_label1, mixed_label2

class BaseMiniBatchBlending3(metaclass=ABCMeta):
    """Base class for Image Aliasing."""

    def __init__(self, num_classes1, num_classes2, smoothing=0.):
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.off_value1 = smoothing / self.num_classes1
        self.on_value1 = 1. - smoothing + self.off_value1
        self.off_value2 = smoothing / self.num_classes2
        self.on_value2 = 1. - smoothing + self.off_value2

    @abstractmethod
    def do_blending(self, imgs, imgs_diff, label1, label2, **kwargs):
        pass

    def __call__(self, imgs, imgs_diff, label1, label2, **kwargs):
        """Blending data in a mini-batch.

        Images are float tensors with the shape of (B, N, C, H, W) for 2D
        recognizers or (B, N, C, T, H, W) for 3D recognizers.

        Besides, labels are converted from hard labels to soft labels.
        Hard labels are integer tensors with the shape of (B, 1) and all of the
        elements are in the range [0, num_classes - 1].
        Soft labels (probability distribution over classes) are float tensors
        with the shape of (B, num_classes) and all of the elements are in
        the range [0, 1].

        Args:
            imgs (torch.Tensor): Model input images, float tensor with the
                shape of (B, N, C, H, W) or (B, N, C, T, H, W).
            label1 (torch.Tensor): Hard labels for the first task, integer tensor
                with the shape of (B, 1) and all elements are in range [0, num_classes1).
            label2 (torch.Tensor): Hard labels for the second task, integer tensor
                with the shape of (B, 1) and all elements are in range [0, num_classes2).
            kwargs (dict, optional): Other keyword argument to be used to
                blending imgs and labels in a mini-batch.

        Returns:
            mixed_imgs (torch.Tensor): Blended images, float tensor with the
                same shape as the input imgs.
            mixed_label1 (torch.Tensor): Blended soft labels for the first task,
                float tensor with the shape of (B, num_classes1) and all elements are in range [0, 1].
            mixed_label2 (torch.Tensor): Blended soft labels for the second task,
                float tensor with the shape of (B, num_classes2) and all elements are in range [0, 1].
        """
        one_hot_label1 = one_hot(label1, num_classes=self.num_classes1, on_value=self.on_value1, off_value=self.off_value1, device=label1.device)
        one_hot_label2 = one_hot(label2, num_classes=self.num_classes2, on_value=self.on_value2, off_value=self.off_value2, device=label2.device)

        mixed_imgs,mixed_imgs_diff, mixed_label1, mixed_label2 = self.do_blending(imgs, imgs_diff, one_hot_label1, one_hot_label2, **kwargs)

        return mixed_imgs, mixed_imgs_diff, mixed_label1, mixed_label2



class MixupBlending(BaseMiniBatchBlending):
    """Implementing Mixup in a mini-batch.

    This module is proposed in `mixup: Beyond Empirical Risk Minimization
    <https://arxiv.org/abs/1710.09412>`_.
    Code Reference https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/utils/mixup.py # noqa

    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, alpha=.2, smoothing=0.):
        super().__init__(num_classes=num_classes, smoothing=smoothing)
        self.beta = Beta(alpha, alpha)

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with mixup."""
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.beta.sample()
        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)

        mixed_imgs = lam * imgs + (1 - lam) * imgs[rand_index, :]
        mixed_label = lam * label + (1 - lam) * label[rand_index, :]

        return mixed_imgs, mixed_label


class CutmixBlending(BaseMiniBatchBlending):
    """Implementing Cutmix in a mini-batch.
    This module is proposed in `CutMix: Regularization Strategy to Train Strong
    Classifiers with Localizable Features <https://arxiv.org/abs/1905.04899>`_.
    Code Reference https://github.com/clovaai/CutMix-PyTorch
    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, alpha=.2, smoothing=0.):
        super().__init__(num_classes=num_classes, smoothing=smoothing)
        self.beta = Beta(alpha, alpha)

    @staticmethod
    def rand_bbox(img_size, lam):
        """Generate a random boudning box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = torch.tensor(int(w * cut_rat))
        cut_h = torch.tensor(int(h * cut_rat))

        # uniform
        cx = torch.randint(w, (1, ))[0]
        cy = torch.randint(h, (1, ))[0]

        bbx1 = torch.clamp(cx - cut_w // 2, 0, w)
        bby1 = torch.clamp(cy - cut_h // 2, 0, h)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, w)
        bby2 = torch.clamp(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with cutmix."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix {kwargs}'

        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.beta.sample()

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)
        imgs[:, ..., bby1:bby2, bbx1:bbx2] = imgs[rand_index, ..., bby1:bby2,
                                                  bbx1:bbx2]
        lam = 1 - (1.0 * (bbx2 - bbx1) * (bby2 - bby1) /
                   (imgs.size()[-1] * imgs.size()[-2]))

        label = lam * label + (1 - lam) * label[rand_index, :]

        return imgs, label


class LabelSmoothing(BaseMiniBatchBlending):
    def do_blending(self, imgs, label, **kwargs):
        return imgs, label


class CutmixMixupBlending(BaseMiniBatchBlending):
    def __init__(self, num_classes=400, smoothing=0.1, mixup_alpha=.8, cutmix_alpha=1, switch_prob=0.5):
        super().__init__(num_classes=num_classes, smoothing=smoothing)
        self.mixup_beta = Beta(mixup_alpha, mixup_alpha)
        self.cutmix_beta = Beta(cutmix_alpha, cutmix_alpha)
        self.switch_prob = switch_prob

    @staticmethod
    def rand_bbox(img_size, lam):
        """Generate a random boudning box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = torch.tensor(int(w * cut_rat))
        cut_h = torch.tensor(int(h * cut_rat))

        # uniform
        cx = torch.randint(w, (1, ))[0]
        cy = torch.randint(h, (1, ))[0]

        bbx1 = torch.clamp(cx - cut_w // 2, 0, w)
        bby1 = torch.clamp(cy - cut_h // 2, 0, h)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, w)
        bby2 = torch.clamp(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_cutmix(self, imgs, label, **kwargs):
        """Blending images with cutmix."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix {kwargs}'

        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.cutmix_beta.sample()

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)
        imgs[:, ..., bby1:bby2, bbx1:bbx2] = imgs[rand_index, ..., bby1:bby2,
                                                  bbx1:bbx2]
        lam = 1 - (1.0 * (bbx2 - bbx1) * (bby2 - bby1) /
                   (imgs.size()[-1] * imgs.size()[-2]))

        label = lam * label + (1 - lam) * label[rand_index, :]
        return imgs, label

    def do_mixup(self, imgs, label, **kwargs):
        """Blending images with mixup."""
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.mixup_beta.sample()
        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)

        mixed_imgs = lam * imgs + (1 - lam) * imgs[rand_index, :]
        mixed_label = lam * label + (1 - lam) * label[rand_index, :]

        return mixed_imgs, mixed_label

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with MViT style. Cutmix for half for mixup for the other half."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix_half_mixup {kwargs}'

        if np.random.rand() < self.switch_prob :
            return self.do_cutmix(imgs, label)
        else:
            return self.do_mixup(imgs, label)



class CutmixMixupBlending2(BaseMiniBatchBlending2):
    def __init__(self, num_classes1=120, num_classes2=106, smoothing=0.1, mixup_alpha=.8, cutmix_alpha=1, switch_prob=0.5):
        super().__init__(num_classes1=num_classes1, num_classes2=num_classes2, smoothing=smoothing)
        self.mixup_beta = Beta(mixup_alpha, mixup_alpha)
        self.cutmix_beta = Beta(cutmix_alpha, cutmix_alpha)
        self.switch_prob = switch_prob

    @staticmethod
    def rand_bbox(img_size, lam):
        """Generate a random bounding box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = int(w * cut_rat.item())
        cut_h = int(h * cut_rat.item())

        cx = torch.randint(w, (1,)).item()
        cy = torch.randint(h, (1,)).item()

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_cutmix(self, imgs, label1, label2, **kwargs):
        """Blending images with cutmix."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix {kwargs}'

        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.cutmix_beta.sample()  # do not convert to .item()

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)

        imgs[:, :, bby1:bby2, bbx1:bbx2] = imgs[rand_index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))

        label1 = lam * label1 + (1 - lam) * label1[rand_index, :]
        label2 = lam * label2 + (1 - lam) * label2[rand_index, :]
        return imgs, label1, label2

    def do_mixup(self, imgs, label1, label2, **kwargs):
        """Blending images with mixup."""
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.mixup_beta.sample().item()  # this can be a scalar
        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)

        mixed_imgs = lam * imgs + (1 - lam) * imgs[rand_index, :]
        mixed_label1 = lam * label1 + (1 - lam) * label1[rand_index, :]
        mixed_label2 = lam * label2 + (1 - lam) * label2[rand_index, :]

        return mixed_imgs, mixed_label1, mixed_label2

    def do_blending(self, imgs, label1, label2, **kwargs):
        """Blending images with MViT style. Cutmix for half and mixup for the other half."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix_half_mixup {kwargs}'

        if np.random.rand() < self.switch_prob:
            return self.do_cutmix(imgs, label1, label2)
        else:
            return self.do_mixup(imgs, label1, label2)





class CutmixMixupBlending3(BaseMiniBatchBlending3):
    def __init__(self, num_classes1=120, num_classes2=106, smoothing=0.1, mixup_alpha=.8, cutmix_alpha=1, switch_prob=0.5):
        super().__init__(num_classes1=num_classes1, num_classes2=num_classes2, smoothing=smoothing)
        self.mixup_beta = Beta(mixup_alpha, mixup_alpha)
        self.cutmix_beta = Beta(cutmix_alpha, cutmix_alpha)
        self.switch_prob = switch_prob

    @staticmethod
    def rand_bbox(img_size, lam):
        """Generate a random bounding box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = int(w * cut_rat.item())
        cut_h = int(h * cut_rat.item())

        cx = torch.randint(w, (1,)).item()
        cy = torch.randint(h, (1,)).item()

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_cutmix(self, imgs, imgs_diff, label1, label2, **kwargs):
        """Blending images with cutmix."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix {kwargs}'

        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.cutmix_beta.sample()  # do not convert to .item()

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)

        imgs[:, :, bby1:bby2, bbx1:bbx2] = imgs[rand_index, :, bby1:bby2, bbx1:bbx2]
        imgs_diff[:, :, bby1:bby2, bbx1:bbx2] = imgs_diff[rand_index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))

        label1 = lam * label1 + (1 - lam) * label1[rand_index, :]
        label2 = lam * label2 + (1 - lam) * label2[rand_index, :]
        return imgs, imgs_diff, label1, label2

    def do_mixup(self, imgs,imgs_diff, label1, label2, **kwargs):
        """Blending images with mixup."""
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.mixup_beta.sample().item()  # this can be a scalar
        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)

        mixed_imgs_diff = lam * imgs_diff + (1 - lam) * imgs_diff[rand_index, :]
        mixed_imgs = lam * imgs + (1 - lam) * imgs[rand_index, :]
        mixed_label1 = lam * label1 + (1 - lam) * label1[rand_index, :]
        mixed_label2 = lam * label2 + (1 - lam) * label2[rand_index, :]

        return mixed_imgs, mixed_imgs_diff, mixed_label1, mixed_label2

    def do_blending(self, imgs,imgs_diff, label1, label2, **kwargs):
        """Blending images with MViT style. Cutmix for half and mixup for the other half."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix_half_mixup {kwargs}'

        if np.random.rand() < self.switch_prob:
            return self.do_cutmix(imgs, imgs_diff,label1, label2)
        else:
            return self.do_mixup(imgs, imgs_diff,label1, label2)



class BaseMiniBatchBlending4(metaclass=ABCMeta):
    """Base class for Image Aliasing."""

    def __init__(self, num_classes1, num_classes2, smoothing=0.):
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.off_value1 = smoothing / self.num_classes1
        self.on_value1 = 1. - smoothing + self.off_value1
        self.off_value2 = smoothing / self.num_classes2
        self.on_value2 = 1. - smoothing + self.off_value2

    @abstractmethod
    def do_blending(self, imgs1, imgs2, imgs3, label1, label2, **kwargs):
        pass

    def __call__(self, imgs1, imgs2, imgs3, label1, label2, **kwargs):
        one_hot_label1 = one_hot(label1, num_classes=self.num_classes1, on_value=self.on_value1, off_value=self.off_value1, device=label1.device)
        one_hot_label2 = one_hot(label2, num_classes=self.num_classes2, on_value=self.on_value2, off_value=self.off_value2, device=label2.device)

        mixed_imgs1, mixed_imgs2, mixed_imgs3, mixed_label1, mixed_label2 = self.do_blending(imgs1, imgs2, imgs3, one_hot_label1, one_hot_label2, **kwargs)

        return mixed_imgs1, mixed_imgs2, mixed_imgs3, mixed_label1, mixed_label2

class BaseMiniBatchBlending4_real60(metaclass=ABCMeta):
    """Base class for Image Aliasing."""

    def __init__(self, num_classes1, num_classes2,num_uuu,num_vvv,num_www,num_xxx,num_yyy, smoothing=0.):
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.num_uuu = num_uuu
        self.num_vvv = num_vvv
        self.num_www = num_www
        self.num_xxx = num_xxx
        self.num_yyy = num_yyy

        self.off_value1 = smoothing / self.num_classes1
        self.on_value1 = 1. - smoothing + self.off_value1
        self.off_value2 = smoothing / self.num_classes2
        self.on_value2 = 1. - smoothing + self.off_value2

        self.off_uuu = smoothing / self.num_uuu
        self.on_uuu = 1. - smoothing + self.off_uuu
        self.off_vvv = smoothing / self.num_vvv
        self.on_vvv = 1. - smoothing + self.off_vvv
        self.off_www = smoothing / self.num_www
        self.on_www = 1. - smoothing + self.off_www
        self.off_xxx = smoothing / self.num_xxx
        self.on_xxx = 1. - smoothing + self.off_xxx
        self.off_yyy = smoothing / self.num_yyy
        self.on_yyy = 1. - smoothing + self.off_yyy

    @abstractmethod
    def do_blending(self, imgs1, imgs2, imgs3, label1, label2,label_u,label_v,label_w,label_x,label_y, **kwargs):
        pass

    def __call__(self, imgs1, imgs2, imgs3, label1, label2,label_u,label_v,label_w,label_x,label_y, **kwargs):
        one_hot_label1 = one_hot(label1, num_classes=self.num_classes1, on_value=self.on_value1, off_value=self.off_value1, device=label1.device)
        one_hot_label2 = one_hot(label2, num_classes=self.num_classes2, on_value=self.on_value2, off_value=self.off_value2, device=label2.device)

        one_hot_label_u = one_hot(label_u, num_classes=self.num_uuu, on_value=self.on_uuu, off_value=self.off_uuu, device=label_u.device)
        one_hot_label_v = one_hot(label_v, num_classes=self.num_vvv, on_value=self.on_vvv, off_value=self.off_vvv, device=label_v.device)
        one_hot_label_w = one_hot(label_w, num_classes=self.num_www, on_value=self.on_www, off_value=self.off_www, device=label_w.device)
        one_hot_label_x = one_hot(label_x, num_classes=self.num_xxx, on_value=self.on_xxx, off_value=self.off_xxx, device=label_x.device)
        one_hot_label_y = one_hot(label_y, num_classes=self.num_yyy, on_value=self.on_yyy, off_value=self.off_yyy, device=label_y.device)

        mixed_imgs1, mixed_imgs2, mixed_imgs3, mixed_label1, mixed_label2,mixed_label_u,mixed_label_v,mixed_label_w,mixed_label_x,mixed_label_y= \
            self.do_blending(imgs1, imgs2, imgs3, one_hot_label1, one_hot_label2,one_hot_label_u,one_hot_label_v,one_hot_label_w,one_hot_label_x,one_hot_label_y, **kwargs)                                                               

        return mixed_imgs1, mixed_imgs2, mixed_imgs3, mixed_label1, mixed_label2, mixed_label_u,mixed_label_v,mixed_label_w,mixed_label_x,mixed_label_y


class CutmixMixupBlending4_real60(BaseMiniBatchBlending4_real60):
    def __init__(self, num_classes1=60, num_classes2=22,num_uuu=2,num_vvv=5,num_www=2,num_xxx=9,num_yyy=12,smoothing=0.1, mixup_alpha=.8, cutmix_alpha=1, switch_prob=0.5):
        super().__init__(num_classes1=num_classes1, num_classes2=num_classes2,num_uuu=num_uuu,num_vvv=num_vvv,num_www=num_www,num_xxx=num_xxx,num_yyy=num_yyy, smoothing=smoothing)
        self.mixup_beta = Beta(mixup_alpha, mixup_alpha)
        self.cutmix_beta = Beta(cutmix_alpha, cutmix_alpha)
        self.switch_prob = switch_prob

    @staticmethod
    def rand_bbox(img_size, lam):
        """Generate a random bounding box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = int(w * cut_rat.item())
        cut_h = int(h * cut_rat.item())

        cx = torch.randint(w, (1,)).item()
        cy = torch.randint(h, (1,)).item()

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_cutmix(self, imgs1, imgs2, imgs3, label1, label2,label_u,label_v,label_w,label_x,label_y, **kwargs):
        """Blending images with cutmix."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix {kwargs}'

        batch_size = imgs1.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.cutmix_beta.sample()  # do not convert to .item()

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs1.size(), lam)

        imgs1[:, :, bby1:bby2, bbx1:bbx2] = imgs1[rand_index, :, bby1:bby2, bbx1:bbx2]
        imgs2[:, :, bby1:bby2, bbx1:bbx2] = imgs2[rand_index, :, bby1:bby2, bbx1:bbx2]
        imgs3[:, :, bby1:bby2, bbx1:bbx2] = imgs3[rand_index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs1.size()[-1] * imgs1.size()[-2]))

        label1 = lam * label1 + (1 - lam) * label1[rand_index, :]
        label2 = lam * label2 + (1 - lam) * label2[rand_index, :]

        label_u = lam * label_u + (1 - lam) * label_u[rand_index, :]
        label_v = lam * label_v + (1 - lam) * label_v[rand_index, :]
        label_w = lam * label_w + (1 - lam) * label_w[rand_index, :]
        label_x = lam * label_x + (1 - lam) * label_x[rand_index, :]
        label_y = lam * label_y + (1 - lam) * label_y[rand_index, :]

        return imgs1, imgs2, imgs3, label1, label2,label_u,label_v,label_w,label_x,label_y

    def do_mixup(self, imgs1, imgs2, imgs3, label1, label2,label_u,label_v,label_w,label_x,label_y, **kwargs):
        """Blending images with mixup."""
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.mixup_beta.sample().item()  # this can be a scalar
        batch_size = imgs1.size(0)
        rand_index = torch.randperm(batch_size)
        
        # print(f"imgs1 shape: {imgs1.shape}, rand_index shape: {rand_index.shape}, label1 shape: {label1.shape}, label2 shape: {label2.shape}")
        mixed_imgs1 = lam * imgs1 + (1 - lam) * imgs1[rand_index, :]
        mixed_imgs2 = lam * imgs2 + (1 - lam) * imgs2[rand_index, :]
        mixed_imgs3 = lam * imgs3 + (1 - lam) * imgs3[rand_index, :]
        mixed_label1 = lam * label1 + (1 - lam) * label1[rand_index, :]
        mixed_label2 = lam * label2 + (1 - lam) * label2[rand_index, :]

        mixed_label_u = lam * label_u + (1 - lam) * label_u[rand_index, :]
        mixed_label_v = lam * label_v + (1 - lam) * label_v[rand_index, :]
        mixed_label_w = lam * label_w + (1 - lam) * label_w[rand_index, :]
        mixed_label_x = lam * label_x + (1 - lam) * label_x[rand_index, :]
        mixed_label_y = lam * label_y + (1 - lam) * label_y[rand_index, :]

        return mixed_imgs1, mixed_imgs2, mixed_imgs3, mixed_label1, mixed_label2, mixed_label_u, mixed_label_v, mixed_label_w, mixed_label_x, mixed_label_y

    def do_blending(self, imgs1, imgs2, imgs3, label1, label2, label_u,label_v,label_w,label_x,label_y, **kwargs):
        """Blending images with MViT style. Cutmix for half and mixup for the other half."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix_half_mixup {kwargs}'

        if np.random.rand() < self.switch_prob:
            return self.do_cutmix(imgs1, imgs2, imgs3, label1, label2, label_u,label_v,label_w,label_x,label_y)
        else:
            return self.do_mixup(imgs1, imgs2, imgs3, label1, label2, label_u,label_v,label_w,label_x,label_y)


class CutmixMixupBlending4(BaseMiniBatchBlending4):
    def __init__(self, num_classes1=120, num_classes2=106, smoothing=0.1, mixup_alpha=.8, cutmix_alpha=1, switch_prob=0.5):
        super().__init__(num_classes1=num_classes1, num_classes2=num_classes2, smoothing=smoothing)
        self.mixup_beta = Beta(mixup_alpha, mixup_alpha)
        self.cutmix_beta = Beta(cutmix_alpha, cutmix_alpha)
        self.switch_prob = switch_prob

    @staticmethod
    def rand_bbox(img_size, lam):
        """Generate a random bounding box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = int(w * cut_rat.item())
        cut_h = int(h * cut_rat.item())

        cx = torch.randint(w, (1,)).item()
        cy = torch.randint(h, (1,)).item()

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_cutmix(self, imgs1, imgs2, imgs3, label1, label2, **kwargs):
        """Blending images with cutmix."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix {kwargs}'

        batch_size = imgs1.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.cutmix_beta.sample()  # do not convert to .item()

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs1.size(), lam)

        imgs1[:, :, bby1:bby2, bbx1:bbx2] = imgs1[rand_index, :, bby1:bby2, bbx1:bbx2]
        imgs2[:, :, bby1:bby2, bbx1:bbx2] = imgs2[rand_index, :, bby1:bby2, bbx1:bbx2]
        imgs3[:, :, bby1:bby2, bbx1:bbx2] = imgs3[rand_index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs1.size()[-1] * imgs1.size()[-2]))


        label1 = lam * label1 + (1 - lam) * label1[rand_index, :]
        label2 = lam * label2 + (1 - lam) * label2[rand_index, :]
        return imgs1, imgs2, imgs3, label1, label2

    def do_mixup(self, imgs1, imgs2, imgs3, label1, label2, **kwargs):
        """Blending images with mixup."""
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.mixup_beta.sample().item()  # this can be a scalar
        batch_size = imgs1.size(0)
        rand_index = torch.randperm(batch_size)

        mixed_imgs1 = lam * imgs1 + (1 - lam) * imgs1[rand_index, :]
        mixed_imgs2 = lam * imgs2 + (1 - lam) * imgs2[rand_index, :]
        mixed_imgs3 = lam * imgs3 + (1 - lam) * imgs3[rand_index, :]
        mixed_label1 = lam * label1 + (1 - lam) * label1[rand_index, :]
        mixed_label2 = lam * label2 + (1 - lam) * label2[rand_index, :]

        return mixed_imgs1, mixed_imgs2, mixed_imgs3, mixed_label1, mixed_label2

    def do_blending(self, imgs1, imgs2, imgs3, label1, label2, **kwargs):
        """Blending images with MViT style. Cutmix for half and mixup for the other half."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix_half_mixup {kwargs}'

        if np.random.rand() < self.switch_prob:
            return self.do_cutmix(imgs1, imgs2, imgs3, label1, label2)
        else:
            return self.do_mixup(imgs1, imgs2, imgs3, label1, label2)



class BaseMiniBatchBlending6(metaclass=ABCMeta):
    """Base class for Image Aliasing."""

    def __init__(self, num_classes1, num_classes2, smoothing=0.):
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.off_value1 = smoothing / self.num_classes1
        self.on_value1 = 1. - smoothing + self.off_value1
        self.off_value2 = smoothing / self.num_classes2
        self.on_value2 = 1. - smoothing + self.off_value2

    @abstractmethod
    def do_blending(self, imgs1, imgs2, imgs3, imgs4, imgs5, label1, label2, **kwargs):
        pass

    def __call__(self, imgs1, imgs2, imgs3, imgs4, imgs5, label1, label2, **kwargs):
        one_hot_label1 = one_hot(label1, num_classes=self.num_classes1, on_value=self.on_value1, off_value=self.off_value1, device=label1.device)
        one_hot_label2 = one_hot(label2, num_classes=self.num_classes2, on_value=self.on_value2, off_value=self.off_value2, device=label2.device)

        mixed_imgs1, mixed_imgs2, mixed_imgs3, mixed_imgs4, mixed_imgs5, mixed_label1, mixed_label2 = self.do_blending(imgs1, imgs2, imgs3, imgs4, imgs5, one_hot_label1, one_hot_label2, **kwargs)

        return mixed_imgs1, mixed_imgs2, mixed_imgs3, mixed_imgs4, mixed_imgs5, mixed_label1, mixed_label2

class CutmixMixupBlending6(BaseMiniBatchBlending6):
    def __init__(self, num_classes1=120, num_classes2=106, smoothing=0.1, mixup_alpha=.8, cutmix_alpha=1, switch_prob=0.5):
        super().__init__(num_classes1=num_classes1, num_classes2=num_classes2, smoothing=smoothing)
        self.mixup_beta = Beta(mixup_alpha, mixup_alpha)
        self.cutmix_beta = Beta(cutmix_alpha, cutmix_alpha)
        self.switch_prob = switch_prob

    @staticmethod
    def rand_bbox(img_size, lam):
        """Generate a random bounding box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = int(w * cut_rat.item())
        cut_h = int(h * cut_rat.item())

        cx = torch.randint(w, (1,)).item()
        cy = torch.randint(h, (1,)).item()

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_cutmix(self, imgs1, imgs2, imgs3, imgs4, imgs5, label1, label2, **kwargs):
        """Blending images with cutmix."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix {kwargs}'

        batch_size = imgs1.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.cutmix_beta.sample()  # do not convert to .item()

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs1.size(), lam)

        imgs1[:, :, bby1:bby2, bbx1:bbx2] = imgs1[rand_index, :, bby1:bby2, bbx1:bbx2]
        imgs2[:, :, bby1:bby2, bbx1:bbx2] = imgs2[rand_index, :, bby1:bby2, bbx1:bbx2]
        imgs3[:, :, bby1:bby2, bbx1:bbx2] = imgs3[rand_index, :, bby1:bby2, bbx1:bbx2]
        imgs4[:, :, bby1:bby2, bbx1:bbx2] = imgs4[rand_index, :, bby1:bby2, bbx1:bbx2]
        imgs5[:, :, bby1:bby2, bbx1:bbx2] = imgs5[rand_index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs1.size()[-1] * imgs1.size()[-2]))

        label1 = lam * label1 + (1 - lam) * label1[rand_index, :]
        label2 = lam * label2 + (1 - lam) * label2[rand_index, :]
        return imgs1, imgs2, imgs3, imgs4, imgs5, label1, label2

    def do_mixup(self, imgs1, imgs2, imgs3, imgs4, imgs5, label1, label2, **kwargs):
        """Blending images with mixup."""
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.mixup_beta.sample().item()  # this can be a scalar
        batch_size = imgs1.size(0)
        rand_index = torch.randperm(batch_size)

        mixed_imgs1 = lam * imgs1 + (1 - lam) * imgs1[rand_index, :]
        mixed_imgs2 = lam * imgs2 + (1 - lam) * imgs2[rand_index, :]
        mixed_imgs3 = lam * imgs3 + (1 - lam) * imgs3[rand_index, :]
        mixed_imgs4 = lam * imgs4 + (1 - lam) * imgs4[rand_index, :]
        mixed_imgs5 = lam * imgs5 + (1 - lam) * imgs5[rand_index, :]
        mixed_label1 = lam * label1 + (1 - lam) * label1[rand_index, :]
        mixed_label2 = lam * label2 + (1 - lam) * label2[rand_index, :]

        return mixed_imgs1, mixed_imgs2, mixed_imgs3, mixed_imgs4, mixed_imgs5, mixed_label1, mixed_label2

    def do_blending(self, imgs1, imgs2, imgs3, imgs4, imgs5, label1, label2, **kwargs):
        """Blending images with MViT style. Cutmix for half and mixup for the other half."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix_half_mixup {kwargs}'

        if np.random.rand() < self.switch_prob:
            return self.do_cutmix(imgs1, imgs2, imgs3, imgs4, imgs5, label1, label2)
        else:
            return self.do_mixup(imgs1, imgs2, imgs3, imgs4, imgs5, label1, label2)

class BaseMiniBatchBlending7(metaclass=ABCMeta):
    """Base class for Image Aliasing."""

    def __init__(self, num_classes1, num_classes2, smoothing=0.):
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.off_value1 = smoothing / self.num_classes1
        self.on_value1 = 1. - smoothing + self.off_value1
        self.off_value2 = smoothing / self.num_classes2
        self.on_value2 = 1. - smoothing + self.off_value2

    @abstractmethod
    def do_blending(self, imgs1, imgs2, imgs3, imgs4, imgs5, imgs6, label1, label2, **kwargs):
        pass

    def __call__(self, imgs1, imgs2, imgs3, imgs4, imgs5, imgs6, label1, label2, **kwargs):
        """Blending data in a mini-batch.

        Images are float tensors with the shape of (B, N, C, H, W) for 2D
        recognizers or (B, N, C, T, H, W) for 3D recognizers.

        Besides, labels are converted from hard labels to soft labels.
        Hard labels are integer tensors with the shape of (B, 1) and all of the
        elements are in the range [0, num_classes - 1].
        Soft labels (probability distribution over classes) are float tensors
        with the shape of (B, num_classes) and all of the elements are in
        the range [0, 1].

        Args:
            imgs1, imgs2, imgs3, imgs4, imgs5, imgs6 (torch.Tensor): Model input images, float tensor with the
                shape of (B, N, C, H, W) or (B, N, C, T, H, W).
            label1 (torch.Tensor): Hard labels for the first task, integer tensor
                with the shape of (B, 1) and all elements are in range [0, num_classes1).
            label2 (torch.Tensor): Hard labels for the second task, integer tensor
                with the shape of (B, 1) and all elements are in range [0, num_classes2).
            kwargs (dict, optional): Other keyword argument to be used to
                blending imgs and labels in a mini-batch.

        Returns:
            mixed_imgs (torch.Tensor): Blended images, float tensor with the
                same shape as the input imgs.
            mixed_label1 (torch.Tensor): Blended soft labels for the first task,
                float tensor with the shape of (B, num_classes1) and all elements are in range [0, 1].
            mixed_label2 (torch.Tensor): Blended soft labels for the second task,
                float tensor with the shape of (B, num_classes2) and all elements are in range [0, 1].
        """
        one_hot_label1 = one_hot(label1, num_classes=self.num_classes1, on_value=self.on_value1, off_value=self.off_value1, device=label1.device)
        one_hot_label2 = one_hot(label2, num_classes=self.num_classes2, on_value=self.on_value2, off_value=self.off_value2, device=label2.device)

        mixed_imgs1, mixed_imgs2, mixed_imgs3, mixed_imgs4, mixed_imgs5, mixed_imgs6, mixed_label1, mixed_label2 = self.do_blending(
            imgs1, imgs2, imgs3, imgs4, imgs5, imgs6, one_hot_label1, one_hot_label2, **kwargs
        )

        return mixed_imgs1, mixed_imgs2, mixed_imgs3, mixed_imgs4, mixed_imgs5, mixed_imgs6, mixed_label1, mixed_label2

class CutmixMixupBlending7(BaseMiniBatchBlending7):
    def __init__(self, num_classes1=120, num_classes2=106, smoothing=0.1, mixup_alpha=.8, cutmix_alpha=1, switch_prob=0.5):
        super().__init__(num_classes1=num_classes1, num_classes2=num_classes2, smoothing=smoothing)
        self.mixup_beta = Beta(mixup_alpha, mixup_alpha)
        self.cutmix_beta = Beta(cutmix_alpha, cutmix_alpha)
        self.switch_prob = switch_prob

    @staticmethod
    def rand_bbox(img_size, lam):
        """Generate a random bounding box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = int(w * cut_rat.item())
        cut_h = int(h * cut_rat.item())

        cx = torch.randint(w, (1,)).item()
        cy = torch.randint(h, (1,)).item()

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_cutmix(self, imgs1, imgs2, imgs3, imgs4, imgs5, imgs6, label1, label2, **kwargs):
        """Blending images with cutmix."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix {kwargs}'

        batch_size = imgs1.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.cutmix_beta.sample()  # do not convert to .item()

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs1.size(), lam)

        imgs1[:, :, bby1:bby2, bbx1:bbx2] = imgs1[rand_index, :, bby1:bby2, bbx1:bbx2]
        imgs2[:, :, bby1:bby2, bbx1:bbx2] = imgs2[rand_index, :, bby1:bby2, bbx1:bbx2]
        imgs3[:, :, bby1:bby2, bbx1:bbx2] = imgs3[rand_index, :, bby1:bby2, bbx1:bbx2]
        imgs4[:, :, bby1:bby2, bbx1:bbx2] = imgs4[rand_index, :, bby1:bby2, bbx1:bbx2]
        imgs5[:, :, bby1:bby2, bbx1:bbx2] = imgs5[rand_index, :, bby1:bby2, bbx1:bbx2]
        imgs6[:, :, bby1:bby2, bbx1:bbx2] = imgs6[rand_index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs1.size()[-1] * imgs1.size()[-2]))

        label1 = lam * label1 + (1 - lam) * label1[rand_index, :]
        label2 = lam * label2 + (1 - lam) * label2[rand_index, :]
        return imgs1, imgs2, imgs3, imgs4, imgs5, imgs6, label1, label2

    def do_mixup(self, imgs1, imgs2, imgs3, imgs4, imgs5, imgs6, label1, label2, **kwargs):
        """Blending images with mixup."""
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.mixup_beta.sample().item()  # this can be a scalar
        batch_size = imgs1.size(0)
        rand_index = torch.randperm(batch_size)

        mixed_imgs1 = lam * imgs1 + (1 - lam) * imgs1[rand_index, :]
        mixed_imgs2 = lam * imgs2 + (1 - lam) * imgs2[rand_index, :]
        mixed_imgs3 = lam * imgs3 + (1 - lam) * imgs3[rand_index, :]
        mixed_imgs4 = lam * imgs4 + (1 - lam) * imgs4[rand_index, :]
        mixed_imgs5 = lam * imgs5 + (1 - lam) * imgs5[rand_index, :]
        mixed_imgs6 = lam * imgs6 + (1 - lam) * imgs6[rand_index, :]
        mixed_label1 = lam * label1 + (1 - lam) * label1[rand_index, :]
        mixed_label2 = lam * label2 + (1 - lam) * label2[rand_index, :]

        return mixed_imgs1, mixed_imgs2, mixed_imgs3, mixed_imgs4, mixed_imgs5, mixed_imgs6, mixed_label1, mixed_label2

    def do_blending(self, imgs1, imgs2, imgs3, imgs4, imgs5, imgs6, label1, label2, **kwargs):
        """Blending images with MViT style. Cutmix for half and mixup for the other half."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix_half_mixup {kwargs}'

        if np.random.rand() < self.switch_prob:
            return self.do_cutmix(imgs1, imgs2, imgs3, imgs4, imgs5, imgs6, label1, label2)
        else:
            return self.do_mixup(imgs1, imgs2, imgs3, imgs4, imgs5, imgs6, label1, label2)

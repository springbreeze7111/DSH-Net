import torch
from torch import nn
import torch.nn.functional as F

class HungarianMatcher3D(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    def compute_cls_loss(self, inputs, targets):
        """ Classification loss (NLL)
            implemented in compute_loss()
        """
        raise NotImplementedError

    def compute_dice_loss(self, inputs, targets):
        """ mask dice loss
            inputs (B*K, C, H, W)
            target (B*K, D, H, W)
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        num_masks = len(inputs)

        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks


def compute_loss_hungarian(outputs, targets, idx, matcher, num_classes, point_rend=False, num_points=12544,
                           oversample_ratio=3.0, importance_sample_ratio=0.75, no_object_weight=None,
                           cost_weight=[2, 5, 5]):
    """output is a dict only contain keys ['pred_masks', 'pred_logits'] """
    # outputs_without_aux = {k: v for k, v in output.items() if k != "aux_outputs"}

    indices = matcher(outputs, targets)
    src_idx = matcher._get_src_permutation_idx(indices)  # return a tuple of (batch_idx, src_idx)
    tgt_idx = matcher._get_tgt_permutation_idx(indices)  # return a tuple of (batch_idx, tgt_idx)
    assert len(tgt_idx[0]) == sum([len(t["masks"]) for t in targets])  # verify that all masks of (K1, K2, ..) are used

    # step2 : compute mask loss
    src_masks = outputs["pred_masks"]
    src_masks = src_masks[src_idx]  # [len(src_idx[0]), D, H, W] -> (K1+K2+..., D, H, W)
    target_masks = torch.cat([t["masks"] for t in targets], dim=0)  # (K1+K2+..., D, H, W) actually
    src_masks = src_masks[:, None]  # [K..., 1, D, H, W]
    target_masks = target_masks[:, None]

    if point_rend:  # only calculate hard example
        with torch.no_grad():
            # num_points=12544 config in cityscapes

            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks.float(),
                lambda logits: calculate_uncertainty(logits),
                num_points,
                oversample_ratio,
                importance_sample_ratio,
            )  # [K, num_points=12544, 3]

            point_labels = point_sample_3d(
                target_masks.float(),
                point_coords.float(),
                align_corners=False,
            ).squeeze(1)  # [K, 12544]

        point_logits = point_sample_3d(
            src_masks.float(),
            point_coords.float(),
            align_corners=False,
        ).squeeze(1)  # [K, 12544]

        src_masks, target_masks = point_logits, point_labels

    loss_mask_ce = matcher.compute_ce_loss(src_masks, target_masks)
    loss_mask_dice = matcher.compute_dice_loss(src_masks, target_masks)

    # step3: compute class loss
    src_logits = outputs["pred_logits"].float()  # (B, num_query, num_class+1)
    target_classes_o = torch.cat([t["labels"] for t in targets], dim=0)  # (K1+K2+, )
    target_classes = torch.full(
        src_logits.shape[:2], num_classes, dtype=torch.int64, device=src_logits.device
    )  # (B, num_query, num_class+1)
    target_classes[src_idx] = target_classes_o

    if no_object_weight is not None:
        empty_weight = torch.ones(num_classes + 1).to(src_logits.device)
        empty_weight[-1] = no_object_weight
        loss_cls = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
    else:
        loss_cls = F.cross_entropy(src_logits.transpose(1, 2), target_classes)

    loss = (cost_weight[0] / 10) * loss_cls + (cost_weight[1] / 10) * loss_mask_ce + (
                cost_weight[2] / 10) * loss_mask_dice  # 2:5:5, like hungarian matching
    # print("idx {}, loss {}, loss_cls {}, loss_mask_ce {}, loss_mask_dice {}".format(idx, loss, loss_cls, loss_mask_ce, loss_mask_dice))
    return loss


def get_uncertain_point_coords_with_randomness(
        coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.
    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    n_dim = 3
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)  # 12544 * 3, oversampled
    point_coords = torch.rand(num_boxes, num_sampled, n_dim,
                              device=coarse_logits.device)  # (K, 37632, 3); uniform dist [0, 1)
    point_logits = point_sample_3d(coarse_logits, point_coords, align_corners=False)  # (K, 1, 37632)
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)  # 9408

    num_random_points = num_points - num_uncertain_points  # 3136
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]

    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]  # [K, 9408]

    point_coords = point_coords.view(-1, n_dim)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, n_dim
    )  # [K, 9408, 3]

    if num_random_points > 0:
        # from detectron2.layers import cat
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, n_dim, device=coarse_logits.device),
            ],
            dim=1,
        )  # [K, 12544, 3]

    return point_coords


def point_sample_3d(input, point_coords, **kwargs):
    """
    from detectron2.projects.point_rend.point_features
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, D, H, W) that contains features map on a D x H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 3) or (N, Dgrid, Hgrid, Wgrid, 3) that contains
        [0, 1] x [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Dgrid, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2).unsqueeze(2)  # why

    # point_coords should be (N, D, H, W, 3)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)

    if add_dim:
        output = output.squeeze(3).squeeze(3)

    return output

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))
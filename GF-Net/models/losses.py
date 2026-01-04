import torch
import torch.nn as nn
import torch.nn.functional as F # Make sure F is imported
import logging # Import logging
logger = logging.getLogger(__name__) # Get logger for warnings

# Assuming these imports exist 
# If not, these classes/functions might need to be defined or imported differently
try:
    from . import base
    from . import functional as F_segmodels # Rename to avoid conflict with nn.functional
    from . import _modules as modules
    SEGMODELS_AVAILABLE = True
except ImportError:
    # Define dummy base.Loss if segmentation_models_pytorch is not used/available
    # This allows the existing code structure to remain without erroring
    # but the original losses (JaccardLoss, DiceLoss) won't work without the library.
    class BaseLoss(nn.Module):
        def __init__(self, name=None, **kwargs):
            super().__init__()
            self.__name__ = name if name is not None else type(self).__name__
    base = type('BaseModule', (object,), {'Loss': BaseLoss}) # Mock base module
    modules = type('ModulesModule', (object,), {'Activation': nn.Identity}) # Mock modules
    SEGMODELS_AVAILABLE = False
    print("Warning: segmentation_models_pytorch components (base, functional, _modules) not found. Original JaccardLoss/DiceLoss may not work.")


# =============================================================================
# Original Losses 
# =============================================================================

if SEGMODELS_AVAILABLE:
    class JaccardLoss(base.Loss):

        def __init__(self, eps=1e-7, activation=None, ignore_channels=None,
                     per_image=False, class_weights=None, **kwargs):
            super().__init__(**kwargs)
            self.eps = eps
            self.activation = modules.Activation(activation, dim=1)
            self.per_image = per_image
            self.ignore_channels = ignore_channels
            self.class_weights = class_weights

        def forward(self, y_pr, y_gt):
            y_pr = self.activation(y_pr)
            return 1 - F_segmodels.jaccard( # Use renamed F_segmodels
                y_pr, y_gt,
                eps=self.eps,
                threshold=None,
                ignore_channels=self.ignore_channels,
                per_image=self.per_image,
                class_weights=self.class_weights,
            )


    class DiceLoss(base.Loss):
        """Original Dice Loss (likely from segmentation-models-pytorch)."""
        def __init__(self, eps=1e-7, beta=1., activation=None, ignore_channels=None,
                     per_image=False, class_weights=None, drop_empty=False,
                     aux_loss_weight=0, aux_loss_thres=50, **kwargs):
            super().__init__(**kwargs)
            self.eps = eps
            self.beta = beta
            self.activation = modules.Activation(activation, dim=1) # Expects logits, applies activation
            self.ignore_channels = ignore_channels
            self.per_image = per_image
            self.class_weights = class_weights
            self.drop_empty = drop_empty
            self.aux_loss_weight = aux_loss_weight
            self.aux_loss_thres = aux_loss_thres

        def forward(self, y_pr, y_gt):
            # y_pr: logits (N, C, H, W)
            # y_gt: target indices (N, H, W) or one-hot (N, C, H, W)
            y_pr = self.activation(y_pr) # Apply activation (e.g., softmax) to get probabilities

            # Original aux loss calculation (seems specific)
            if self.aux_loss_weight > 0:
                # This part seems designed for binary or specific multi-class scenarios
                # It calculates a class presence loss based on thresholds
                gt_presence = torch.sum(y_gt, axis=[2, 3], keepdim=True) # Assumes y_gt might be one-hot?
                gt_presence = (gt_presence > self.aux_loss_thres).type(gt_presence.dtype)

                if y_pr.shape[1] > 1:
                    pr_argmax = torch.argmax(y_pr, axis=1, keepdim=True)
                else:
                    pr_argmax = (y_pr > 0.5)
                pr_argmax = pr_argmax.type(y_pr.dtype)
                pr_presence = torch.sum(pr_argmax, axis=[2, 3], keepdim=True)
                pr_presence = (pr_presence > self.aux_loss_thres).type(pr_presence.dtype)

                # This functional.binary_crossentropy might be specific to segmodels library
                class_loss = F_segmodels.binary_crossentropy(pr_presence, gt_presence)
                class_loss = class_loss.mean()

            # Calculate Dice loss using F_segmodels.f_score
            dice_loss = 1 - F_segmodels.f_score(
                y_pr, y_gt, # f_score likely handles one-hot conversion internally
                beta=self.beta,
                eps=self.eps,
                threshold=None, # Expects probabilities
                ignore_channels=self.ignore_channels,
                per_image=self.per_image,
                class_weights=self.class_weights,
                drop_empty=self.drop_empty,
            )

            if self.aux_loss_weight > 0:
                return dice_loss * (1 - self.aux_loss_weight) + class_loss * self.aux_loss_weight

            return dice_loss


    class BCELoss(base.Loss):
        """Original BCE Loss wrapper (likely from segmentation-models-pytorch)."""
        def __init__(self, pos_weight=1., neg_weight=1., reduction='mean', label_smoothing=None, scale=1):
            super().__init__()
            assert reduction in ['mean', None, False]
            self.pos_weight = pos_weight
            self.neg_weight = neg_weight
            self.reduction = reduction
            self.label_smoothing = label_smoothing
            self.scale = scale

        def forward(self, pr, gt):
            if len(gt.shape) < len(pr.shape):
                gt = gt.unsqueeze(axis=-1)
            # Uses functional.binary_crossentropy from segmodels
            loss = F_segmodels.binary_crossentropy(
                pr, gt,
                pos_weight=self.pos_weight,
                neg_weight=self.neg_weight,
                label_smoothing=self.label_smoothing,
            )

            if self.reduction == 'mean':
                loss = loss.mean()

            return loss * self.scale


    class BinaryFocalLoss(base.Loss):
        """Original Binary Focal Loss (likely from segmentation-models-pytorch)."""
        def __init__(self, alpha=1, gamma=2, class_weights=None, logits=False, reduction='mean', label_smoothing=None):
            super().__init__()
            assert reduction in ['mean', None]
            self.alpha = alpha
            self.gamma = gamma
            self.logits = logits
            self.reduction = reduction
            self.class_weights = class_weights if class_weights is not None else 1.
            self.label_smoothing = label_smoothing

        def forward(self, pr, gt):
            if self.logits:
                bce_loss = nn.functional.binary_cross_entropy_with_logits(pr, gt, reduction='none')
            else:
                # Uses functional.binary_crossentropy from segmodels
                bce_loss = F_segmodels.binary_crossentropy(pr, gt, label_smoothing=self.label_smoothing)

            pt = torch.exp(- bce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
            # Apply class weights if provided (assuming binary weights)
            focal_loss = focal_loss * torch.tensor(self.class_weights).to(focal_loss.device)

            if self.reduction == 'mean':
                focal_loss = focal_loss.mean()

            return focal_loss

# --- Standard PyTorch Loss Wrappers (from original file) ---

class L1Loss(nn.L1Loss, base.Loss):
    pass

class MSELoss(nn.MSELoss, base.Loss):
    pass

class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    """Wraps nn.CrossEntropyLoss. Instantiate directly in train.py for weights."""
    pass

class NLLLoss(nn.NLLLoss, base.Loss):
    pass

class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass


# =============================================================================
# Multi-Class Loss Implementations 
# =============================================================================

class MultiClassFocalLoss(nn.Module):
    """
    Focal Loss for Multi-Class Segmentation.

    Args:
        alpha (float or list/tensor, optional): Weighting factor for classes.
            Can be a single float (applied to positive class in binary manner)
            or a list/tensor of weights for each class. Defaults to None.
        gamma (float): Focusing parameter. Defaults to 2.0.
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Defaults to 'mean'.
        ignore_index (int): Specifies a target value that is ignored
            and does not contribute to the input gradient. Defaults to 255.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=255):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.alpha = alpha

        # Handle alpha: If it's a list or tensor, ensure it's a tensor
        if isinstance(alpha, (list, tuple, torch.Tensor)):
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        # If alpha is a single float, it's often used differently in multi-class
        # (e.g., balancing positive/negative in a one-vs-all sense, less common)
        # For multi-class, alpha is typically per-class weights.
        # If alpha is float, we might ignore it or raise error, let's ignore for now.
        elif isinstance(alpha, (float, int)):
             print(f"Warning: Alpha as float ({alpha}) in MultiClassFocalLoss is ambiguous. Provide per-class weights or None.")
             self.alpha = None


    def forward(self, inputs, targets):
        """
        Forward pass.

        Args:
            inputs (torch.Tensor): Predicted logits, shape (N, C, H, W).
            targets (torch.Tensor): Ground truth labels, shape (N, H, W).

        Returns:
            torch.Tensor: Calculated focal loss.
        """
        # Ensure target is long type
        targets = targets.long()

        # Calculate Cross Entropy loss without reduction
        # LogSoftmax + NLLLoss is numerically more stable than Softmax + CrossEntropy
        log_softmax_inputs = F.log_softmax(inputs, dim=1)
        ce_loss = F.nll_loss(log_softmax_inputs, targets,
                             ignore_index=self.ignore_index, reduction='none')

        # Get probabilities of the correct class (pt = exp(-ce_loss))
        pt = torch.exp(-ce_loss)

        # Calculate Focal Loss: (1 - pt)^gamma * ce_loss
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        # Apply alpha weighting (per-class weights)
        if self.alpha is not None:
            if not isinstance(self.alpha, torch.Tensor):
                 raise TypeError("Alpha must be a Tensor for multi-class weighting.")
            if self.alpha.device != loss.device:
                self.alpha = self.alpha.to(loss.device)

            # Create a mask for valid (non-ignored) pixels
            valid_mask = (targets != self.ignore_index)
            # Get targets only for valid pixels
            targets_valid = targets[valid_mask]

            # Get alpha weights corresponding to the valid target classes
            alpha_t = self.alpha[targets_valid]

            # Apply weights only to the loss values of valid pixels
            # Need to apply alpha_t to the corresponding loss[valid_mask] elements
            loss[valid_mask] = alpha_t * loss[valid_mask]


        # Apply reduction
        if self.reduction == 'mean':
            # Calculate mean loss only over non-ignored pixels
            if self.ignore_index is not None:
                valid_pixels = (targets != self.ignore_index).sum()
                if valid_pixels > 0:
                    loss = loss.sum() / valid_pixels
                else:
                    # Handle case where all pixels might be ignored
                    loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)
            else:
                 loss = loss.mean() # Mean over all pixels if no ignore_index
        elif self.reduction == 'sum':
            loss = loss.sum()
        # If reduction is 'none', loss is returned as is (per-pixel)

        return loss


class MultiClassDiceLoss(nn.Module):
    """
    Dice Loss for Multi-Class Segmentation. Assumes model outputs logits.

    Args:
        smooth (float): A smoothing factor to avoid division by zero. Defaults to 1.0.
        ignore_index (int): Specifies a target value that is ignored. Defaults to 255.
        apply_softmax (bool): Whether to apply softmax to the input logits. Defaults to True.
        weight (torch.Tensor, optional): A manual rescaling weight given to each class.
                                         Shape (C,). Defaults to None.
        average (str): Method to average loss over classes. 'micro' averages over all pixels,
                       'macro' averages per-class scores. Defaults to 'macro'.
        average_foreground (bool): If True and average='macro', only averages over
                                   foreground classes (1 to C-1). Defaults to True.
    """
    # Ensure the __init__ signature matches the call in train.py
    def __init__(self, smooth=1.0, ignore_index=255, apply_softmax=True, weight=None, average='macro', average_foreground=True):
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.apply_softmax = apply_softmax
        # Store weight as tensor if provided
        if weight is not None and not isinstance(weight, torch.Tensor):
             self.weight = torch.tensor(weight, dtype=torch.float)
        else:
             self.weight = weight # Can be None or already a Tensor
        if average not in ['micro', 'macro']:
            raise ValueError("average must be 'micro' or 'macro'")
        self.average = average
        self.average_foreground = average_foreground # Store the argument

    def forward(self, inputs, targets):
        # inputs: (N, C, H, W) - logits
        # targets: (N, H, W) - class indices
        num_classes = inputs.shape[1]
        batch_size = inputs.shape[0]

        # Ensure target is long type
        targets = targets.long()

        # Apply softmax if needed
        if self.apply_softmax:
            probs = F.softmax(inputs, dim=1)
        else:
            probs = inputs # Assume inputs are already probabilities

        # Create a mask for valid (non-ignored) pixels [N, H, W]
        valid_mask = (targets != self.ignore_index)

        # Create one-hot encoding for the target [N, H, W, C] -> [N, C, H, W]
        # Mask the targets *before* one-hot encoding to avoid encoding the ignore_index
        targets_masked_for_onehot = targets * valid_mask.long()
        targets_one_hot = F.one_hot(targets_masked_for_onehot, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).contiguous()

        # Expand valid_mask to match probs shape [N, 1, H, W] for broadcasting
        valid_mask_expanded = valid_mask.unsqueeze(1)

        # Mask probabilities based on valid pixels
        probs_masked = probs * valid_mask_expanded
        # targets_one_hot is already masked implicitly by using targets_masked_for_onehot

        if self.average == 'micro':
            # Flatten spatial dimensions and sum over batch and classes
            intersection = torch.sum(probs_masked * targets_one_hot)
            cardinality = torch.sum(probs_masked + targets_one_hot)
            dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        elif self.average == 'macro':
            # Calculate intersection and union per class, summing over spatial dims and batch
            intersection = torch.sum(probs_masked * targets_one_hot, dim=(0, 2, 3)) # Sum over N, H, W -> (C,)
            cardinality = torch.sum(probs_masked + targets_one_hot, dim=(0, 2, 3)) # Sum over N, H, W -> (C,)

            # Calculate Dice coefficient per class
            dice_score_per_class = (2. * intersection + self.smooth) / (cardinality + self.smooth) # Shape: (C,)

            # --- Apply weights and average_foreground logic ---
            if self.weight is not None:
                # Ensure weight tensor is on the correct device
                if self.weight.device != dice_score_per_class.device:
                    self.weight = self.weight.to(dice_score_per_class.device)
                # Ensure weights match number of classes
                if self.weight.shape[0] != num_classes:
                     raise ValueError(f"Class weights shape {self.weight.shape} does not match num_classes {num_classes}")

                # Select classes and weights based on average_foreground flag
                if self.average_foreground and num_classes > 1:
                    scores_to_avg = dice_score_per_class[1:] # Select foreground classes
                    w = self.weight[1:] # Select corresponding weights
                else:
                    scores_to_avg = dice_score_per_class # Use all classes
                    w = self.weight # Use all weights

                # Weighted average of selected per-class scores
                if w.sum() > 0: # Avoid division by zero if weights sum to zero
                    dice_score = (scores_to_avg * w).sum() / w.sum()
                else:
                     logger.warning("Sum of weights for Dice loss averaging is zero.")
                     dice_score = torch.tensor(0.0, device=inputs.device) # Or 1.0 if loss is 1-dice? Let's use 0.0 score -> loss 1.0

            else: # No weights provided
                 # Average Dice score across selected classes (simple mean)
                 if self.average_foreground and num_classes > 1:
                     scores_to_avg = dice_score_per_class[1:] # Mean over foreground classes only
                 else:
                     scores_to_avg = dice_score_per_class # Mean over all C classes

                 # Avoid NaN if scores_to_avg is empty (e.g., num_classes=1 and avg_foreground=True)
                 if scores_to_avg.numel() > 0:
                      dice_score = scores_to_avg.mean()
                 else:
                      logger.warning("No scores to average for Dice loss (num_classes=1 and average_foreground=True?).")
                      dice_score = torch.tensor(0.0, device=inputs.device) # Default score to 0 -> loss 1.0

        # Final Dice loss
        dice_loss = 1. - dice_score

        return dice_loss


# =============================================================================
# Original Combined Losses 
# =============================================================================

if SEGMODELS_AVAILABLE:
    class FocalDiceLoss(base.Loss):
        """Original Binary Focal + Dice Loss."""
        def __init__(self, lamdba=2):
            super().__init__()
            self.lamdba = lamdba
            self.focal = BinaryFocalLoss() # Uses the original BinaryFocalLoss
            self.dice = DiceLoss(eps=10.) # Uses the original DiceLoss

        def __call__(self, y_pred, y_true):
            # Assumes binary inputs/targets suitable for the original losses
            return self.lamdba * self.focal(y_pred, y_true) + self.dice(y_pred, y_true)


    class BCEDiceLoss(base.Loss):
        """Original Binary BCE + Dice Loss."""
        def __init__(self, lamdba=2):
            super().__init__()
            self.lamdba = lamdba
            self.bce = BCELoss() # Uses the original BCELoss
            self.dice = DiceLoss(eps=10.) # Uses the original DiceLoss

        def __call__(self, y_pred, y_true):
            # Original implementation had specific pre-processing for binary case
            if y_pred.shape[1] > 1:
                # This part seems specific to converting multi-class output to binary
                y_pred = torch.sigmoid(y_pred)
                y_pred = torch.unsqueeze(y_pred[:, 1, :, :], dim=1) # Takes class 1 probability?
            # Ensure y_true is float and handle ignore index (e.g., 255)
            y_true = torch.unsqueeze(y_true, dim=1).float()
            y_true[y_true == 255] = 0 # Assuming 255 is ignore, map to background 0? Check logic.

            return self.lamdba * self.bce(y_pred, y_true) + self.dice(y_pred, y_true)


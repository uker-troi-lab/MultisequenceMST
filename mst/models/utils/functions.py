import torch 
import torch.nn.functional as F 
from torchvision.utils import draw_segmentation_masks
from matplotlib.pyplot import get_cmap

def one_hot(tensor,  num_classes=-1):
    """Wrapper for pytorch one-hot encoding
    
    Arguments:
        tensor {torch.Tensor} -- Tensor to be encoded of shape [Batch, (Depth), Height, Width]
    
    Keyword Arguments:
        num_classes {int} -- number of classes (default: {-1})
    
    Returns:
        torch.Tensor -- Tensor of shape [Batch, Classes, (Depth), Height, Width]
    
    Warning:
        This function destroys the backprogation path ! 
    """
    return F.one_hot(tensor.long(),num_classes).permute(0,tensor.ndim,*list(range(1,tensor.ndim)))



def minmax_norm(x, max=1, dim=None, smooth_nr=0, smooth_dr=0):
    """Normalizes input to [0, max] for each batch and channel if dim is None.

    Args:
        x (torch.Tensor): Tensor to be normalized, Shape [Batch, Channel, *]

    Returns:
        torch.Tensor: Normalized tensor 
    """
    if dim is None:
        return torch.stack([ torch.stack([(ch-ch.min()+smooth_nr)/(ch.max()-ch.min()+smooth_dr)*max for ch in batch]) for batch in x])
    else:
        min_val = x.min(dim=dim, keepdim=True)[0]
        max_val = x.max(dim=dim, keepdim=True)[0]
        return  (x - min_val+smooth_nr) / (max_val - min_val + smooth_dr) 


def tensor2image(tensor, batch=0):
    """Transform tensor into shape of multiple 2D RGB/gray images. 
        Keep 2D images as they are (gray or RGB).  
        For 3D images, pick 'batch' and use depth and interleaved channels as batch (multiple gray images). 

    Args:
        tensor (torch.Tensor): Image of shape [B, C, H, W] or [B, C, D, H, W]

    Returns:
        torch.Tensor: Image of shape [B, C, H, W] or [DxC,1, H, W]  (Compatible with torchvision.utils.save_image)
    """
    return (tensor if tensor.ndim<5 else torch.swapaxes(tensor[batch], 0, 1).reshape(-1, *tensor.shape[-2:])[:,None])


def tensor_mask2image(tensor, mask_hot, batch=0, alpha=0.25, colors=None, exclude_chs=[], exclude_classes=[0]):
    """Transform a tensor and a one-hot mask into multiple 2D RGB images.

    Args:
        tensor (torch.Tensor): Image tensor. Can be 3D volume of shape [B, C, D, W, H] or 2D of shape [B, C, H, W]
        mask_hot (torch.Tensor): One-Hot encoded mask of shape [B, Classes, D, W, H] or [B, Classes, H, W] or Binary mask (Classes=1)
        batch (int, optional): Batch to use if input is 3D. Defaults to 0.
        alpha (float, optional): 1-Transparency. Defaults to 0.25.
        colors: List or string containing the colors
        exclude_chs: image channels that should not overlay by mask 
        exclude_classes : classes that should be excluded - typically background (0)


    Returns:
        torch.Tensor: Tensor of 2D-RGB images with transparent mask on each. For 3D will be [CxD, 3, H, W] for 2D will be [B, 3, H, W] 
    """
    
    mask_hot = one_hot(mask_hot[:, 0], 2) if mask_hot.shape[1] ==1 else mask_hot # Ensure one-hot 
    mask_hot = mask_hot.type(torch.bool).cpu() # To bool and cpu (see bug below)
    mask_hot = mask_hot if mask_hot.ndim<5 else torch.swapaxes(mask_hot[batch], 0, 1) # 3D [B, C, D, H, W] -> [D, C, H, W]. 2D [B, C, H, W] -> [B, C, H, W]
    mask_hot = torch.stack([ mask_hot[:, cls_idx] for cls_idx in range(mask_hot.shape[1]) if cls_idx not in exclude_classes ], dim=1) 
    
    image = minmax_norm(tensor, 255).type(torch.uint8).cpu() # To uint8 and cpu (see bug below)
    image = image[None] if image.ndim==4 else image[batch][:,:,None] # 3D [B, C, D, H, W] -> [C, D, 1, H, W]. 2D [B, C, H, W] -> [1, B, C, H, W] 
    image = torch.cat([image for _ in range(3)], dim=2) if image.shape[2]!=3 else image # Ensure RGB  [*, 3, H, W] 
    image = torch.stack([draw_segmentation_masks(i, m, alpha=alpha, colors=colors) if ch not in exclude_chs else i for ch, img_ch in enumerate(image)  for i,m in zip(img_ch, mask_hot) ]) # [B, 3, H, W]  # BUG Apparently only supports cpu()
    return image/255.0


def tensor_cam2image(tensor, cam, batch=0, alpha=0.5, color_map=get_cmap('jet')):
    """Transform a tensor and a (grad) cam into multiple 2D RGB images.

    Args:
        tensor (torch.Tensor): Image tensor in range [0, 1]. Can be 3D volume of shape [B, C, D, W, H] or 2D of shape [B, C, H, W]
        cam (torch.Tensor): (Grad-) cam in range [0, 1] of shape [B, 1, D, W, H] or [B, 1, H, W]
        batch (int, optional): Batch to use if input is 3D. Defaults to 0.
        alpha (float, optional): 1-Transparency. Defaults to 0.25.

    Returns:
        torch.Tensor: Tensor of 2D-RGB images with transparent mask on each. For 3D will be [CxD, 3, H, W] for 2D will be [B, 3, H, W] 
    """
    img = tensor2image(tensor, batch) #  -> [B, C, H, W]
    img = torch.cat([img for _ in range(3)], dim=1) if img.shape[1]!=3 else img # Ensure RGB  [B, 3, H, W] 
    cam_img = tensor2image(cam, batch) #  -> [B, 1, H, W]
    cam_img = cam_img[:,0].cpu().numpy() # -> [B, H, W]
    cam_img = torch.tensor(color_map(cam_img)) # -> [B, H, W, 4], color_map expects input to be [0.0, 1.0]
    cam_img = torch.moveaxis(cam_img, -1, 1)[:, :3] # -> [B, 3, H, W]

    overlay = (1-alpha)*img + alpha*cam_img

    return overlay
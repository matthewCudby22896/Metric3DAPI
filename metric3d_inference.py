
import torch
import numpy as np
import cv2

MODEL_VERSIONS = {
    'small' : 'metric3d_vit_small',
    'large' : 'metric3d_vit_large',
    'giant' : 'metric3d_vit_giant2',
}

TORCH_HUB_USER = 'yvanyin/metric3d'

class metric3d_inference_generator:
    def __init__(self):
        self.model_small = None
        self.model_large = None
        self.model_giant = None
        self.models = {}
        self.input_size = (616, 1064)
        
        if not torch.cuda.is_available(): 
            raise Exception("CUDA is not available")

    def _get_model(self, version : str):
        if version not in MODEL_VERSIONS:
            raise ValueError(f"Unkown version: {version}")
        
        if version not in self.models:
            self.models[version] = torch.hub.load(TORCH_HUB_USER, MODEL_VERSIONS[version], pretrain=True)
        
        return self.models[version]
    
    def estimate_depth(self, org_rgb: np.ndarray, version : str) -> np.ndarray:
        model = self._get_model(version)
        
        # rescale the image
        h, w = org_rgb.shape[:2]
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        rgb = cv2.resize(org_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        
        # add padding
        h, w = rgb.shape[:2]
        pad_h, pad_w = self.input_size[0] - h, self.input_size[1] - w
        pad_h_half, pad_w_half = pad_h // 2, pad_w // 2
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=PADDING_CLR)
        
        # normalise 
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - mean), std)
        rgb = rgb[None, :, :, :].cuda()
        
        # forward propogatrion
        with torch.no_grad():
            pred_depth, _, _ = model({'input' : rgb})
        
        # remove padding
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
        
        # upsample to original size
        pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], org_rgb.shape[:2], mode='bilinear').squeeze()
        pred_depth = pred_depth.cpu().numpy()
        
        return pred_depth
import os
import sys
import torch
from PIL import Image

external_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'external', 'ZoeDepth'))

if external_path not in sys.path:
    sys.path.append(external_path)

# from zoedepth.models.builder import build_model
# from zoedepth.utils.config import get_config

class ZoeDepth:
    def __init__(self):
        #self.conf = get_config("zoedepth", "infer")
        #self.depth_model = build_model(self.conf)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_model = torch.hub.load(repo_or_dir='isl-org/ZoeDepth',
                                          model="ZoeD_N",
                                          pretrained=True).to(self.device).eval()

    def infer_depth(self, img):
        depth = self.depth_model.infer_pil(img)
        raw_depth = Image.fromarray((depth*256).astype('uint16'))
        return raw_depth

    def infer_direct_depth(self, img):
        depth = self.depth_model.infer_pil(img)

        return depth

import cv2
import torch
import torch.nn.functional as F
from easydict import EasyDict

from echols.clfnet import MobilenetV2
from echols.log import logger


class ChamberClassifier:
    def __init__(self, ckpt=None, input_size=256, batch_size=1, device='cpu'):
        self.config = EasyDict(locals())
        self.model = self._load_model()
        self.image_size = input_size

        self.idx2chamber = {0: "2C", 1: "3C", 2: "4C", 3: "none"}

    def _load_model(self):
        model = MobilenetV2().cpu()
        if self.config.ckpt:
            logger.debug(f'Load clf model from {self.config.ckpt}')
            ckpt_data = torch.load(self.config.ckpt, map_location='cpu')
            model.load_state_dict(ckpt_data["model"])
        model.to(self.config.device).eval()
        return model

    def predict_chamber(self, dataset):
        frames = [item['image'] for item in dataset]
        ret = self.run(frames)
        return ret

    def run(self, frames):
        frames = [self._preprocess(frame) for frame in frames]
        frame_chunks = self._split_frames_into_chunks(frames)
        with torch.no_grad():
            pred = torch.cat(
                [self.model(chunk.to(self.config.device)) for chunk in frame_chunks],
                dim=0
            )
            pred = torch.mean(F.softmax(pred, dim=1), dim=0)
            pred = torch.argmax(pred).item()
        chamber = self.idx2chamber[pred]
        return chamber

    def _preprocess(self, frame):
        frame = cv2.resize(frame, (self.image_size, self.image_size))
        frame = frame.transpose((2, 0, 1)) / 255.0
        frame = frame[None, ...]
        frame = torch.from_numpy(frame).float()
        return frame

    def _split_frames_into_chunks(self, frames):
        frame_chunks = [
            torch.cat(frames[i:i+self.config.batch_size], dim=0) \
                for i in range(0, len(frames), self.config.batch_size)
        ]
        return frame_chunks


if __name__ == '__main__':
    net = ChamberClassifier('ckpts/mobilenetv2_0049_0.9507_best.pth')
    img = cv2.imread('assets/dog.png')

    ret = net.run([img])
    logger.debug(f'{ret} classifier is oke') # should be None cause dog is not a chamber

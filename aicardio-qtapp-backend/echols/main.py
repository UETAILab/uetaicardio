import os
import time
import json
import yaml
import argparse
import numpy as np

from easydict import EasyDict

from echols.log import logger
from echols.datasets import DICOMDataset
from echols.classifier import ChamberClassifier
from echols.detector import YOLOv3LVDetector
from echols.segmentor import DICOMSegmentor
from echols.segmentor import DICOMCropSegmentor
from echols.ef_calculator import EFCalculator
from echols.ef_calculator_pro import EFCalculatorPro


def parse_args(dat=None):
    parser = argparse.ArgumentParser(description='Echols')
    parser.add_argument("--dicom_path", type=str, help="DICOM path")
    parser.add_argument("--output_dir", type=str, default='results', help="Store outputs")
    parser.add_argument("--config_path", type=str, default='config/cuda_configs.yml', help="Yaml config path")
    if dat is not None:
        args = vars(parser.parse_args(dat))
    else:
        args = vars(parser.parse_args())
        assert args['dicom_path'] is not None, "You must provide a dicom path"

    with open(args['config_path'], 'r') as f:
        config = dict(yaml.safe_load(f))

    return EasyDict({**config, **args})

class Worker:
    def __init__(self, args=''):
        args = parse_args(args)

        self.classifier = ChamberClassifier(**args.classifier)

        if args.runtime.use_crop_segmentor:
            self.detector = YOLOv3LVDetector(**args.detector)
            self.segmentor = DICOMCropSegmentor(self.detector, **args.crop_segmentor)
        else:
            self.segmentor = DICOMSegmentor(**args.segmentor)

        self.calculator = EFCalculator()
        self.calculatorp = EFCalculatorPro()

        self.args = args
        logger.debug('Init Full pipeline worker with args:')
        logger.debug(args)

    def run(self, file_path):
        tik = time.time()
        dataset = DICOMDataset(file_path, case_idx=0, **self.args.dataset)
        #chamber = self.classifier.predict_chamber(dataset)
        masks = self.segmentor.predict_masks(dataset)
        #results = self.calculator.compute_ef(
            #masks, dataset.metadata, dataset, self.args.output_dir
        #)
        results = self.calculatorp.compute_ef(
            masks, dataset.metadata, dataset, self.args.output_dir)
        #results['chamber'] = chamber
        logger.info(f'Total running time: {time.time()-tik}')
        return results

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':
    tik = time.time()
    args = parse_args()
    logger.debug(args)
    worker = Worker('')
    results = worker.run(args.dicom_path)

    #print(json.dumps(results, indent=2))
    ef, gls1, gls2 = results["ef"], results["GLS"], results["SLS"]
    logger.debug(results.keys())
    logger.debug(' '.join([str(x) for x in [ef, gls1, gls2]]))

    fname = args.dicom_path.split('/')[-1]
    with open(f'results/{fname}.json', 'w') as f:
        print(json.dumps(results, indent=2, cls=NumpyEncoder), file=f)

    logger.debug(f'Total running time: {time.time()-tik}')

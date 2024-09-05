#!/usr/bin/python3.7
# -*- coding: utf-8 -*-


from common.params import args
import warnings
warnings.filterwarnings('ignore')
from common.ocr_utils import string_to_arrimg
from common.exceptions import ParsingError
import time, mmcv
from loguru import logger as log
from table_structure_recognition.mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot


class Inference():
    def __init__(self, is_slide=False):
        device = 'cuda:' + args.device if args.use_gpu else args.device
        # build the model from a config file and a checkpoint file
        self.model = init_segmentor(is_slide, args.config, args.checkpoint, device=device)
        log.info('loading model...')
        if args.use_gpu:
            log.info("use gpu: %s" % args.use_gpu)
            log.info("CUDA_VISIBLE_DEVICES: %s" % str(args.device))
        else:
            log.info("use gpu: %s" % args.use_gpu)

    def __call__(self, img_str, is_resize=True):
        t0 = time.time()
        if isinstance(img_str,str):
            img = string_to_arrimg(img_str, log_flag = True)
            if img is None:
                raise ParsingError('Failed to transform base64 to image.', 4)
        else:
            img = img_str

        # test a single image
        result = inference_segmentor(self.model, img, is_resize)
        # show the results

        if args.is_visualize:
            show_result_pyplot(self.model, img, result, opacity=args.opacity,
                               out_file = args.out_file)

        result = result[0]
        # import matplotlib.pyplot as plt
        # plt.imshow(result)
        # plt.show()
        log.info(f"elapse: {(time.time()-t0):0.2f}s")
        return result


if __name__ == "__main__":
    from common.ocr_utils import imagefile_to_string
    filename = r'test/3.jpg'
    img_str = imagefile_to_string(filename)
    inference = Inference(is_slide = False)
    result = inference(img_str, is_resize = True)
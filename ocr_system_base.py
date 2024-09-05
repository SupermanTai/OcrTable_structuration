# -*- coding: utf-8 -*-


import os, cv2
from common.params import args

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
cpu = args.use_gpu == False
if cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

#一旦不再使用即释放内存垃圾，=1.0 垃圾占用内存大小达到10G时，释放内存垃圾
os.environ["FLAGS_eager_delete_tensor_gb"]="0.0"
#启用快速垃圾回收策略，不等待cuda kernel 结束，直接释放显存
os.environ["FLAGS_fast_eager_deletion_mode"]="1"
#该环境变量设置只占用0%的显存
os.environ["FLAGS_fraction_of_gpu_memory_to_use"]="0"

import ast
import numpy as np
from PIL import Image
from loguru import logger as log
from ppocr.infer.predict_system import TextSystem
from ppocr.infer.utility import draw_ocr_box_txt
import copy, paddle
import argparse
from common.ocr_utils import  fourxy2twoxy


def load_model(args, e2e_algorithm=False):
    log.info("Loading model...")
    if args.use_gpu:
        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
            log.info("use gpu: %s"%args.use_gpu)
            log.info("CUDA_VISIBLE_DEVICES: %s"%_places)
            args.gpu_mem = 500
        except:
            raise RuntimeError(
                "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
            )
    else:
        log.info("use gpu: %s"%args.use_gpu)

    text_sys = TextSystem(args)
    # log.info(args.__dict__)
    if args.warmup:
        img_warm = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(10):
            _ = text_sys(img_warm)
    return e2e_algorithm, text_sys




def parse_args():
    parser = argparse.ArgumentParser(description='OCR')
    parser.add_argument(
        '--image-file',
        type=str,
        required = True,
        help='输入图片所在路径')
    return parser.parse_args()


class OCR():
    def __init__(self, text_sys, img, cls = False):
        self.img = img
        # print(self.img.shape)
        self.dt_boxes, self.rec_res = text_sys(self.img, cls)

    def __call__(self, output_path=None):
        if output_path:
            os.makedirs(output_path, exist_ok = True)
        img = self.img
        img_origin = copy.deepcopy(img)

        dt_boxes, rec_res = self.dt_boxes, self.rec_res
        dt_num = len(dt_boxes)
        result = []
        for dno in range(dt_num):
            text, score = rec_res[dno]
            quadrangle = dt_boxes[dno]
            temp_result = {"text": text,
                           "quadrangle": quadrangle.tolist(),
                           "box": quadrangle.reshape(1, -1).squeeze().tolist(),
                           "bbox": fourxy2twoxy(quadrangle),
                           "score": float(score)}
            result.append(temp_result)

        if args.is_visualize:
            image = Image.fromarray(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB))

            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]

            draw_img = draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=args.drop_score,
                font_path=args.vis_font_path)
            if output_path is not None:
                output_file = os.path.join(output_path,  os.path.split(self.image_file)[-1])
            else:
                output_file = r'test/draw_img.jpg'
            cv2.imencode('.jpg', draw_img[:, :, ::-1])[1].tofile(output_file)
            log.info(f"The visualized image saved in {output_file}")
        paddle.device.cuda.empty_cache()
        return result



def main(image_file):
    # args = parse_args()
    # image_file = args.image_file
    ocr = OCR(text_sys, image_file, cls = True)
    result = ocr()
    return result


if __name__ == "__main__":
    e2e_algorithm, text_sys = load_model(args, e2e_algorithm = False)


    image_file = r'test/1/1.png'
    img = cv2.imdecode(np.fromfile(image_file, dtype = np.uint8), cv2.IMREAD_COLOR)
    result = main(img)
    print(result)
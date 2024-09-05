
from common.params import args
from ocr_system_base import OCR, load_model

e2e_algorithm, text_sys = load_model(args, e2e_algorithm = False)
# _, text_sys_common = load_model(args, e2e_algorithm = False)
# log.info(args.__dict__)

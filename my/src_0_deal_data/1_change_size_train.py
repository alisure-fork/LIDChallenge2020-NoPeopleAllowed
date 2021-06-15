import os
from tqdm import tqdm
from glob import glob
from PIL import Image
import multiprocessing
from alisuretool.Tools import Tools


cam_path="/media/ubuntu/4T/ALISURE/USS/WSS_Model_SS_0602_EVAL/3_DeepLabV3PlusResNet152_201_10_18_1_352_8_balance/ss_7_train_500/train_final"
voc12_root="/media/ubuntu/4T/ALISURE/Data/L2ID/data"


def main(i, images):
    Tools.print("{}".format(i))
    for one in images:
        im = Image.open(one.replace(cam_path, os.path.join(voc12_root, "ILSVRC2017_DET/ILSVRC/Data/DET")).replace(".png", ".JPEG"))
        Image.open(one).resize(im.size, resample=Image.NEAREST).save(Tools.new_dir(one.replace("train_final", "train_final_resize")))
        pass
    pass


all_image = glob(os.path.join(cam_path, "**/*.png"), recursive=True)
split_id = 1000

pools = multiprocessing.Pool(processes=multiprocessing.cpu_count())
for i in range(split_id + 1):
    now_image = all_image[len(all_image) // split_id * i: len(all_image) // split_id * (i + 1)]
    pools.apply_async(main, args=(i, now_image))
    pass


Tools.print("begin")
pools.close()
pools.join()
Tools.print("over")


# Check
for one in tqdm(all_image):
    if not os.path.exists(one.replace("train_final", "train_final_resize")):
        im = Image.open(one.replace(cam_path, os.path.join(
            voc12_root, "ILSVRC2017_DET/ILSVRC/Data/DET")).replace(".png", ".JPEG"))
        Image.open(one).resize(im.size, resample=Image.NEAREST).save(Tools.new_dir(one.replace("train_final", "train_final_resize")))
        pass
    pass

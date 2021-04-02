import os
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from alisuretool.Tools import Tools


if __name__ == '__main__':
    data_root = "/media/ubuntu/4T/ALISURE/Data/L2ID/data"
    annotations_raw_path = os.path.join(data_root, "LID_track1_annotations", "track1_val_annotations_raw")
    annotations_color_path = os.path.join(data_root, "LID_track1_annotations", "track1_val_annotations")

    palette = np.load('palette.npy').tolist()
    all_image = glob(os.path.join(annotations_raw_path, "*.png"))
    for image in all_image:
        data_raw = np.asarray(Image.open(image))
        im_raw = Image.fromarray(data_raw, "P")
        im_raw.putpalette(palette)
        # im_raw.save("1.png")

        im_color = Image.open(image.replace("track1_val_annotations_raw", "track1_val_annotations"))
        # im_color.save("2.png")
        pass
    pass

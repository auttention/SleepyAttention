import glob
import numpy as np
from PIL import Image


def validate_spectrograms():
    addrs = glob.glob("spectrograms/train/*.png") + glob.glob("spectrograms/devel/*.png") + glob.glob("spectrograms/test/*.png")
    for addr in addrs:
        img = np.array(Image.open(addr))
        if img.shape != (120, 160):
            #img = Image.open(addr).convert('L')
            #img.save(addr)
            print(addr)


if __name__ == "__main__":
    validate_spectrograms()
import numpy as np
from PIL import Image

im_parse = Image.open('/mnt/aix23904/아르포아/virtual_tryon/Fit_data/test/image-parse-v3/01_sou_000008_000038_wear_01outer_01coat_woman.png')
im_parse2 = Image.open('/mnt/aix23904/아르포아/virtual_tryon/VITON-HD/test/image-parse-v3/00006_00.png')
parse_array = np.array(im_parse)
parse_array2 = np.array(im_parse2)
print(np.unique(parse_array))
print(np.unique(parse_array2))

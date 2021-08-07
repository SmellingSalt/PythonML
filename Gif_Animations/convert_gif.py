# import glob
# filenames=[]
# for image in glob.glob('Gif_Animations/OUTPUT_DIR/Boundary/*.png'):
#     filenames.append(image)
#     print(image)

mypath='Gif_Animations/OUTPUT_DIR/Loss/'
from os import listdir
from os.path import isfile, join
filenames = [mypath+f for f in sorted(listdir(mypath)) if isfile(join(mypath, f))]
# filenames.sort

for thing in filenames:
    print(thing)

import imageio
with imageio.get_writer('movie.gif', mode='I',duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
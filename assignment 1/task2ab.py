import matplotlib.pyplot as plt
import pathlib
import numpy as np
import PIL
from utils import read_im, save_im, normalize

output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


im = read_im(pathlib.Path("images", "duck.jpeg"))
plt.imshow(im)


#print(im.shape)
def greyscale(im):
    
   
    """ Converts an RGB image to greyscale

    Args:
        im ([type]): [np.array of shape [H, W, 3]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]
    im = 0.212*R+0.7152*G+0.0722*B

    return im


im_greyscale = greyscale(im)
save_im(output_dir.joinpath("duck_greyscale.jpeg"), im_greyscale, cmap="gray")
plt.imshow(im_greyscale, cmap="gray")

#print(im_greyscale)
def inverse(im):
    """ Finds the inverse of the greyscale image

    Args:
        im ([type]): [np.array of shape [H, W]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    # YOUR CODE HERE
    # width,height, colors = im.shape
    # for w in range(width):
    #     for h in range(height):
    px = im_greyscale[:]

            #px=im_greyscale.getpixel((w,h))
            #print(px)
    im = 255-px
    im = normalize(im)
    return im

im_invert = inverse(im)
save_im(output_dir.joinpath("duck_invert.jpeg"), im_invert, cmap="gray")
plt.imshow(im_invert, cmap="gray")

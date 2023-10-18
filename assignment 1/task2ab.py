import matplotlib.pyplot as plt
import pathlib
import PIL
from utils import read_im, save_im
output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


im = read_im(pathlib.Path("images", "duck.jpeg"))
plt.imshow(im)


print(im.shape)
def greyscale(im):
    
    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]
    im = 0.212*R+0.7152*G+0.0722*B
    """ Converts an RGB image to greyscale

    Args:
        im ([type]): [np.array of shape [H, W, 3]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """

    return im


im_greyscale = greyscale(im)
save_im(output_dir.joinpath("duck_greyscale.jpeg"), im_greyscale, cmap="gray")
plt.imshow(im_greyscale, cmap="gray")


def inverse(im):
    """ Finds the inverse of the greyscale image

    Args:
        im ([type]): [np.array of shape [H, W]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    # YOUR CODE HERE
    width,height = im_greyscale.size
    for w in range(width):
        for h in range(height):
            px=im_greyscale.getpixel((w,h))
            px = 255-px
            
    return im

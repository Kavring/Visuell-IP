import matplotlib.pyplot as plt
import pathlib
import numpy as np
from utils import read_im, save_im, normalize
output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


im = read_im(pathlib.Path("images", "duck.jpeg"))
plt.imshow(im)


def convolve_im(im, kernel,
                ):
    """ A function that convolves im with kernel

    Args:
        im ([type]): [np.array of shape [H, W, 3]]
        kernel ([type]): [np.array of shape [K, K]]

    Returns:
        [type]: [np.array of shape [H, W, 3]. should be same as im]
    """
    assert len(im.shape) == 3
    assert (kernel.shape[0] % 2) == 1

    k_size = kernel.shape[0]
    mid = (k_size-1)/2
    w, h, col = im.shape
    print(im.shape)
    kernel = np.flip(kernel)
    #entering the image
    
    R_sum = 0
    G_sum = 0
    B_sum = 0

    for i in range(h):
        for j in range(w):

            #entering the kernel 
            #sum all values her
            for k in range(k_size):
                for l in range(k_size):

                    wi = j - mid + l
                    hi = i - mid + k
                    try:
                        R = im[wi, hi, 0]
                        G = im[wi, hi, 1]
                        B = im[wi, hi, 2]
                    except IndexError:
                        R, G, B = 0, 0, 0
                    else:
                        R = im[wi, hi, 0]
                        G = im[wi, hi, 1]
                        B = im[wi, hi, 2]


                    #matte her:
                    #backwards indexing should work as flipping the kernel
                    
                    R_sum += R * kernel[l , k] # might have to switch k and l
                    G_sum += G * kernel[l , k]
                    B_sum += B * kernel[l , k]

            im[j-1, i-1, 0] = R_sum
            im[j-1, i-1, 1] = G_sum
            im[j-1, i-1, 2] = B_sum

            R_sum = 0
            G_sum = 0
            B_sum = 0
    print(im.shape)
    return im


if __name__ == "__main__":
    # Define the convolutional kernels
    h_b = 1 / 256 * np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ])
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # Convolve images
    im_smoothed = convolve_im(im.copy(), h_b)
    save_im(output_dir.joinpath("im_smoothed.jpg"), im_smoothed)
    im_sobel = convolve_im(im, sobel_x)
    save_im(output_dir.joinpath("im_sobel.jpg"), im_sobel)

    # DO NOT CHANGE. Checking that your function returns as expected
    assert isinstance(
        im_smoothed, np.ndarray),         f"Your convolve function has to return a np.array. " + f"Was: {type(im_smoothed)}"
    assert im_smoothed.shape == im.shape,         f"Expected smoothed im ({im_smoothed.shape}" + \
        f"to have same shape as im ({im.shape})"
    assert im_sobel.shape == im.shape,         f"Expected smoothed im ({im_sobel.shape}" + \
        f"to have same shape as im ({im.shape})"
    plt.subplot(1, 2, 1)
    plt.imshow(normalize(im_smoothed))

    plt.subplot(1, 2, 2)
    plt.imshow(normalize(im_sobel))
    plt.show()

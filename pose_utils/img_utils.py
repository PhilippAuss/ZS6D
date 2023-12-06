import cv2
import numpy as np


def make_quadratic_crop(image, bbox):
    # Define the bounding box
    x_left, y_top, width, height = bbox

    # Calculate the size of the square crop based on the longer side
    longer_side = max(width, height)
    crop_size = (longer_side, longer_side)

    # Calculate the center of the bounding box
    center_x = x_left + width / 2
    center_y = y_top + height / 2
    crop_size = min(longer_side, int(max(width/2, height/2) * 2))

    # Calculate the coordinates of the top-left corner of the square crop
    crop_x = int(center_x - crop_size / 2)
    crop_y = int(center_y - crop_size / 2)

    # Check if the crop goes beyond the image boundaries
    if crop_x < 0 or crop_y < 0 or crop_x + crop_size > image.shape[1] or crop_y + crop_size > image.shape[0]:

        # If the crop goes beyond the image boundaries, crop first and add a border using cv2.copyMakeBorder to make the crop quadratic
        crop = image[max(crop_y, 0):min(crop_y+crop_size, image.shape[0]), max(crop_x, 0):min(crop_x+crop_size, image.shape[1])]
        border_size = max(crop_size - crop.shape[1], crop_size - crop.shape[0])
        border_size = max(0, border_size)  # Make sure the border size is not negative
        
        
        if crop_x < 0 or crop_x + crop_size > image.shape[1]:
            left = border_size // 2
            right = border_size - left
            crop = cv2.copyMakeBorder(crop, 0, 0, left, right, cv2.BORDER_REPLICATE)
        elif crop_y < 0 or crop_y + crop_size > image.shape[0]:
            top = border_size // 2
            bottom = border_size - top
            crop = cv2.copyMakeBorder(crop, top, bottom, 0, 0, cv2.BORDER_REPLICATE)
        else:
            print("Something went wrong during rectifying crop")
            return None

    else:
        # If the crop is within the image boundaries, just crop the image
        crop = image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    
    return crop, crop_y, crop_x



def rle_to_mask(rle):
    """Converts a COCOs run-length encoding (RLE) to 3 channel mask with [0,255]

    :param rle: Mask in RLE format
    :return: a 2D binary numpy array where '1's represent the object
    """
    binary_array = np.zeros(np.prod(rle.get('size')), dtype=bool)
    counts = rle.get('counts')
    
    start = 0
    for i in range(len(counts)-1):
        start += counts[i] 
        end = start + counts[i+1] 
        binary_array[start:end] = (i + 1) % 2
    
    binary_mask = binary_array.reshape(*rle.get('size'), order='F')
    
    # First, convert True to 255 and False to 0.
    mask = binary_mask * 255

    # # Then, convert the mask to 3-channel.
    # mask_3c = np.dstack([mask]*3)

    return mask


def get_bounding_box_from_mask(mask):
    # Convert to binary mask (0 and 1) if it is not
    mask_binary = np.where(mask > 0, 1, 0)

    # Find min and max rows and columns with a value of 1
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # return top-left and bottom-right corners and width, height
    x_left = cmin
    y_upper = rmin
    w = cmax - cmin + 1
    h = rmax - rmin + 1
    
    return [x_left, y_upper, w, h]
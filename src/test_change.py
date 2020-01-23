import numpy as np
import cv2
from skimage import measure

def calculate_difference_in_masks(mask_1, mask_2, filename = "", tensor = True, plot = True):
    """
        Calculate the area difference in the two masks
    """
    if tensor:
        m1 = mask_1.numpy().astype(np.uint8)
        m2 = mask_2.numpy().astype(np.uint8)
    
    
    m1_area = m1.sum()
    m2_area = m2.sum()
    
    total_area = m1_area + m2_area
    
    if m2_area > m1_area:
        d = (m2_area - m1_area)/ total_area
        change = ("increase", "+")
    else:
        d = (m1_area - m2_area)/ total_area
        change = ("decrease", "-")
    
    print(f'change detected : {0}{1}%'.format(change[1], d))
    
    

    difference = np.logical_xor(m1, m2)
    
    if plot:
        rows, columns = 1, 3
        fig=plt.figure(figsize=(10, 10))
        fig.add_subplot(rows, columns, 1)
        plt.imshow(m1)
        fig.add_subplot(rows, columns, 2)
        plt.imshow(m2)
        fig.add_subplot(rows, columns, 3)
        plt.imshow(difference)
        

        plt.savefig(filename,bbox_inches='tight')

        
        plt.show()


if __name__ == '__main__':
    
    calculate_difference_in_masks(mask_2,mask_1, filename = "./output/change.png")
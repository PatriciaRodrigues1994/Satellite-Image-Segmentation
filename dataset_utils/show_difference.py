import cv2
import matplotlib.pyplot as plt
from skimage import measure
 

def get_area(mask):
	area = measure.regionprops(mask.astype(np.uint8))	
	area = [prop.area for prop in area][0]
	return area

def cal_diff(mask_1,mask_2,files,image_1,image_2,results_1,results_2, save_file = True):
	len_1 = mask_1.shape[2]
	len_2 = mask_2.shape[2]

	#Number of detections might be unequal
	#combine mask channels.
	m1 = np.zeros((mask_1.shape[:2]))
	for i in range(len_1):
		m1 = np.logical_or(m1,mask_1[:,:,i])

	m2 = np.zeros((mask_2.shape[:2]))
	for i in range(len_2):
		m2 = np.logical_or(m2,mask_2[:,:,i])

	
	#Calculate total area covered by mask_1
	mask_1_area = get_area(m1)
	mask_2_area = get_area(m2)

	m1 = m1.astype(np.uint8)	
	m2 = m2.astype(np.uint8)	

	print(m1.shape)
	print(m2.shape)

	diff = cv2.absdiff(m1,m2)
	diff_area = get_area(diff)

	print("M1 area :",mask_1_area)
	print("M2 area :",mask_2_area)
	print("Diff in area :",diff_area)

	max_area = max(mask_1_area,mask_2_area)

	d = diff_area/max_area
	if mask_1_area > mask_2_area:
		print(files[0],' greater area')
	else:
		print(files[1],' greater area')

	print('Change ',d*100,'%')

	if save_file:
		plt.imshow(mask_1)
		plt.axis('off')
		plt.savefig('change_det/mask_1.png',bbox_inches='tight')
		#plt.show()

		plt.imshow(mask_2)
		plt.axis('off')
		plt.savefig('change_det/mask_2.png',bbox_inches='tight')
		#plt.show()

		plt.imshow(diff)
		plt.axis('off')
		plt.savefig('change_det/change'+str(d*100)+'.png',bbox_inches='tight')
		#plt.show()

	return m1,m2,diff
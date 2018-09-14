import cv2
from pylab import array, plot, show, axis, arange, figure, uint8

'''
img_id = "25015_l"
image_name = "heatmaps-good/FORPAPER/e2043_26FREQ_CROPPED_256_8_"+img_id+".jpg"
img = cv2.imread(image_name)
maxIntensity = 255.0 # depends on dtype of image data
phi = 1
theta = 1
img_contrast = (maxIntensity/phi)*(img/(maxIntensity/theta))**2
img_contrast = array(img_contrast,dtype=uint8)
cv2.imwrite("heatmaps-good/FORPAPER/CONTRAST_"+img_id+".jpg", img_contrast)
'''
image_name = "dataMarsden26FREQ-CROPPED/00010.jpg"
img = cv2.imread(image_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("heatmaps-good/FORPAPER/00010.jpg", img)
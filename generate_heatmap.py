from inference import Inference
import cv2
import numpy as np

infer=Inference(model='hg_280_201')

img=cv2.imread("dataMarsden4-4-SPLIT/06097.jpg")

img = cv2.resize(img, (128, 128))

hms=infer.predictHM(img)

new_img=np.array(img)

for i in range(np.shape(hms)[3]):
    index=np.argmax(hms[0,:,:,i])
    x=index%32*4
    y=int(index/32)*4
    new_img=cv2.circle(new_img,(x,y),3,(0,0,255),-1)
cv2.imwrite("heatmaps/Marsden4-4-280-Epoch200/heatmap06097.jpg",new_img)
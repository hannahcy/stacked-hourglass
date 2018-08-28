from inference import Inference
import cv2
import numpy as np
import copy

infer=Inference(model='trained/hg_4WORDS_MULTI_64_8_1001')

img=cv2.imread("dataMarsden4-4WORDSSELECTION/09985.jpg")

img = cv2.resize(img, (256, 256))

hms=infer.predictHM(img)

print(np.shape(hms))
joint_list = ['e', 't', 'a', 'o', 'i', 'n', 's', 'r', 'h', 'l', 'd', 'c', 'u', 'm', 'f', 'p', 'g', 'w', 'y', 'b', 'v', 'k', 'x', 'j', 'q']
#new_img=np.array(img)

for i in range(np.shape(hms)[3]):
    max = np.amax(hms[0, :, :, i])
    temp = copy.deepcopy(hms[0, :, :, i]*(255/max))
    #img_small = cv2.resize(img, (64, 64))
    #img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('heatmaps-good/Marsden4-4WORDS/Individual/e1000_4WORDS_64_8_09985_'+str(joint_list[i])+'.jpg', temp) #np.maximum(img_small,temp))
    index=np.argmax(hms[0,:,:,i])
    print(str(joint_list[i]), index, max, np.argmin(hms[0,:,:,i]), np.amin(hms[0,:,:,i]))
    x=index%64*4
    y=int(index/64*4)
    print(str(joint_list[i]),x,y)
    #new_img=cv2.circle(img,(x,y),3,(0,255,0),-1) # green for MNIST
    new_img=cv2.circle(img,(x,y),3,(0,0,0),-1) # black for Marsden

new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY) # uncomment for Marsden
cv2.imwrite("heatmaps-good/Marsden4-4WORDS/e1000_4WORDS_64_8_09985.jpg",new_img)
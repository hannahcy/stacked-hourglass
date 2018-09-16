from inference import Inference
import cv2
import numpy as np
import copy

infer=Inference(model='trained/hg_26FREQ_CROPPED_256_8_2043')

examples = ['20013']
    #['25000','25001','25002','25003','25004','25005','25006','25007','25008','25009',
            #'25010', '25011', '25012', '25013', '25014', '25015', '25016', '25017', '25018', '25019']
    #['00010'] #['00358','00826','03783','04646','06097','06351','06402','06685','08684','08836',
 #           '09147','09184','09301','09410','09462','09530','09557','09795','09887','09985',
  #          '10004','10005','10006','10010','10037','10041','10046','10048','10071','10084',
   #         '20000','20001','20002','20003','20004','20005','20006','20007','20008','20009','20010','20011','20012']
joint_list = ['e', 't', 'a', 'o', 'i', 'n', 's', 'r', 'h', 'l', 'd', 'c', 'u', 'm', 'f', 'p', 'g', 'w', 'y', 'b', 'v', 'k', 'x', 'j', 'q', 'z']

for ex in range(len(examples)):
    img = cv2.imread("dataMarsden26FREQ-CROPPEDSELECTION/"+examples[ex]+".jpg")
    img = cv2.resize(img, (256, 256))
    hms = infer.predictHM(img)
    #print(hms.shape)
    for i in range(np.shape(hms)[3]):
        max = np.amax(hms[0, :, :, i])
        temp = copy.deepcopy(hms[0, :, :, i]*(150/max))
        #cv2.imwrite('testing/FORPAPERUNIF301_output' +examples[ex]+ '_' + str(joint_list[i]) + '.jpg', temp)
        temp_big = cv2.resize(temp, (256, 256))
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('heatmaps-good/Marsden26FREQCROPPED/e2043_26FREQ_CROPPED_256_8_'+examples[ex]+'_'+str(joint_list[i])+'.jpg', (img_grey+temp_big)) #np.maximum(img_grey,temp_big))
    print(examples[ex]+".jpg done")



    #index=np.argmax(hms[0,:,:,i])
    #print(str(joint_list[i]), index, max, np.argmin(hms[0,:,:,i]), np.amin(hms[0,:,:,i]))
    #x=index%64*4
    #y=int(index/64*4)
    #print(str(joint_list[i]),x,y)
    #new_img=cv2.circle(img,(x,y),3,(0,255,0),-1) # green for MNIST
    #new_img=cv2.circle(img,(x,y),3,(0,0,0),-1) # black for Marsden

#new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY) # uncomment for Marsden
#cv2.imwrite("heatmaps-good/Marsden4-4WORDS/e1296_4WORDS_64_8_10006.jpg",new_img)
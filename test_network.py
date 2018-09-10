from inference import Inference
import cv2
import numpy as np
import copy

'''
read in labels to multi-dimensional array
for each true letter,
get max activations over small patch for all 26 heatmaps
set max1, 2 -5
calculate each of 5 top-n accuracies
'''

trained_model = 'trained/hg_26FREQ_CROPPED_256_8_2043'
filelabelsIn = 'datasetMarsden26FREQ-TEST-CROPPED.txt'
dirImages = 'dataMarsden26FREQ-TEST-CROPPED/'

num_joints = 26
num_examples = 100
nameOffset = 10000

joint_list = ['e', 't', 'a', 'o', 'i', 'n', 's', 'r', 'h', 'l', 'd', 'c', 'u', 'm', 'f', 'p', 'g', 'w', 'y', 'b', 'v', 'k', 'x', 'j', 'q', 'z']

with open(filelabelsIn, 'r') as f:
    lines = f.read()

lines = lines.split('\n')
lines_as_list = []
for line in lines:
    # print(line)
    line = line.split()
    lines_as_list.append(line)

new_list = []

# print(str(lines_as_list))

for i in range(len(lines_as_list) - 1):
    new_line = []
    name = lines_as_list[i][0]
    new_line.append(name)
    type = lines_as_list[i][1]
    new_line.append(type)
    new_line.append(lines_as_list[i][2:])
    new_list.append(new_line)

infer=Inference(model=trained_model)

correct = [0,0,0,0,0]
total_letters = 0

for ex in range(num_examples):
    img = cv2.imread(dirImages+str(nameOffset+ex)+".jpg")
    img = cv2.resize(img, (256, 256))
    hms = infer.predictHM(img)
    for type in range(num_joints):
        tokens = copy.deepcopy(new_list[type+(ex*num_joints)][2])
        token = 0
        #print(str(nameOffset + ex), joint_list[type], np.amax(hms[0, :, :, type]))
        while token < len(tokens)-1:
            x = int((int(tokens[token])/256) * 64)
            y = int((int(tokens[token+1])/256) * 64)
            if x > 61:
                x = 61
            if y > 61:
                y = 61
            if x > 0:
                #print(x,y)
                total_letters += 1
                max_activations = [0] * num_joints
                for map in range(num_joints):
                    norm_max = np.amax(hms[0, :, :, map])
                    heatmap = copy.deepcopy(hms[0, :, :, map])
                    if norm_max > 0.5:
                        heatmap = heatmap*(1 / norm_max)
                    max_in_region = 0
                    for xx in range(x-2,x+3):
                        for yy in range(y-2,y+3):
                            activation = heatmap[xx][yy]
                            if activation > max_in_region:
                                max_in_region = activation
                    max_activations[map] = max_in_region
                #print(max_activations)
                max1 = np.amax(max_activations)
                index = max_activations.index(max1)
                if index == type:
                    correct[0] += 1
                    correct[1] += 1
                    correct[2] += 1
                    correct[3] += 1
                    correct[4] += 1
                else:
                    max_activations[index] = 0
                    max2 = max(max_activations)
                    index = max_activations.index(max2)
                    if index == type:
                        correct[1] += 1
                        correct[2] += 1
                        correct[3] += 1
                        correct[4] += 1
                    else:
                        max_activations[index] = 0
                        max3 = max(max_activations)
                        index = max_activations.index(max3)
                        if index == type:
                            correct[2] += 1
                            correct[3] += 1
                            correct[4] += 1
                        else:
                            max_activations[index] = 0
                            max4 = max(max_activations)
                            index = max_activations.index(max4)
                            if index == type:
                                correct[3] += 1
                                correct[4] += 1
                            else:
                                max_activations[index] = 0
                                max5 = max(max_activations)
                                index = max_activations.index(max5)
                                if index == type:
                                    correct[4] += 1
            token += 2

top1_accuracy = (correct[0]/total_letters)*100
top2_accuracy = (correct[1]/total_letters)*100
top3_accuracy = (correct[2]/total_letters)*100
top4_accuracy = (correct[3]/total_letters)*100
top5_accuracy = (correct[4]/total_letters)*100
print("Top1:"+str(top1_accuracy)+"%")
print("Top2:"+str(top2_accuracy)+"%")
print("Top3:"+str(top3_accuracy)+"%")
print("Top4:"+str(top4_accuracy)+"%")
print("Top5:"+str(top5_accuracy)+"%")
print(str(correct[0]),str(correct[1]),str(correct[2]),str(correct[3]),str(correct[4]),str(total_letters))

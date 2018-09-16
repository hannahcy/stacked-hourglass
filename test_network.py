from inference import Inference
import cv2
import numpy as np
import copy


def makeGaussian(height, width, sigma=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    sigma is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]
    if center is None:
        x0 = width // 2
        y0 = height // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def generate_hm(height, width, joints, locations):
    """ Generate a full Heap Map for every joints in an array
    Args:
        height			: Wanted Height for the Heat Map
        width			: Wanted Width for the Heat Map
        joints			: Array of Joints
        locations		: list of lists of locations (for each joint) HANNAH
    """
    num_joints = len(joints)
    num_tokens = len(locations[0]) * 50
    hm = np.zeros((height, width, num_joints, num_tokens), dtype=np.float32)
    for typea in range(num_joints):
        # if not(np.array_equal(joints[i], [-1,-1])) and weight[i] == 1:
        # s = int(np.sqrt(maxlenght) * maxlenght * 10 / 4096) + 2
        s = int(
            np.sqrt(width) * width * 10 / 4096) - 5  # CHANGED FROM +2, -5 for "280-small", -10 for "280-tiny" HANNAH
        tokena = 0
        while tokena < len(locations[typea][2])-1:
            # print(locations[type][token][0])
            if int(locations[typea][2][tokena]) < 1:
                hm[:, :, typea, tokena] = np.zeros((height, width))
            else:
                hm[:, :, typea, tokena] = makeGaussian(height, width, sigma=s, center=(
                int(locations[typea][2][tokena]), int(locations[typea][2][tokena+1])))
            tokena +=2
    # have hm of shape [height,width,types,tokens]
    # need to combine all the tokens for each type (simple addition, since they're all zeros otherwise?
    condensed_hm = np.zeros((height, width, num_joints), dtype=np.float32)
    for typeb in range(num_joints):
        for tokenb in range(num_tokens):
            condensed_hm[:, :, typeb] = condensed_hm[:, :, typeb] + hm[:, :, typeb, tokenb]
    return condensed_hm
'''
read in labels to multi-dimensional array
for each true letter,
get max activations over small patch for all 26 heatmaps
set max1, 2 -5
calculate each of 5 top-n accuracies
'''

topN = True
F1 = True
full_hm = False
threshold = 0.5

trained_model = 'trained/hg_26FREQ_CROPPED_256_4_216' # 'trained/hg_26FREQ_CROPPED_256_8_501' #
filelabelsIn = 'datasetMarsdenREALTEST.txt' #'convicts.txt' # 'datasetMarsden26FREQ-CROPPEDSELECTION.txt' #
dirImages = 'dataMarsdenREALTEST/' #'dataMarsden26FREQ-CROPPEDSELECTION/' #

epoch = 216
data = "Real" # "Training" "Validation" "Testing" "Real"
num_joints = 26
num_examples = 20 # 100 for training, validation, testing. 20 for REALTEST
nameOffset = 25000 # 8035 for training, 9442 for validation, 10000 for testing, 25000 for REALTEST

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

if full_hm:
    # for each image
    # construct Gaussian heatmap (copy code from datagen)
    # get heatmap through infer
    # compare the two, pixelwise
    # try just average difference, 1-ans, ans*100
    total_err = 0
    for ex in range(num_examples):
        img = cv2.imread(dirImages+str("{:05n}".format(nameOffset+ex))+".jpg")
        img = cv2.resize(img, (256, 256))
        hms = infer.predictHM(img)
        start = ex*num_joints
        end = start + num_joints
        locations = copy.deepcopy(new_list[start:end])
        targets = generate_hm(64,64,joint_list,locations)
        height = 64
        for map in range(num_joints):
            output = copy.deepcopy(hms[0, :, :, map])
            target = copy.deepcopy(targets[:,:,map])
            total = 0
            for i in range(height):
                for j in range(height):
                    diff = output[i][j] - target[i][j]
                    abs_diff = abs(diff)
                    total += abs_diff
            err = total/(height*height)
            print("Average error for " + joint_list[map] + ": " + str(err))
            total_err += err
        print(str(nameOffset+ex)+".jpg Done!")
        print(str(total_err/(num_joints*(ex+1))) + " error so far!")
    average_err = total_err/(num_joints*num_examples)
    print("Average Error: "+str(average_err))
    accuracy = (1-average_err)*100
    print("Average Accuracy: " + str(accuracy)+"%")


if F1:
    total_letters = 0
    true_pos = 0
    false_pos = 0
    for ex in range(num_examples):
        img = cv2.imread(dirImages + str("{:05n}".format(nameOffset + ex)) + ".jpg")
        #img = cv2.imread(dirImages+str(nameOffset+ex)+".jpg")
        img = cv2.resize(img, (256, 256))
        hms = infer.predictHM(img)
        for type in range(num_joints):
            tokens = copy.deepcopy(new_list[type+(ex*num_joints)][2])
            token = 0
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
                        #norm_max = np.amax(hms[0, :, :, map])
                        heatmap = copy.deepcopy(hms[0, :, :, map])
                        #if norm_max > 0.5:
                            #heatmap = heatmap*(1 / norm_max)
                        max_in_region = 0
                        for xx in range(x-2,x+3):
                            for yy in range(y-2,y+3):
                                activation = heatmap[yy][xx]
                                if activation > max_in_region:
                                    max_in_region = activation
                        max_activations[map] = max_in_region
                    for letter in range(len(max_activations)):
                        if max_activations[letter] > threshold:
                            if letter == type:
                                true_pos += 1
                            else:
                                false_pos += 1
                token += 2
    precision = (true_pos/(true_pos+false_pos))*100
    recall = (true_pos/total_letters)*100
    f1 = (2*precision*recall)/(precision+recall) # harmonic average of precision and recall
    print(trained_model)
    print("Epoch: "+str(epoch) + ", " + data + " dataset" )
    print("Precision: "+str(round(precision,2))+"%")
    print("Recall: " + str(round(recall,2)) + "%")
    print("F1 score: " + str(round(f1,2)) + "%")
    print("Total letters: "+str(total_letters)+" True pos: "+str(true_pos)+" False pos: "+str(false_pos))

if topN:
    correct = [0, 0, 0, 0, 0]
    total_letters = 0
    for ex in range(num_examples):
        img = cv2.imread(dirImages + str("{:05n}".format(nameOffset + ex)) + ".jpg")
        #img = cv2.imread(dirImages+str(nameOffset+ex)+".jpg")
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
                                activation = heatmap[yy][xx]
                                if activation > max_in_region:
                                    max_in_region = activation
                        max_activations[map] = max_in_region
                    #print(joint_list[type]+":")
                    max1 = np.amax(max_activations)
                    index = max_activations.index(max1)
                    #print("1: "+joint_list[index])
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
                        #print("2: "+joint_list[index])
                        if index == type:
                            correct[1] += 1
                            correct[2] += 1
                            correct[3] += 1
                            correct[4] += 1
                        else:
                            max_activations[index] = 0
                            max3 = max(max_activations)
                            index = max_activations.index(max3)
                            #print("3: "+joint_list[index])
                            if index == type:
                                correct[2] += 1
                                correct[3] += 1
                                correct[4] += 1
                            else:
                                max_activations[index] = 0
                                max4 = max(max_activations)
                                index = max_activations.index(max4)
                                #print("4: "+joint_list[index])
                                if index == type:
                                    correct[3] += 1
                                    correct[4] += 1
                                else:
                                    max_activations[index] = 0
                                    max5 = max(max_activations)
                                    index = max_activations.index(max5)
                                    #print("5: "+joint_list[index])
                                    if index == type:
                                        correct[4] += 1
                token += 2

    top1_accuracy = (correct[0]/total_letters)*100
    top2_accuracy = (correct[1]/total_letters)*100
    top3_accuracy = (correct[2]/total_letters)*100
    top4_accuracy = (correct[3]/total_letters)*100
    top5_accuracy = (correct[4]/total_letters)*100
    print("Top1: "+str(round(top1_accuracy,2))+"%")
    print("Top2: "+str(round(top2_accuracy,2))+"%")
    print("Top3: "+str(round(top3_accuracy,2))+"%")
    print("Top4: "+str(round(top4_accuracy,2))+"%")
    print("Top5: "+str(round(top5_accuracy,2))+"%")
    print(str(correct[0]),str(correct[1]),str(correct[2]),str(correct[3]),str(correct[4]),str(total_letters))

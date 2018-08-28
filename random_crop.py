import random as r
import cv2

'''
Set variables:
width
height
size of crop
number of types

Randomly choose x: 0-(width-256) and y: 0-(height-256)
Read in image
Crop from x to x+256, y to y+256

Read in relevant lines from file
Copy first element (name) over without change
Copy second element (type) over without change
For each remaining item //2, (ie each remaining pair)
Turn into an int, subtract x from first
Turn into int, subtract y from second
If either are outside the range 0-256, turn them both into -1
Copy into new file
'''

width = 700
height = 700
crop_size = 256

num_types = 25

filelabelsIn = 'datasetMarsden25-25WORDS-MULTI.txt'
filelabelsOut = 'datasetMarsden25-25WORDS-CROPPED.txt'

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

final_list = []
### new_list now contains all info

# for loop to process all images at once
for image in range(10000):

    fileimageIn = 'dataMarsden25-25WORDS/'+str("{:05n}".format(image))+'.jpg'
    fileimageOut = 'dataMarsden25-25CROPPED/'+str("{:05n}".format(image))+'.jpg'

    x = r.randint(0, width-crop_size)
    y = r.randint(0, height-crop_size)
    #print(x, y)

    orig_img = cv2.imread(fileimageIn)
    crop_img = orig_img[y:y+crop_size, x:x+crop_size]
    cv2.imwrite(fileimageOut, crop_img)

    #print(str(new_list))
    #print(len(new_list))

    for type in range(num_types):
        name = new_list[type+(image*num_types)][0]
        letter = new_list[type+(image*num_types)][1]
        temp_list = []
        temp_list.append(name)
        temp_list.append(letter)
        for token in range(len(new_list[type+(image*num_types)][2])//2):
            token_x = int(new_list[type+(image*num_types)][2][token * 2])
            token_y = int(new_list[type+(image*num_types)][2][(token * 2) + 1])
            new_x = token_x - x
            new_y = token_y - y
            if new_x < 7 or new_y < 7 or new_x >= (crop_size-7) or new_y >= (crop_size-7):
                new_x = -1
                new_y = -1
            temp_list.append(str(new_x))
            temp_list.append(str(new_y))
        final_list.append(temp_list)

#print(str(final_list))

with open(filelabelsOut, 'a') as f:
    for line in range(len(final_list)):
        f.write(' '.join(final_list[line]))
        f.write('\n')



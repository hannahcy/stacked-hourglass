'''
read in a file
line by line, append the relevant information to new line
append new line to new file
'''

fileIn = 'datasetMarsden4-4WORDS.txt'
fileOut = 'datasetMarsden4-4WORDS-MULTI.txt'
num_types = 4
types = ['e', 't', 'a', 'o', 'i', 'n', 's', 'r' ,'h', 'l', 'd', 'c', 'u', 'm', 'f', 'p', 'g', 'w', 'y', 'b', 'v', 'k', 'x', 'j', 'q']

with open(fileIn, 'r') as f:
    lines = f.read()

lines = lines.split('\n')
lines_as_list = []
for line in lines:
    #print(line)
    line = line.split()
    lines_as_list.append(line)

#print(str(lines_as_list))

new_list = []

#print(str(lines_as_list))

for i in range(len(lines_as_list)-1):
    new_line = []
    name = lines_as_list[i][0]
    name = name[:-1]
    new_line.append(name)
    new_line.append(lines_as_list[i][num_types+1:])
    new_list.append(new_line)

#print(str(new_list))

final_list = []

i = 0
while i < len(new_list):
    for type in range(num_types):
        final_line = []
        name = new_list[i][0]
        final_line.append(name)
        final_line.append(types[type])
        #print(len(new_list[i][1]))
        for token in range(len(new_list[i][1])//2):
            #### THESE WERE ORIGINALLY GENERATED IN THE WRONG ORDER SO THIS FIXES IT
            final_line.append(new_list[i + token][1][(type * 2) + 1])  # append x
            final_line.append(new_list[i+token][1][type*2]) # append y
        #print(final_line)
        final_list.append(final_line)
    i = i + num_types

print(str(final_list))
with open(fileOut, 'a') as f:
    for line in range(len(final_list)):
        f.write(' '.join(final_list[line]))
        f.write('\n')

#
    #f.write(str(final_list))
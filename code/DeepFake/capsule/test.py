import test22
import os
import hashlib
import random


# dataset = '/data1/comp/comp/data1/fakedata/FF++/manipulated_sequences/pic/Deepfakes/c23/test/'
dataset = '/data1/images/stylegan/val/fake/'
test = test22.Test()
pro1 = 0
pro2 = 0
count1 = 0
count2 = 0
n = 0
# file = open('result2.txt', 'w')

for filename in os.listdir(dataset):
    # print(filename)
    label, prob = test.detect(dataset + '/' + filename)
    # fd = open(dataset + '/' + filename, "rb")
    # f = fd.read()
    # pmd5 = hashlib.md5(f)
    if label == 0:
        count1 = count1 + 1
        pro1 = pro1 + prob
        # prob3 = random.uniform(0, 0.1)
        # file.write(pmd5.hexdigest())
        # file.write(' ')
        # file.write(str(1 - prob))
        # file.write('\n')
        print(filename, ' is real', ' ', prob)
    elif label == 1:
        count2 = count2 + 1
        pro2 = pro2 + prob
        # prob3 = random.uniform(0.9, 1)
        # file.write(pmd5.hexdigest())
        # file.write(' ')
        # file.write(str(prob))
        # file.write('\n')
        print(filename, ' is fake', ' ', prob)
    else:
        print(1)
    n = n + 1
    if n == 500:
        break
if count1 != 0:
    print(pro1/count1)
print(pro2/count2)
print(count1, ' ', count2)
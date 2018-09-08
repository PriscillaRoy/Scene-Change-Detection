import numpy as np

def ThreeD(a, b, c):
    lst = [[['#' for col in range(a)] for col in range(b)] for row in range(c)]
    return lst


list1 = ThreeD(2,3,4)
list1[0][2] = [5,6]
list1.append([3,7])
list1.append([2,3])
list1.append([4,5])
list1.append([7,8])

list1[1][0][0] = [7]

#thresholded = [[-1] * 3 for _ in range(4)]
print(list1)
print(list1[5][1])


list2 = [[2,3],[4,5],[6,7],[8,9]]
list2.append([10,11])
list3 = [[6,7]]
list3.append([8,9])
print ("List2 : ",list2)
print ("List3 : ",list3)

for i in range(0, len(list3)):
    if(list3[i] in list2):
        index = list2.index(list3[i])
        print(index)
        list2[index] = [255,255]

print ("List2 : ",list2)
print ("List3 : ",list3)

for i in range(0,len(list2)):
    print(list2[i])
    if(list2[i] ==[255,255]):
        print("Inside if loop")
        continue
    else:
        print("Inside else")
        list2[i] =[0,0]

print ("List2 : ",list2)
print ("List3 : ",list3)

list4 = []
for i in range(0,6):
    list4.append(i)


x = np.reshape(list4, (2,3))
print(x)



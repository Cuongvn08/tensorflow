# -*- coding: utf-8 -*-

dict = {}
#dict['Key(0)'] = 0

for i in range(10):
    dict['Key(%i)'%i] = i

print(len(dict))

#print(dict) 
# {'a': 2, 'b': 1}
# keys must be different

#for i in len(dict):
    #print(dict[i])
    
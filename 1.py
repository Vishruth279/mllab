import statistics
from math import sqrt
import numpy as np
l1=[115.3, 115.3, 195.5, 120.5, 110.2, 90.4, 105.6, 110.9, 116.3, 122.3, 125.4]
a=len(l1)

#mean
s=sum(l1)
mean=(s/a)
print("mean:",mean)

#median
l1.sort()
if a%2==0:
    med1=l1[a//2]
    med2=l1[a//2 - 1]
    med=(med1+med2)/2
else:
    med=l1[a//2]
print("median:",med)
    
#mode
uniq=[]
mod=[]
for i in l1:
    if i not in uniq:
        uniq.append(i)
    else:
        mod.append(i)
print("mode:",mod)

#standard deviation
SUM= 0
for i in l1 :
    SUM +=(i-mean)**2
stdeV = sqrt(SUM/(len(l1)-1)) 
print("Standard devation",stdeV)

# varience
var=stdeV**2
print("varience",var)

# min-max normalization
min_value = min(l1)
max_value = max(l1)
min_max_norm = [(value - min_value) / (max_value - min_value) for value in l1]
print("Min-Max Normalization:", min_max_norm)

# min-max standardization
mean = sum(l1) / len(l1)
variance = sum((value - mean) ** 2 for value in l1) / len(l1)
standardization = [(value - mean) / variance ** 0.5 for value in l1]
print("Standardization:", standardization)
print("*******")

#inbuilt
print("using inbuilt fn:")
x=statistics.mean(l1)
print("mean:",x)

y=statistics.median(l1)
print("median:",y)

z=statistics.mode(l1)
print("mode:",z)

p=statistics.stdev(l1)
print("standard deviation:",p)

q=p**2
print("varience",q)

min_max_norm = (l1 - np.min(l1)) / ( np.max(l1) - np.min(l1))
print("Min-Max Normalization:", min_max_norm)

standardization = (l1 - np.mean(l1)) / np.std(l1)
print("Standardization:", standardization)

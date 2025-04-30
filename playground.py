import numpy as np

l1 = np.random.randint(1, 100, size=30)
l2 = np.random.randint(1, 100, size=30)

l3 = [(i + j) / 2 for i, j in zip(l1, l2)]

print(l3)

print("l1 and l2 avg", (np.mean(l1) + np.mean(l2))/2)
print("l3 avg: ", np.mean(l3))
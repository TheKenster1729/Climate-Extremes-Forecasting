<<<<<<< HEAD
import numpy as np

=======
<<<<<<< HEAD
import pandas as pd

df = pd.read_csv("full_processed_data.csv")
print(df.head())
=======
import numpy as np

>>>>>>> 2e2e3d2b (WIP)
l1 = np.random.randint(1, 100, size=30)
l2 = np.random.randint(1, 100, size=30)

l3 = [(i + j) / 2 for i, j in zip(l1, l2)]

print(l3)

print("l1 and l2 avg", (np.mean(l1) + np.mean(l2))/2)
<<<<<<< HEAD
print("l3 avg: ", np.mean(l3))
=======
print("l3 avg: ", np.mean(l3))
>>>>>>> be14593b33a08ae29df60822b7dcc702a6d70f62
>>>>>>> 2e2e3d2b (WIP)

import numpy as np
import matplotlib.pyplot as plt


# alcohol,volatile acidity,sulphates,pH,quality
dataset = np.loadtxt(
    "wine_quality.csv",
    delimiter=",",
    skiprows=1
)

# Task 1.1
print(dataset.mean(axis=0))
print(np.median(dataset))
print(dataset.var(ddof=1, axis=0))
print(dataset.min(axis=0))
print(dataset.max(axis=0))

# Task 1.2
max_qual = dataset[:, -1].max(axis=0)
print("\n", dataset[dataset[:, -1] == max_qual][:, -2].mean())

# Task 1.3
max_qual = dataset[:, -1].max(axis=0)
min_qual = dataset[:, -1].min(axis=0)
rebuilt_qual = dataset[:, -1] - min_qual
print("\n", rebuilt_qual / (max_qual - min_qual))

# Task 2.1
unique_qual_lvl = np.unique(dataset[:, -1])
plt.pie(unique_qual_lvl, labels=unique_qual_lvl)
plt.show()

# Task 2.2
max_qual = dataset[:, -1].max(axis=0)
min_qual = dataset[:, -1].min(axis=0)
low_quality = dataset[dataset[:, -1] == min_qual]
high_quality = dataset[dataset[:, -1] == max_qual]

plt.xlabel("Volatile acidity")
plt.ylabel("Alcohol")
plt.scatter(x=low_quality[:, 1], y=low_quality[:, 0], edgecolors="green")
plt.scatter(x=high_quality[:, 1], y=high_quality[:, 0], edgecolors="orange")
plt.show()

# Глядя на такой график, можно сделать вывод о том, что в основном, дешёвом вине
# преобладает летучая кислота, а в составе дорогого её явно меньше, при этом объём спирта больше.
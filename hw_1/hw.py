import numpy as np


# Task 1
print("Task 1")
temps = np.array(
    [15.2, 16.8, 14.5, 17.0, 16.1],
    dtype="float32"
)

print(temps.sum())
print(temps.mean())
print(temps.min())
print(temps.max())

# Task 2
print("\nTask 2")

h1 = np.array([45, 50, 47], dtype="float32")
h2 = np.array([48, 46, 52], dtype="float32")

print(h1 + h2)
print(h1 * h2)

# Task 3
print("\nTask 3")

X = np.array([
    [20.1, 20.3, 19.8],
    [21.0, 20.7, 20.2],
    [19.5, 19.8, 19.3],
    [20.8, 21.1, 20.6]
],
    dtype="float32"
)

print(X.mean(axis=0))
print(X.sum(axis=1))
print(X.var(ddof=1, axis=0))
print(X.var(ddof=1, axis=0).min())

# Task 4
print("\nTask 4")

X = np.array([
    [20.1, 20.3, 19.8],
    [21.0, 20.7, 20.2],
    [19.5, 19.8, 19.3],
    [20.8, 21.1, 20.6]
],
    dtype="float32"
)

col_min = X.min(axis=0)
col_max = X.max(axis=0)
col_range = col_max - col_min

print((X - col_min) / col_range)

# Task 5
print("\nTask 5")

ph = np.array([
    [7.1, 7.4, 7.0],
    [6.9, 7.2, 7.1],
    [7.3, 7.5, 7.2],
    [7.0, 7.1, 6.8],
    [6.8, 6.9, 6.7],
    [7.4, 7.6, 7.3]
],
    dtype="float32"
)

print(ph.mean(axis=1))
print(ph.sum(axis=0))
print(ph.sum(axis=1))
print(ph.var(ddof=1, axis=0))

# Task 6
print("\nTask 6")

consumption = np.array([
    [8, 6, 5], # Mon
    [10, 7, 6], # Tue
    [ 9, 8, 7], # Wed
    [11, 10, 9], # Thu
    [14, 12, 11], # Fri
    [16, 15, 13], # Sat
    [12, 11, 10] # Sun
],
    dtype="float32"
)
days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
houses = ['H1','H2','H3']

print(consumption.sum(axis=0))
print(consumption.sum(axis=1))
print(consumption.mean(axis=0))
print(days[consumption.sum(axis=1).argmax()])
print(consumption.var(ddof=1, axis=0)) # -> наиболее стабильным по потреблению является H1

# Task 7
print("\nTask 7")

sensors = np.array([
    [15, 101, 20, 0.5],
    [16, 100, 21, 0.6],
    [15, 102, 19, 0.4],
    [17, 103, 22, 0.7],
    [18, 104, 23, 0.6],
    [19, 105, 24, 0.8],
    [17, 103, 22, 0.5]
],
    dtype="float32"
)
types = ['TempSensor','PressureSensor','FlowSensor','VibrationSensor']

print(sensors.sum(axis=0))
print(sensors.mean(axis=0))
print(sensors.var(ddof=1, axis=0))
print(types[sensors.sum(axis=0).argmax()])
print(types[sensors.var(ddof=1, axis=0).argmin()])
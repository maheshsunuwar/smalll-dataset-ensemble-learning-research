import numpy as np
import skfuzzy as fuzz

# Generate some sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1])

# Define the input and output universes
x_universe = np.linspace(x.min(), x.max(), num=100)
y_universe = np.linspace(y.min(), y.max(), num=100)

# Define the fuzzy sets
x_set = fuzz.trapmf(x_universe, [1, 2, 4, 6])
y_set = fuzz.trapmf(y_universe, [0, 0, 1, 1])

# Define the fuzzy rules
rules = []
rules.append(fuzz.relation_min(x_set[0], y_set[0]))
rules.append(fuzz.relation_min(x_set[0], y_set[1]))
rules.append(fuzz.relation_min(x_set[1], y_set[0]))
rules.append(fuzz.relation_min(x_set[2], y_set[2]))
rules.append(fuzz.relation_min(x_set[3], y_set[2]))
rules.append(fuzz.relation_min(x_set[4], y_set[2]))
rules.append(fuzz.relation_min(x_set[5], y_set[1]))
rules.append(fuzz.relation_min(x_set[6], y_set[0]))
rules.append(fuzz.relation_min(x_set[6], y_set[1]))
rules.append(fuzz.relation_min(x_set[7], y_set[1]))
rules.append(fuzz.relation_min(x_set[8], y_set[2]))
rules.append(fuzz.relation_min(x_set[9], y_set[2]))

# Define the output universe
z_universe = np.linspace(0, 1, num=100)

# Apply the fuzzy rules to the input data
z = np.zeros((len(x_universe), len(y_universe)))
for i in range(len(x_universe)):
    for j in range(len(y_universe)):
        inputs = [x_universe[i], y_universe[j]]
        outputs = [fuzz.interp_membership(x_universe, x_set, inputs[0]),
                   fuzz.interp_membership(y_universe, y_set, inputs[1])]
        z[i, j] = fuzz.defuzz(z_universe, fuzz.relation_and(*outputs), 'centroid')

# Predict on new data
x_test = np.array([7.5, 3.5, 2.0])
y_test = np.array([0.8, 0.2, 0.5])
z_test = np.zeros(len(x_test))
for i in range(len(x_test)):
    inputs = [x_test[i], y_test[i]]
    outputs = [fuzz.interp_membership(x_universe, x_set, inputs[0]),
               fuzz.interp_membership(y_universe, y_set, inputs[1])]
    z_test[i] = fuzz.defuzz(z_universe, fuzz.relation_and(*outputs), 'centroid')
    if z_test[i] > 0.5:
        print("Input ({}, {}) belongs to class 1".format(x_test[i], y_test[i]))
    else:
        print("Input ({}, {}) belongs to class 0".format(x_test[i], y_test[i]))

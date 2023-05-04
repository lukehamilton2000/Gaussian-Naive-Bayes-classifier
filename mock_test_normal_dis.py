import numpy
import math
# this program uses a Gaussian Naive Bayes classifier to model this data

# suppose we have a new point X (in this situation its (2.0, 2.1))
x1 = 2
x2 = 2.1

# initialises arrays with data (insert your own here)
class1_density = [1.52, 1.46, 1.68, 1.47, 1.53, 1.6]
class1_hardness = [1.95, 1.76, 2.11, 2.08, 1.94, 1.97]
class2_density = [2.04, 1.78, 1.67, 1.82, 1.65]
class2_hardness = [2.24, 2.69, 2.66, 2.43, 2.9]
class3_density = [2.15, 1.55, 2.92, 1.23]
class3_hardness = [1.04, 1.08, 0.84, 0.85]

# calcs all the means for classes 1, 2, 3
class1_density_mean = numpy.mean(class1_density)
class1_hardness_mean = numpy.mean(class1_hardness)
class2_density_mean = numpy.mean(class2_density)
class2_hardness_mean = numpy.mean(class2_hardness)
class3_density_mean = numpy.mean(class3_density)
class3_hardness_mean = numpy.mean(class3_hardness)

# calcs standard deviation
class1_density_SD = numpy.std(class1_density)
class1_hardness_SD = numpy.std(class1_hardness)
class2_density_SD = numpy.std(class2_density)
class2_hardness_SD = numpy.std(class2_hardness)
class3_density_SD = numpy.std(class3_density)
class3_hardness_SD = numpy.std(class3_hardness)

# calcs the c0 for each class (number of points in each class divided by total number of points)
class1_c0 = (6 / 15)
class2_c0 = (5 / 15)
class3_c0 = (4 / 15)


# P probability that x1 (density) given C1 (class 1)
Px1C1 = (1.0 / (class1_density_SD * numpy.sqrt(2.0 * numpy.pi))) * numpy.exp(-0.5 * numpy.square((x1 - class1_density_mean) / class1_density_SD))
Px2C1 = (1.0 / (class1_hardness_SD * numpy.sqrt(2.0 * numpy.pi))) * numpy.exp(-0.5 * numpy.square((x2 - class1_hardness_mean) / class1_hardness_SD))
Px1C2 = (1.0 / (class2_density_SD * numpy.sqrt(2.0 * numpy.pi))) * numpy.exp(-0.5 * numpy.square((x1 - class2_density_mean) / class2_density_SD))
Px2C2 = (1.0 / (class2_hardness_SD * numpy.sqrt(2.0 * numpy.pi))) * numpy.exp(-0.5 * numpy.square((x2 - class2_hardness_mean) / class2_hardness_SD))
Px1C3 = (1.0 / (class3_density_SD * numpy.sqrt(2.0 * numpy.pi))) * numpy.exp(-0.5 * numpy.square((x1 - class3_density_mean) / class3_density_SD))
Px2C3 = (1.0 / (class3_hardness_SD * numpy.sqrt(2.0 * numpy.pi))) * numpy.exp(-0.5 * numpy.square((x2 - class3_hardness_mean) / class3_hardness_SD))

# final calc!!!!!!!!!
class1_L = math.log(class1_c0) + math.log(Px1C1) + math.log(Px2C1)
class2_L = math.log(class2_c0) + math.log(Px1C2) + math.log(Px2C2)
class3_L = math.log(class3_c0) + math.log(Px1C3) + math.log(Px2C3)

print("class 1 L", class1_L)
print("class 2 L", class2_L)
print("class 3 L", class3_L)

# closest to 0 is class 2, showing that the new point X is closest to class 

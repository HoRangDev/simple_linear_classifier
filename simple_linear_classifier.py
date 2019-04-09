import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GradientDescent:
    def __init__(self, init_weight, learning_rate, datas, class_w1_idx, class_w2_idx, epochs):
        self.learning_rate = learning_rate
        self.class_1_data = datas[class_w1_idx]
        self.class_2_data = datas[class_w2_idx]
        self.class_w1_idx = class_w1_idx
        self.class_w2_idx = class_w2_idx
        self.weight = init_weight
        self.epochs = epochs

    def get_missclasified_samples(self):
        missclasified = []
        for sample in self.class_1_data:
            classified = np.dot(self.weight, sample)
            if (classified <= 0):
                # missclasified
                missclasified.append(sample)
        for sample in self.class_2_data:
            classified = np.dot(self.weight, sample)
            if (classified >= 0):
                missclasified.append(sample)
        return missclasified

    def d_criterion_func(self, missclassified_samples):
        return None

    def learn(self):
        k = 0
        ret = [[], [], []]
        while (k < self.epochs):
            missclassified_samples = self.get_missclasified_samples()
            print('current epochs: ' , k, ' weight: ', self.weight, ' missclasified: ', len(missclassified_samples))
            ret[0].append(self.weight[0])
            ret[1].append(self.weight[1])
            ret[2].append(float(len(missclassified_samples)))
            k += 1
            prog = self.learning_rate * self.d_criterion_func(missclassified_samples)
            self.weight = self.weight - prog
        return ret

class PerceptronClassifier(GradientDescent):
    def __init__(self, init_weight, learning_rate, datas, class_w1_idx, class_w2_idx, epochs):
        super().__init__(init_weight, learning_rate,
                        datas, class_w1_idx, class_w2_idx,
                       epochs)

    def d_criterion_func(self, missclassified_samples):
        return np.sum(missclassified_samples)
   

class RelaxationClassifier(GradientDescent):
        def __init__(self, init_weight, learning_rate, datas, class_w1_idx, class_w2_idx, epochs, margin):
            super().__init__(init_weight, learning_rate, 
                             datas, class_w1_idx, class_w2_idx, 
                             epochs)
            self.margin = margin

        def d_criterion_func(self, missclassified_samples):
            # b is margin
            sum = np.zeros(shape = 2)
            for sample in missclassified_samples:
                sum += ((np.dot(self.weight, sample) - self.margin)/np.power(np.linalg.norm(sample), 2))*sample
            return sum

class LMSClassifier(GradientDescent):
    def __init__(self, init_weight, learning_rate, datas, class_w1_idx, class_w2_idx, epochs, margin):
            super().__init__(init_weight, learning_rate, 
                             datas, class_w1_idx, class_w2_idx, 
                             epochs)
            self.margin = margin

    def d_criterion_func(self, missclassified_samples):
        sum = np.zeros(shape = 2)
        for sample in missclassified_samples:
            sum += ((np.dot(self.weight, sample) - self.margin) * sample)
        return sum

def load_datas(filePath):
    # 3 classes, each 10 samples, x y class_idx
    loaded_data = [[], [] , []]
    data_file = open(filePath)

    while True:
        read_line = data_file.readline()
        if (not read_line):
            break;
        splited = read_line.split(' ')
        loaded_data[int(splited[2])].append(np.array([float(splited[0]), float(splited[1])]))
    return loaded_data

def print_datas(datas):
    count = 0
    for data_list in datas:
        print('Class ', count)
        count += 1
        for data in data_list:
            print('x:', data[0], ' y:', data[1])

def mean2D(samples):
    meanX = 0
    meanY = 0
    for sample in samples:
        meanX += sample[0]
        meanY += sample[1]
    meanX /= len(samples)
    meanY /= len(samples)
    return (meanX, meanY)

datas = load_datas('data.txt')
print_datas(datas)

means = [mean2D(datas[0]), mean2D(datas[1]), mean2D(datas[2])]

percep_classifier = PerceptronClassifier(np.array([(means[0][0] + means[1][0])/2.0, (means[0][1] + means[1][1])/2.0]),
                                        0.00009, datas, 0, 1, 1000 )
percep_classifier.learn()

#relax_classifier = RelaxationClassifier(np.array([1.0, 1.5]), 0.001, datas, 0, 2, 1500, 0.1)
relax_classifier = RelaxationClassifier(np.array([5.0, 6.0]), 0.005, datas, 0, 2, 1000, 0.01)
relax_classifier.learn()

lms_classifier = LMSClassifier(np.array([5.0, 6.0]), 0.005, datas, 0, 2, 1000, 0.01)
lms_classifier.learn()
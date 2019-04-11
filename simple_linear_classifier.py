import os
import numpy as np
import matplotlib.pyplot as plt

class LearningProc:
    def __init__(self, init_weight, learning_rate, datas, class_w1_idx, class_w2_idx, epochs, bias):
        self.learning_rate = learning_rate
        self.class_1_data = datas[class_w1_idx]
        self.class_2_data = datas[class_w2_idx]
        self.class_w1_idx = class_w1_idx
        self.class_w2_idx = class_w2_idx
        self.weight = init_weight
        self.epochs = epochs
        self.bias = bias

    def get_missclasified_samples(self):
        missclasified = []
        for sample in self.class_1_data:
            classified = np.dot(self.weight, sample)
            if (classified <= self.bias):
                # missclasified
                missclasified.append([sample, -1])
        for sample in self.class_2_data:
            classified = np.dot(self.weight, sample)
            if (classified >= self.bias):
                missclasified.append([sample, 1])
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

class PerceptronClassifier(LearningProc):
    def __init__(self, init_weight, learning_rate, datas, class_w1_idx, class_w2_idx, epochs):
        super().__init__(init_weight, learning_rate,
                        datas, class_w1_idx, class_w2_idx,
                       epochs, 0.0)

    def d_criterion_func(self, missclassified_samples):
        crit = 0.0
        for sample in missclassified_samples:
            crit += sample[0] * sample[1]
        return crit

class RelaxationClassifier(LearningProc):
        def __init__(self, init_weight, learning_rate, datas, class_w1_idx, class_w2_idx, epochs, bias):
            super().__init__(init_weight, learning_rate, 
                             datas, class_w1_idx, class_w2_idx, 
                             epochs,
                             bias)

        def get_missclasified_samples(self):
            missclasified = []
            for sample in self.class_1_data:
                classified = np.dot(self.weight, sample)
                if (classified <= self.bias):
                    # missclasified
                    missclasified.append([sample, -1])
            for sample in self.class_2_data:
                classified = np.dot(self.weight, sample)
                if (classified <= self.bias):
                    missclasified.append([sample, 1])
            return missclasified

        def d_criterion_func(self, missclassified_samples):
            # b is margin
            sum = np.zeros(shape = 2)
            for sample in missclassified_samples:
                sum += ((np.dot(self.weight, sample[0])-self.bias)/np.power(np.linalg.norm(sample[0]), 2))*sample[0]
            return sum

class LMSClassifier(LearningProc):
    def __init__(self, init_weight, learning_rate, datas, class_w1_idx, class_w2_idx, epochs, bias):
            super().__init__(init_weight, learning_rate, 
                             datas, class_w1_idx, class_w2_idx, 
                             epochs,
                             bias)

    def d_criterion_func(self, missclassified_samples):
        sum = np.zeros(shape = 2)
        for sample in missclassified_samples:
            sum += ((np.dot(self.weight, sample[0]) - self.bias) * sample[0])
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
    meanX = 0.0
    meanY = 0.0
    for sample in samples:
        meanX += sample[0]
        meanY += sample[1]
    meanX /= len(samples)
    meanY /= len(samples)
    return (meanX, meanY)

def max2D(samples):
    maxX = -float('inf')
    maxY = -float('inf')
    for sample in samples:
        if (maxX < sample[0]):
            maxX = sample[0]
        if (maxY < sample[1]):
            maxY = sample[1]
    return (maxX, maxY)

def min2D(samples):
    minX = float('inf')
    minY = float('inf')
    for sample in samples:
        if (minX > sample[0]):
            minX = sample[0]
        if (minY > sample[1]):
            minY = sample[1]
    return (minX, minY)

def median2D(samples):
    min = min2D(samples)
    max = max2D(samples)
    return (min[0] + max[0]/2.0, min[1] + max[1]/2.0)

def plot_samples(samples, mask, label):
    class_samples_x = []
    class_samples_y = []

    for sample in samples:
        class_samples_x.append(sample[0])
        class_samples_y.append(sample[1])

    plt.plot(class_samples_x, class_samples_y,
            mask, label = label)

def plot_datas(datas, masks):
    label_idx = 1
    for samples in datas:
        label = 'class ' + str(label_idx)
        plot_samples(samples, masks[label_idx-1], label)
        label_idx += 1

datas = load_datas('data.txt')
print_datas(datas)

means = [mean2D(datas[0]), mean2D(datas[1]), mean2D(datas[2])]

init_weights = [
    np.zeros(shape = 2),
    np.array([(means[0][0] + means[1][0])/2.0, (means[0][1] + means[1][1])/2.0]),
    np.array(means[0]),
    np.array(means[1]),
    np.array(means[2]),
    np.array(median2D(datas[0])),
    np.array(median2D(datas[1])),
    np.array(median2D(datas[2])),
    np.array(min2D(datas[0])),
    np.array(min2D(datas[1])),
    np.array(min2D(datas[2])),
    np.array(max2D(datas[0])),
    np.array(max2D(datas[1])),
    np.array(max2D(datas[2]))]

percep_classifier = PerceptronClassifier(init_weights[3],
                                        0.01, datas, 0, 1, 100 )
#percep_classifier = PerceptronClassifier(np.array([0.0, 0.0]),
#                                        0.0005, datas, 0, 1, 500 )
#percep_classifier.learn()

#relax_classifier = RelaxationClassifier(np.array([1.0, 1.5]), 0.001, datas, 0, 2, 1500, 0.1)
# learning rate 를 0.1로 올리면 weight_vector가 어느 한 점으로 수렴해버림
relax_classifier = RelaxationClassifier(init_weights[0], 0.01, datas, 0, 2, 100, 0.1)
relax_classifier.learn()

lms_classifier = LMSClassifier(init_weights[3], 0.01, datas, 0, 2, 100, 0.2)
#lms_classifier.learn()

plot_datas(datas, ['ro', 'bo', 'ko'])
plt.legend(loc="upper right")
plt.show()
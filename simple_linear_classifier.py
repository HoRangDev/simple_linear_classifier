import os
import numpy as np

class GradientDescent:
    def __init__(self, init_weight, learning_rate, datas, class_w1_idx, class_w2_idx):
        self.learning_rate = learning_rate
        self.class_1_data = datas[class_w1_idx]
        self.class_2_data = datas[class_w2_idx]
        self.class_w1_idx = class_w1_idx
        self.class_w2_idx = class_w2_idx
        self.weight = init_weight

    def get_missclasified_samples(self):
        missclasified = []
        for sample in self.class_1_data:
            classified = np.dot(self.weight, sample)
            if (classified < 0):
                # missclasified
                missclasified.append(sample)
        for sample in self.class_2_data:
            classified = np.dot(self.weight, sample)
            if (classified > 0):
                missclasified.append(sample)
        return missclasified

    def d_criterion_func(self, missclassified_samples):
        return None

    def learn(self):
        return None

class PerceptronClassifier(GradientDescent):
    def __init__(self, init_weight, learning_rate, datas, class_w1_idx, class_w2_idx, epochs):
        super().__init__(init_weight, learning_rate, datas, class_w1_idx, class_w2_idx)
        self.epochs = epochs

    def d_criterion_func(self, missclassified_samples):
        return np.sum(missclassified_samples)
    
    def learn(self):
        k = 0
        missclassified_samples = super().get_missclasified_samples()
        while (k < self.epochs):
            k += 1
            prog = self.learning_rate * self.d_criterion_func(missclassified_samples)
            self.weight = self.weight - prog
            missclassified_samples = super().get_missclasified_samples()
            print('current epochs: ' , k, ' weight: ', self.weight, ' missclasified: ', len(missclassified_samples))

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

datas = load_datas('data.txt')
print_datas(datas)

classifier = PerceptronClassifier(np.array([1.0, 1.5]), 0.001, datas, 0, 1, 50 )
classifier.learn()
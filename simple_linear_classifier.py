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

    def learn(self, print_message=False):
        k = 0
        ret = [[], [], []]
        while (k < self.epochs):
            missclassified_samples = self.get_missclasified_samples()
            if print_message:
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

def plot_learning_datas(ret, epoch, label, mark):
    epoch_space = np.linspace(1, epoch+1, epoch)
    plt.title(label + ' epoch-#missclasification')
    plt.xlabel('Epoch')
    plt.ylabel('#Misclassification')
    plt.plot(epoch_space, ret[2], mark, label = label)
    plt.show()

def plot_learning_rate_variation(init_weights, learning_rates, datas):
    missclassified_datas = [[], [], []]
    for learning_rate in learning_rates:
        percep_classifier = PerceptronClassifier(init_weights[2], learning_rate,
                                           datas, 0, 1, 
                                           100 )
    
        missclassified_nums = percep_classifier.learn()[2]
        missclassified_datas[0].append(missclassified_nums[-1])
    
        relax_classifier = RelaxationClassifier(init_weights[2], learning_rate,
                                              datas, 0, 2, 
                                              100, 0.1)
    
        missclassified_nums = relax_classifier.learn()[2]
        missclassified_datas[1].append(missclassified_nums[-1])
    
        lms_classifier = LMSClassifier(init_weights[2], learning_rate,
                                      datas, 0, 2,
                                     100, 0.1)
    
        missclassified_nums = lms_classifier.learn()[2]
        missclassified_datas[2].append(missclassified_nums[-1])
    
    plt.title('Learning rate variation')
    plt.xlabel('Learning rates')
    plt.ylabel('#Misclassified Samples')
    plt.plot(learning_rates, missclassified_datas[0], 'r-', label='Perceptron')
    plt.plot(learning_rates, missclassified_datas[1], 'k--', label='Relaxation')
    plt.plot(learning_rates, missclassified_datas[2], 'b-', label='LMS')
    plt.legend(loc="upper right")
    plt.show()

def plot_epoch_variation(init_weights, epochs, datas):
    missclassified_datas = [[], [], []]
    for epoch in epochs:
        percep_classifier = PerceptronClassifier(init_weights[2], 0.01,
                                           datas, 0, 1, 
                                           epoch )
    
        missclassified_nums = percep_classifier.learn()[2]
        missclassified_datas[0].append(missclassified_nums[-1])
    
        relax_classifier = RelaxationClassifier(init_weights[2], 0.01,
                                              datas, 0, 2, 
                                              epoch, 0.1)
    
        missclassified_nums = relax_classifier.learn()[2]
        missclassified_datas[1].append(missclassified_nums[-1])
    
        lms_classifier = LMSClassifier(init_weights[2], 0.01,
                                      datas, 0, 2,
                                     epoch, 0.1)
    
        missclassified_nums = lms_classifier.learn()[2]
        missclassified_datas[2].append(missclassified_nums[-1])
    
    plt.title('Epoch-#Missclaisifed variation')
    plt.xlabel('Epochs per Learning')
    plt.ylabel('#Misclassified Samples')
    plt.plot(epochs, missclassified_datas[0], 'b-', label='Perceptron')
    plt.plot(epochs, missclassified_datas[1], 'k--', label='Relaxation')
    plt.plot(epochs, missclassified_datas[2], 'g-', label='LMS')
    plt.legend(loc="upper right")
    plt.show()

def plot_percept_init_weight_var(init_weights, datas):
    percep_classifier = PerceptronClassifier(init_weights[3],
                                            0.01, datas, 0, 1, 100 )
    
    ret = percep_classifier.learn()
    #def plot_learning_datas(ret, epoch, _label, mark):
    plot_learning_datas(ret, 100, 'Percep #1', 'b-')
    
    percep_classifier = PerceptronClassifier(init_weights[5],
                                            0.01, datas, 0, 1, 100 )
    ret = percep_classifier.learn()
    plot_learning_datas(ret, 100, 'Percep #2 Meidan of class 1', 'b-')
    
    percep_classifier = PerceptronClassifier(init_weights[8],
                                            0.01, datas, 0, 1, 100 )
    ret = percep_classifier.learn()
    plot_learning_datas(ret, 100, 'Percep #3 Min of class 1', 'b-')
    
    percep_classifier = PerceptronClassifier(init_weights[11],
                                            0.01, datas, 0, 1, 100 )
    ret = percep_classifier.learn()
    plot_learning_datas(ret, 100, 'Percep #4 Max of class 1', 'b-')
    
    # Random initial_weight는 class 3의 median 으로 가정하였습니다.
    percep_classifier = PerceptronClassifier(init_weights[7],
                                            0.01, datas, 0, 1, 100 )
    ret = percep_classifier.learn()
    plot_learning_datas(ret, 100, 'Percep #5 Random', 'b-')

def plot_relaxation_init_weight_var(init_weights, datas):
    relax_classifier = RelaxationClassifier(init_weights[0], 0.01, datas, 0, 2, 100, 0.1)
    ret = relax_classifier.learn()
    plot_learning_datas(ret, 100, 'Relaxation #1 Zero', 'k-')

    relax_classifier = RelaxationClassifier(init_weights[5], 0.01, datas, 0, 2, 100, 0.1)
    ret = relax_classifier.learn()
    plot_learning_datas(ret, 100, 'Relaxation #2 Median of class 1', 'k-')

    relax_classifier = RelaxationClassifier(init_weights[2], 0.01, datas, 0, 2, 100, 0.1)
    ret = relax_classifier.learn()
    plot_learning_datas(ret, 100, 'Relaxation #3 Mean of class 1', 'k-')

    # Random initial_weight는 class 2의 median 으로 가정하였습니다.
    relax_classifier = RelaxationClassifier(init_weights[3], 0.01, datas, 0, 2, 100, 0.1)
    ret = relax_classifier.learn()
    plot_learning_datas(ret, 100, 'Relaxation #4 Random init weight', 'k-')

def plot_lms_init_weight_var(init_weights, datas):
    lms_classifier = LMSClassifier(init_weights[0], 0.01, datas, 0, 2, 100, 0.1)
    ret = lms_classifier.learn()
    plot_learning_datas(ret, 100, 'LMS #1 Zero', 'g-')

    lms_classifier = LMSClassifier(init_weights[5], 0.01, datas, 0, 2, 100, 0.1)
    ret = lms_classifier.learn()
    plot_learning_datas(ret, 100, 'LMS #2 Median of class 1', 'g-')

    lms_classifier = LMSClassifier(init_weights[2], 0.01, datas, 0, 2, 100, 0.1)
    ret = lms_classifier.learn()
    plot_learning_datas(ret, 100, 'LMS #3 Mean of class 1', 'g-')

    # Random initial_weight는 class 2의 median 으로 가정하였습니다.
    lms_classifier = LMSClassifier(init_weights[3], 0.01, datas, 0, 2, 100, 0.1)
    ret = lms_classifier.learn()
    plot_learning_datas(ret, 100, 'LMS #4 Random init weight', 'g-')

#############################################################################################################

# Initialize Default values
datas = load_datas('data.txt')
print_datas(datas)
learning_rates = np.linspace(0.0001, 1, 200)
epochs = np.linspace(1, 500, 50)

means = [mean2D(datas[0]), mean2D(datas[1]), mean2D(datas[2])]

init_weights = [
    np.zeros(shape = 2), #0
    np.array([(means[0][0] + means[1][0])/2.0, (means[0][1] + means[1][1])/2.0]), #1
    np.array(means[0]), # 2
    np.array(means[1]), # 3
    np.array(means[2]), # 4
    np.array(median2D(datas[0])), # 5
    np.array(median2D(datas[1])), # 6
    np.array(median2D(datas[2])), # 7
    np.array(min2D(datas[0])), # 8
    np.array(min2D(datas[1])), # 9
    np.array(min2D(datas[2])), # 10
    np.array(max2D(datas[0])), # 11
    np.array(max2D(datas[1])), # 12
    np.array(max2D(datas[2]))] # 13
#################################################################################

# Data ploting
#   plt.title('Training datas')
#   plot_datas(datas, ['ro', 'bo', 'ko'])
#   plt.legend(loc="upper right")
#   plt.show()

# perceptron, relaxation, windrow

# Plotting Learnin rate variation
#   learning_rate \in [0.0001, 1)
#   bias(margin) = 0.1
#   init_weights = class 1's sample mean
#   epochs = 100
plot_learning_rate_variation(init_weights, learning_rates, datas)

# Plotting Epoch variation
#   learning_rate = 0.01
#   bias(margin) = 0.1
#   init_weights = class 1's sample mean
#   epochs \in [1, 500]
plot_epoch_variation(init_weights, epochs, datas)

# Plotting Initial weight variation of perceptron classifier
#   learning rate = 0.01
#   epochs = 100    
plot_percept_init_weight_var(init_weights, datas)

# Plotting Initial weight variation of relaxation classifier
#   learning rate = 0.01
#   epochs = 100
#   margin = 0.1
plot_relaxation_init_weight_var(init_weights, datas)

# Plotting Initial weight variation of relaxation classifier
#   learning rate = 0.01
#   epochs = 100
#   margin = 0.1
plot_lms_init_weight_var(init_weights, datas)

#relax_classifier = RelaxationClassifier(np.array([1.0, 1.5]), 0.001, datas, 0, 2, 1500, 0.1)
# learning rate 를 0.1로 올리면 weight_vector가 어느 한 점으로 수렴해버림
#relax_classifier = RelaxationClassifier(init_weights[0], 0.01, datas, 0, 2, 100, 0.1)
#relax_classifier.learn()

#lms_classifier = LMSClassifier(init_weights[3], 0.01, datas, 0, 2, 100, 0.2)
#lms_classifier.learn()

import sys
import numpy as np

class Net:
    def __init__(self, num_of_neurons): 
        #num_of_neurons is a list of number of neurons in every layer
        #self.num_of_layers = len(num_of_neurons)
        self.num_of_neurons = num_of_neurons
        self.weights = []

    def forward(self, input): #input is a list of input integers
        neuron_outputs = input[:]

        for layer in range(len(self.num_of_neurons)-1):
            new_neuron_outputs = []
            layer1_len = self.num_of_neurons[layer]
            layer2_len = self.num_of_neurons[layer+1]

            for neuron_layer2 in range(layer2_len):
                output = 0

                for neuron_layer1 in range(layer1_len):                            
                    output +=  float(neuron_outputs[neuron_layer1]) * \
                        self.weights[layer][neuron_layer1*layer2_len + neuron_layer2]
                    
                output += self.weights[layer][-neuron_layer2 -1] #bias

                if layer != (len(self.num_of_neurons)-2):
                    new_neuron_outputs.append(sigmoid(output))
                else:
                    return output

            neuron_outputs = new_neuron_outputs[:]

        
    
    def initialize_weights(self):
        for layer in range(len(self.num_of_neurons) - 1): 
            self.weights.append(np.random.normal(0, 0.01, self.num_of_neurons[layer]*self.num_of_neurons[layer+1]) \
                                + self.num_of_neurons[layer+1])              
            #append (weights between layers, bias in backward order)
    
def calculate_error(real_y, predicted_y):
    return (float(real_y) - predicted_y)**2
    

def train(num_of_iterations, population, elitism, p, K, data):
    population_size = len(population)
    for iter in range(num_of_iterations):    
        for chromosome in population.keys():
            total_error = 0

            for row in data:
                predicted_y = chromosome.forward(row[:-1])
                real_y = row[-1]
                total_error += calculate_error(real_y, predicted_y)

            population[chromosome] = len(data) / total_error

        #print the smallest error
        if (iter+1) % (num_of_iterations/5) == 0:
            best_fitness = max(population.values())
            print("[Train error @{}]: {}".format(iter+1, round(1 / best_fitness, 6)))


        
        #change population
        new_population = []

        #elitism
        sorted_keys = sorted(population, key=population.get, reverse=True)
        elite_chromosomes = {}
        for i in range(elitism):
            elite_chromosomes[sorted_keys[i]] = 0

        #selection
        parents = list(population.keys())
        sum_of_fitness = sum(population.values())

        for i in range(population_size - elitism):
            #1st parent
            rand_num = np.random.uniform(0, sum_of_fitness)
            cnt = 0
            while rand_num > 0:
                parent1 = parents[cnt]
                rand_num -= population[parents[cnt]]
                cnt += 1

            #2nd parent
            new_parents = [x for x in parents if x != parent1]
            #new_parents = [x for x in parents if not np.array_equal(x.weights, parent1.weights) ]
            new_sum_of_fitness = sum_of_fitness - population[parent1]

            rand_num = np.random.uniform(0, new_sum_of_fitness)
            cnt = 0
            while rand_num > 0:
                parent2 = parents[cnt]
                rand_num -= population[new_parents[cnt]]
                cnt += 1

            child = Net(parent1.num_of_neurons)

            child_weights = []

            for row1, row2 in zip(parent1.weights, parent2.weights):
                new_row = [(w1 + w2) / 2 for w1, w2 in zip(row1, row2)]
                child_weights.append(new_row)

            child.weights = child_weights[:]

            new_population.append(child)

        #mutation
        for k in range(len(new_population)):
            for i in range(len(new_population[k].weights)):
                for j in range(len(new_population[k].weights[i])):
                    if np.random.rand() < p:
                        new_population[k].weights[i][j] += np.random.normal(0, K)

        #join
        population = {key: 0 for key in new_population}
        population.update(elite_chromosomes)



    best_fitness = -1
    for chromosome in population.keys():
        total_error = 0

        for row in data:
            predicted_y = chromosome.forward(row[:-1])
            real_y = row[-1]
            total_error += calculate_error(real_y, predicted_y)

        population[chromosome] = 1 / total_error

    for key in population.keys():
        if population[key] > best_fitness:
            best_fitness = population[key]
            best_chromosome = key


    return best_chromosome

def test(nn, data):
    test_MSE = 0

    for row in data:
        predicted_y = nn.forward(row[:-1])
        test_MSE += calculate_error(predicted_y, float(row[-1]))

    test_MSE /= len(test_data)

    print("[Test error]: {}".format(round(test_MSE, 6)))
    return

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )

def get_data(file_path):

    data = []
    with open(file_path, "r") as file:
        for i, row in enumerate(file):
            if i!=0:
                list1 = row.split(",")
                data.append(list1)

    return data

#main

dict_of_args = {}

for i in range (1,len(sys.argv),2):
    dict_of_args[sys.argv[i]] = sys.argv[i+1]

train_data = get_data(dict_of_args["--train"])
test_data = get_data(dict_of_args["--test"])

net_name = dict_of_args["--nn"]
num_of_iterations = int(dict_of_args["--iter"])
population_size = int(dict_of_args["--popsize"])
elitism = int(dict_of_args["--elitism"])
p = float(dict_of_args["--p"]) #possibility of mutation
K = float(dict_of_args["--K"]) #std dev of Gaussian noise of mutation

num_of_inputs = len(train_data[0]) - 1

if net_name == "5s":
    nn_params = [num_of_inputs, 5, 1]

elif net_name == "20s":
    nn_params = [num_of_inputs, 20, 1]

elif net_name == "5s5s":
    nn_params = [num_of_inputs, 5, 5, 1]

#nn.initialize_weights(net_name)

population = {} # {nn : fitness}

for i in range (population_size):
    nn = Net(nn_params)
    nn.initialize_weights()
    population[nn] = 0

best_nn = train(num_of_iterations, population, elitism, p, K, train_data)
test(best_nn, test_data)
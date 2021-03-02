import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def getRandomWalks(num_tsets, sequences_per_tset, num_states, start_position, seed=12):
    '''
    :param num_sequences:
    :param nodes:
    :param start_position:
    :return: list of random walks for training
    '''
    if seed:
        np.random.seed(seed)
    training_sets = [] #List of training Sequences
    for p in range(num_tsets):
        training_sequences = [] #List of sequences
        for i in range(sequences_per_tset):
            position = start_position
            sequence = []
            while position != 0 and position != num_states-1:
                state = [0] * num_states
                state[position] = 1
                sequence.append(state[1:-1])
                if np.random.uniform() > .5:
                    position -= 1
                else:
                    position += 1
            training_sequences.append(sequence)
        training_sets.append(training_sequences)
    #[list of training sets[list of sequences[[0,0,0,1,0]]]]
    return training_sets

def learn_weights(sequence, _lambda, alpha, p):
    '''
    Perform weight updates from Sutton(4) on given training sequence
    '''
    num_steps = len(sequence)
    num_states = len(sequence[0])
    #We popped off reward states and sequences so this will be the state before entering the terminal state
    z = sequence[-1][-1] #Get the reward of the sequence
    delta_w = [0]*num_states
    lambda_seq = [1]

    for step in range(num_steps): #For each training example
        # print("Iteration:")
        steps_up_to_now = sequence[0:step+1]
        # print("training sequence")
        # print(steps_up_to_now)
        # print("labmda sequence")
        # print(lambda_seq)
        if step == num_steps-1:
            #Entering terminal State
            # print("terminal state info")
            # print(sequence[-1])
            # print(np.dot(p, sequence[-1]))
            delta_p = z - np.dot(p, sequence[-1])
        else:
            #Non Terminal State
            delta_p = np.dot(p, sequence[step + 1]) - np.dot(p, sequence[step])
        temp = []
        for i in range(len(steps_up_to_now)):
            temp += [np.multiply(steps_up_to_now[i], lambda_seq[i])]
        delta_w = np.add(delta_w, np.multiply(alpha,  np.multiply(delta_p, np.sum(temp, axis = 0))))
        # print("delta w")
        # print(delta_w)
        lambda_seq = np.append(np.multiply(lambda_seq, _lambda), 1)
    return delta_w

def figureThree():
    actual_values = [1/6, 2/6, 3/6, 4/6, 5/6]
    alpha = .01
    epislon = .00001
    RMSE = []
    training_sequences = getRandomWalks(100, 10, 7, 3)
    _lambdas = [0,.1,.3,.5,.7,.9, 1]
    for _lambda in _lambdas:
        weights = [.5] * 5
        for tset in range(100): #Iterate through training set
            while True: #convergence of updates to weights for sequnce
                deltas = [0] * 5
                #Iterate over the same training sequence until we get convergence
                for seq in range(10): #Accumulate delta w's
                    deltas = np.add(deltas, learn_weights(training_sequences[tset][seq], _lambda, alpha, weights))
                weights = np.add(weights, deltas)
                if np.sum(np.absolute(deltas)) <= epislon:
                    break
        # print("lambda: " + str(_lambda))
        # print("-----------------------------------------------")
        # print("weights")
        # print(weights)
        # print("deltas")
        # print(deltas)
        # print("-----------------------------------------------")
        RMSE.append(mean_squared_error(actual_values, weights, squared=False))
    plt.plot(_lambdas, RMSE)
    plt.ylabel('RMSE', fontsize=15)
    plt.xlabel('lambda', fontsize=15)
    plt.xticks(_lambdas,
               fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

def figureFour():
    actual_values = [1/6, 2/6, 3/6, 4/6, 5/6]
    alphas_fig4 = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9, 1]
    lambdas_fig4 = [0, 0.3, 0.8, 1.0]
    colors_fig4 = ['#da008a', '#7400e3', '#009bdf', '#c5c23d']
    training_sequences = getRandomWalks(100, 10, 7, 3)
    RMSE_matrix = []
    for m in range(len(lambdas_fig4)):
        RMSE_matrix.append([0]*len(alphas_fig4))
    for i in range(len(lambdas_fig4)):
        for j in range(len(alphas_fig4)):
            for tset in range(100):
                weights = [.5] * 5
                for seq in range(10):
                    #Weights are updated after each training sequence
                    seq_len = len(training_sequences[tset][seq])
                    if seq_len>8:
                        continue
                    weights = np.add(weights, learn_weights(training_sequences[tset][seq], lambdas_fig4[i], alphas_fig4[j], weights))
                RMSE_matrix[i][j] += mean_squared_error(actual_values, weights, squared=False)
    RMSE_matrix = np.divide(RMSE_matrix, 100)
    for p in range(len(lambdas_fig4)):
        plt.plot(alphas_fig4, RMSE_matrix[p], label='lambda = ' + str(lambdas_fig4[p]), color=colors_fig4[p])
    plt.xlabel('alpha', fontsize=15)
    plt.ylabel('RMSE', fontsize=15)
    plt.legend(loc='best', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

def figureFive():
    actual_values = [1/6, 2/6, 3/6, 4/6, 5/6]
    training_sequences = getRandomWalks(100, 10, 7, 3)
    alphas_fig5 = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    lambdas_fig5 = [0, 0.3, 0.8, 1.0]
    RMSE_matrix = []
    for m in range(len(lambdas_fig5)):
        RMSE_matrix.append([0]*len(alphas_fig5))
    for i in range(len(lambdas_fig5)):
        for j in range(len(alphas_fig5)):
            for tset in range(100):
                weights = [.5] * 5
                for seq in range(10):
                    weights = np.add(weights, learn_weights(training_sequences[tset][seq], lambdas_fig5[i], alphas_fig5[j], weights))
                RMSE_matrix[i][j] += mean_squared_error(actual_values, weights, squared=False)
    RMSE_matrix = np.divide(RMSE_matrix, 100)
    RMSE_plot = []
    for row in RMSE_matrix:
        RMSE_plot.append(min(row))
    plt.plot(lambdas_fig5, RMSE_plot)
    plt.ylabel('RMSE using the best alpha', fontsize=15)
    plt.xlabel('lambda', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

if __name__ == "__main__":
    figureThree()
    figureFour()
    figureFive()

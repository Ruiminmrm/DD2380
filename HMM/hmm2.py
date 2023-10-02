class HMM():
    def __init__(self):
        self.pi = [] # initial state probability 
        self.A = []  # transition matrix
        self.B = []  # emission matrix
        self.observation = [] # observation matrix 
        
    def convert_string_to_matrix(self):
        a_list = input().split()
        b_list = input().split()
        pi_list = input().split()
        observation_list = input().split()
        self.A  = self.convert_list_to_matrix(a_list)
        self.B = self.convert_list_to_matrix(b_list)
        self.pi = self.convert_list_to_matrix(pi_list)
        self.observation = self.convert_observation_to_vector(observation_list)
        
    def convert_list_to_matrix(self, ls):
        '''
        input : list [rows, colums, 'str'.....'str']
        output : matrix [[]...[]]
        '''
        rows = int(ls[0])
        colums = int(ls[1])
        matrix = []
        index = 2
        for row in range(rows):
            row = []
            for colum in range(colums):
                row.append(float(ls[index]))
                index += 1
            matrix.append(row)
        return matrix
    
    def convert_observation_to_vector(self, ls):
        '''
        input : list[rows, 'str'...'str']
        output : vector [...]
        '''
        size = int(ls[0])
        vector = [int(ele) for ele in ls[1:size+1]]
        return vector
    
    def viterbi(self): 
        T = len(self.observation)# len of observation state
        N = len(self.A)          # number of hidden state
        
        # initialize matrix delta and path matrix 
        delta = [[0.0] * N for _ in range(T)]
        path = [[0] * N for _ in range(T)]
        
        # initialize first step delta value
        for i in range(N):
            delta[0][i] = self.pi[0][i] * self.B[i][self.observation[0]]
            # initial states for all path are 0
            path[0][i] = 0 
            
        # calculate the probabilities for the remaining time steps
        for t in range(1, T):
            for i in range(N):
                max_prob = 0.0
                prev_state = 0
                for j in range(N):
                    prob = delta[t-1][j] * self.A[j][i]
                    if prob > max_prob:
                        max_prob = prob
                        prev_state = j
                delta[t][i] = max_prob * self.B[i][self.observation[t]]
                path[t][i] = prev_state
        
        # find the most likely hidden state at the final moment
        best_path = [0] * T
        max_prob = 0.0
        best_last_state = 0
        for i in  range(N):
            if delta[T - 1][i] > max_prob:
                max_prob = delta[T - 1][i]
                best_last_state = i
        best_path[T - 1] = best_last_state
        
        # backtrack to find the most likely path
        for t in range(T - 2, -1, -1):
            best_path[t] = path[t + 1][best_path[t + 1]]
        best_path = ' '.join(map(str, best_path))
        return best_path

if __name__ == '__main__':
    hmm = HMM()
    hmm.convert_string_to_matrix()
    prob = hmm.viterbi()
    print(prob)
    
'''
A = "4 4 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.8 0.1 0.1 0.0 "
B = "4 4 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.1 0.0 0.0 0.9 "
pi = "1 4 1.0 0.0 0.0 0.0 "
observation = "4 1 1 2 2 "
hmm = HMM()
res = hmm.convert_string_to_matrix(A,B,pi,observation)
print(hmm.viterbi())
'''
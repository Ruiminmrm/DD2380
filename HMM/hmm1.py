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
        # print(observation_list)
        self.A  = self.convert_list_to_matrix(a_list)
        self.B = self.convert_list_to_matrix(b_list)
        self.pi = self.convert_list_to_matrix(pi_list)
        self.observation = self.convert_observation_to_vector(observation_list)
        # print(self.observation)
        
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
    
    def forward(self): 
        T = len(self.observation)# len of observation state
        N = len(self.A)          # number of hidden state
        
        # initialize forward probability matrix
        alpha = [[0 for x in range(N)] for y in range(T)]
        
        # initialize first step forward probability
        for i in range(N):
            alpha[0][i] = self.pi[0][i] * self.B[i][self.observation[0]]
            
        # calculate the probabilities for the remaining time steps
        for t in range(1, T):
            for i in range(N):
                alpha[t][i] = sum(alpha[t][i] + alpha[t-1][j]*self.A[j][i] for j in range(N)) * self.B[i][self.observation[t]]
                
        # calculate the sum of the probabilities
        probability = 0
        for i in range(N):
            probability = sum(alpha[T - 1])
        return probability
    
if __name__ == '__main__':
    hmm = HMM()
    hmm.convert_string_to_matrix()
    prob = hmm.forward()
    print(prob)

'''   
A = "4 4 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.8 0.1 0.1 0.0 "
B = "4 4 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.1 0.0 0.0 0.9 "
pi = "1 4 1.0 0.0 0.0 0.0 "
obeservation = "8 0 1 2 3 0 1 2 3 "
hmm = HMM()
res = hmm.convert_string_to_matrix(A,B,pi,obeservation)
print(hmm.forward())
''' 
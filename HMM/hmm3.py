import math

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
        
    def matrix_multiplication_recursive(self, matrix_A, matrix_B):
        #initialize result matrix with 0s
        result = [[0 for j in range(len(matrix_B[0]))] for i in range(len(matrix_A))]
        for i in range(len(matrix_A)):
            for j in range(len(matrix_B[0])):
                result[i][j] = 0
                for x in range(len(matrix_A[0])):           
                    result[i][j] += (matrix_A[i][x] * matrix_B[x][j])
        return result 
    
    '''
    scale`的目的是为了数值稳定性和减小数值溢出或下溢的可能性。
    这是因为在计算HMM的前向概率时,涉及多个概率相乘,如果这些概率很小,连续相乘可能导致数值下溢,而如果这些概率很大,连续相乘可能导致数值溢出。
    通过引入缩放因子scale,可以将概率值调整到一个更合适的范围内,从而避免数值问题。
    '''
    def forward(self):
        T, N = len(self.observation), len(self.A)# len of observation state and number of hidden state
        
        # initialize forward probability matrix and scaling factors 
        alpha = [[0 for _ in range(N)] for _ in range(T)]
        scale = [0 for _ in range(T)]
        
        # initialize first step forward probability
        for i in range(N):
            alpha[0][i] = self.pi[0][i] * self.B[i][self.observation[0]]
            scale[0] += alpha[0][i]
        
        # Scaling at time step 0
        scale[0] = 1 / scale[0]
        for i in range(N):
            alpha[0][i] *= scale[0]
            
        # forward algorithm with scaling 
        for t in range(1, T):
            for i in range(N):
                alpha[t][i] = sum(alpha[t - 1][j] * self.A[j][i] for j in range(N)) * self.B[i][self.observation[t]]
                scale[t] += alpha[t][i]
                
            # scaling at time step t
            scale[t] = 1 / scale[t]
            for i in range(N):
                alpha[t][i] *= scale[t]
        
        # calculate the final probability of the observation sequence 
        # probability = 1 / (scale[T - 1] * scale[T - 2] * ... * scale[1])
        return [alpha, scale]     
    
    def backward(self, scale):
        T, N = len(self.observation), len(self.A)
        beta = [[0 for _ in range(N)] for _ in range(T)]
        
        # compute beta at time step T, scale using same value as for alpha
        for i in range(N):
            beta[T - 1][i] = scale[T - 1]
        
        # backward algorithm
        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta[t][i] = sum(self.A[i][j] * self.B[j][self.observation[t + 1]] * beta[t + 1][j] for j in range(N))
                beta[t][i] *= scale[t]
        return beta
    
    # calculates gamma and di_gamma
    def gamma(self, alpha, beta):
        T, N = len(self.observation), len(self.A)
        gamma = [[0.0] * N for _ in range(T)]
        di_gamma = [[[0.0] * N for _ in range(N)] for _ in range(T - 1)]
        
        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    di_gamma[t][i][j] = (alpha[t][i] * self.A[i][j] * self.B[j][self.observation[t + 1]] * beta[t + 1][j])
                    gamma[t][i] += di_gamma[t][i][j]
        
        for i in range(N):
            gamma[T - 1][i] = alpha[T - 1][i]
            
        return [gamma, di_gamma]
        
    def re_estimate(self, gamma, di_gamma):
        T, N, M = len(self.observation), len(self.A), len(self.B[0])
        # estimate pi
        self.pi[0] = gamma[0]
        
        # esitmate A
        for i in range(N):
            denom = sum(gamma[t][i] for t in range(T - 1))
            for j in range(N):
                numer = sum(di_gamma[t][i][j] for t in range(T - 1))
                self.A[i][j] = numer / denom
        
        # estimate B
        for i in range(N):
            denom = sum(gamma[t][i] for t in range(T))
            for j in range(M):
                numer = sum(gamma[t][i] for t in range(T) if self.observation[t] == j)
                self.B[i][j] = numer / denom
                           
    def log_prob(self, scale):
        return -sum(math.log(scale[t]) for t in range(len(self.observation)))
  
    
    def baum_welch(self, max_iterations):
        self.convert_string_to_matrix()
        
        old_log_prob = -math.inf
        for i in range(max_iterations):
            alpha, scale = self.forward()
            beta = self.backward(scale)
            gamma, di_gamma = self.gamma(alpha, beta)
            self.re_estimate(gamma, di_gamma)
            
            log_prob = self.log_prob(scale)
            if log_prob <= old_log_prob:
                break
            old_log_prob = log_prob
        
        print(str(len(self.A)) + ' ' + str(len(self.A[0])) + ' ')
        for row in self.A:
            print(" ".join(map(lambda x: f"{x:.6f}", row)))
        print('\n' + str(len(self.B)) + ' ' + str(len(self.B[0])) + ' ')
        for row in self.B:
            print(" ".join(map(lambda x: f"{x:.6f}", row)))
        
if __name__ == '__main__':
    hmm = HMM()
    hmm.baum_welch(100)
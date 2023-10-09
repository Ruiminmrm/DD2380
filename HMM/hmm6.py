import math

class HMM():
    def __init__(self):
        self.pi = [] # initial state probability 
        self.A = []  # transition matrix
        self.B = []  # emission matrix
        self.observation = [] # observation matrix 
        
    def convert_string_to_matrix(self,A, B,pi,observation):
        a_list = A.split()
        b_list = B.split()
        pi_list = pi.split()
        observation_list = observation.split()
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
    
    def baum_welch(self, max_iterations, A,B,pi,observation):
        self.convert_string_to_matrix(A,B,pi,observation)
        old_log_prob = -math.inf
        for _ in range(max_iterations):
            alpha, scale = self.forward()
            beta = self.backward(scale)
            gamma, di_gamma = self.gamma(alpha, beta)
            self.re_estimate(gamma, di_gamma)
            
            log_prob = self.log_prob(scale)
            if log_prob <= old_log_prob:
                break
            old_log_prob = log_prob
        '''
        print(str(len(self.A)) + ' ' + str(len(self.A[0])) + ' ')
        for row in self.A:
            print(" ".join(map(lambda x: f"{x:.6f}", row)))
        print('\n' + str(len(self.B)) + ' ' + str(len(self.B[0])) + ' ')
        for row in self.B:
            print(" ".join(map(lambda x: f"{x:.6f}", row)))
        print('\n' + str(len(self.pi)) + ' ' + str(len(self.pi[0])) + ' ')
        for row in self.pi:
            print(" ".join(map(lambda x: f"{x:.6f}", row)))
        '''
        return self.A, self.B, self.pi  
    
def diffsum (A,B,pi):
    A_onedim = [element for sublist in A for element in sublist]
    B_onedim = [element for sublist in B for element in sublist]
    pi_onedim = pi[0]
    
    A_converge =  [0.7, 0.05,0.25,0.1,0.8,0.1,0.2,0.3,0.5]
    B_converge = [0.7,0.2,0.1,0,0.1,0.4,0.3,0.2,0,0.1,0.2,0.7]
    pi_converge = [1,0,0]
    
    diff_a = difference(A_onedim, A_converge, 0)
    diff_b = difference(B_onedim, B_converge, 0)
    diff_pi = difference(pi_onedim, pi_converge, 0)
    return diff_a +diff_b + diff_pi

def difference(a,b,diff):
    for i in range(len(a)):
        diff += abs(a[i] - b[i])
    return diff
if __name__ == '__main__':
    A = '             3 3 1 0 0 0 1 0 0 0 1'
    B = '3 4 0.5 0.2 0.11 0.19 0.22 0.28 0.23 0.27 0.19 0.21 0.15 0.45'
    pi = '1 3 0.3 0.2 0.5 '
    observation = '1000 0 3 3 2 1 0 2 3 0 0 2 3 0 0 2 3 0 0 0 0 1 2 2 3 1 3 0 2 3 1 2 3 1 3 3 1 2 0 3 1 2 2 2 1 1 1 0 2 0 3 2 3 2 1 1 2 3 1 2 1 2 1 2 1 1 3 3 1 1 2 3 0 0 1 1 2 3 1 1 0 1 1 1 1 1 3 3 1 1 3 0 0 3 1 3 1 3 3 2 1 3 3 2 3 3 3 1 1 1 1 2 1 3 3 3 0 1 0 0 3 3 2 2 3 3 3 2 1 0 0 3 3 1 0 2 2 2 0 0 2 3 1 1 1 2 2 3 3 1 1 0 0 1 2 0 0 2 0 1 1 0 0 0 0 3 3 1 2 0 0 2 3 2 2 0 1 1 3 1 1 0 3 3 0 3 3 3 2 3 2 3 1 2 1 2 0 0 0 0 3 0 3 3 3 3 1 1 3 2 1 2 0 3 3 3 3 3 3 0 0 2 3 2 1 1 1 3 2 2 2 1 2 1 1 2 3 1 2 3 1 1 1 3 3 2 2 2 1 0 2 2 1 0 3 2 2 3 1 0 1 2 2 2 0 3 1 2 0 3 0 0 3 1 2 2 3 1 1 0 2 2 1 3 2 3 1 0 0 0 2 1 3 3 0 1 0 3 0 2 0 0 3 2 3 3 3 2 3 2 3 1 3 3 0 1 0 2 2 1 2 3 3 1 3 2 3 3 3 1 2 1 3 0 1 2 0 1 3 3 3 3 2 1 1 1 1 3 3 1 1 3 3 2 0 1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 2 0 2 0 3 1 3 1 3 2 0 0 0 0 0 1 2 3 0 1 1 0 2 3 2 1 0 2 1 2 2 2 1 1 2 2 1 3 3 3 3 1 3 1 0 2 1 2 1 3 1 1 3 0 0 1 0 1 3 3 2 2 1 2 3 2 1 2 1 1 1 0 0 0 2 2 2 1 3 3 1 1 2 2 1 1 2 3 1 2 1 0 0 3 2 1 3 2 2 3 0 0 3 3 0 3 0 0 1 0 0 2 3 1 0 0 0 1 2 2 3 2 3 1 0 2 2 3 3 1 1 2 1 1 1 3 2 2 3 2 0 3 3 0 0 0 0 0 0 0 0 2 2 3 3 3 2 3 2 1 3 0 2 1 2 0 0 0 2 3 2 2 2 1 3 3 2 1 2 0 0 3 2 1 2 3 1 0 1 3 3 3 2 2 0 0 0 2 2 3 1 1 3 0 0 3 0 0 3 0 1 3 0 3 0 0 0 0 0 0 0 0 3 2 3 1 2 3 0 3 3 2 0 3 2 1 1 0 1 0 0 0 0 2 3 3 3 1 1 3 1 2 3 0 3 3 3 0 0 1 1 0 3 2 2 3 1 1 3 1 1 3 2 0 2 0 0 2 0 0 0 1 0 1 1 0 0 0 0 0 2 2 3 3 1 3 3 3 1 0 2 3 3 3 0 1 1 1 1 0 0 3 3 3 2 3 2 3 3 2 3 3 3 3 3 2 2 1 3 1 0 1 1 1 0 3 1 0 0 3 3 2 2 2 3 3 1 1 1 2 1 3 2 1 1 2 3 2 2 1 2 2 1 1 2 1 0 0 1 2 1 2 1 1 2 0 0 3 2 3 2 1 1 3 2 3 3 3 0 2 0 3 3 1 1 1 2 2 0 0 0 0 0 1 1 1 3 2 3 2 0 0 3 2 3 1 2 0 3 2 1 2 1 0 1 1 1 2 2 2 2 2 3 3 2 3 2 1 3 3 2 2 2 1 0 1 3 2 1 0 3 2 2 1 1 3 3 3 3 1 0 3 3 3 2 3 3 2 3 0 1 0 1 2 2 1 2 1 2 3 3 3 0 1 2 2 3 1 1 0 3 2 2 1 1 0 0 1 0 0 0 1 1 0 0 0 0 2 0 0 0 3 3 0 0 3 1 2 2 1 0 0 2 2 3 0 3 0 3 2 2 3 3 2 0 2 3 0 0 3 1 0 0 1 2 0 0 3 3 1 1 3 2 3 1 3 3 1 3 3 3 2 3 1 0 1 3 2 1 2 3 3 0 2 0 0 1 0 3 2 0 2 3 1 3 1 1 2 1 1 2 2 1 1 3 2 2 0 0 1 1 0 0 0 2 2 0 0 3 2 2 1 2 3 3 0 0 2 0 0 0 3 3 3 2 0 1 0 1 1 3 '
    hmm = HMM()
    max_iteration = 1000
    iteration = 500
    converge_iter= 0

    min_diff= math.inf
    while iteration <= max_iteration:
        result_a, result_b,result_pi = hmm.baum_welch(iteration, A,B,pi,observation)
        diff = diffsum(result_a, result_b,result_pi)
        print('iteration : ', iteration, '+ diff :',diff)
        if diff < min_diff:
            min_diff = diff
            converge_iter = iteration
            converge_a = result_a
            converge_b = result_b
            converge_pi = result_pi
            print(converge_iter)
        iteration = iteration + 100
        
    print('iteration', converge_iter)
    print(str(len(converge_a)) + ' ' + str(len(converge_a[0])) + ' ')
    for row in converge_a:
        print(" ".join(map(lambda x: f"{x:.6f}", row)))
    print('\n' + str(len(converge_b)) + ' ' + str(len(converge_b[0])) + ' ')
    for row in converge_b:
        print(" ".join(map(lambda x: f"{x:.6f}", row)))
    print('\n' + str(len(converge_pi)) + ' ' + str(len(converge_pi[0])) + ' ')
    for row in converge_pi:
        print(" ".join(map(lambda x: f"{x:.6f}", row)))
        
        

    
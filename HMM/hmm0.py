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
        self.A  = self.convert_list_to_matrix(a_list)
        self.B = self.convert_list_to_matrix(b_list)
        self.pi = self.convert_list_to_matrix(pi_list)
    
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
    
    def matrix_multiplication_recursive(self, matrix_A, matrix_B):
        #initialize result matrix with 0s
        result = [[0 for j in range(len(matrix_B[0]))] for i in range(len(matrix_A))]
        for i in range(len(matrix_A)):
            for j in range(len(matrix_B[0])):
                result[i][j] = 0
                for x in range(len(matrix_A[0])):           
                    result[i][j] += (matrix_A[i][x] * matrix_B[x][j])
        return result 
    
    def next_emission(self, emm, matrix_A, matrix_B):
        matrix = self.matrix_multiplication_recursive(emm, matrix_A)
        matrix = self.matrix_multiplication_recursive(matrix ,matrix_B)
        rows = len(matrix)
        columns = len(matrix[0])
        string = str(rows) + " " + str(columns) + " "
        for i in range(rows):
            row_str = ' '.join(map(str, matrix[i]))
            string = string + row_str
        return string 
        
if __name__ == '__main__':
    hmm = HMM()
    hmm.convert_string_to_matrix()
    res = hmm.next_emission(hmm.pi, hmm.A, hmm.B)
    print(res)

import gevent
import numpy as np
import itertools
import queue

np.set_printoptions(suppress=True)

MAX = 100000           #大M
MIN = 0


class AssignSlover(object):
    '''

    '''

    def __init__(self):
        self.coe_matrix = None
        self.std_coe_matrix = None
        self.std_coe_matrix2 = None
        self.dimx = 0
        self.dimy = 0
        self.flag = []
        self.step = 1
        self.transfer_count = 1
        self.best_solve = []
        self.best_result = MAX*10
        self.process_queue = queue.Queue()
        self.all_mid_matrix = []


    def __start__(self, type = 'min'):
        if type not in ("min","max"):
            raise TypeError("The type must be 'max' or 'min'")
            return
        if type == 'max':
            self.best_result = 0
        self.StdCoeMatrix(type = type)
        self.ViolentSolve(type = type)
        self.SetFlag()
        gevent.spawn(self.Solver, self.MatrixConvert(self.std_coe_matrix))
        print("===解题步骤如下===")
        while 0 in self.flag or not self.process_queue.empty():
            gevent.sleep(0.1)
        else:
            print("\n-->经检验，已经遍历得到全部最优解")
            if type == 'min':
                print("-->最优解为%f" % (self.best_result - abs(self.dimx - self.dimy) * MAX))
            else:
                print("-->最优解为%f" % (self.best_result - abs(self.dimx - self.dimy) * MIN))
            print("-->最优的指派矩阵为：")
            for obj in self.best_solve:
                print(obj)


    def SetCoeMatrix(self, matrix):
        '''set Coefficient Matrix and its dimensions'''
        self.coe_matrix = matrix
        self.dimx, self.dimy = matrix.shape
        if self.dimx == self.dimy:
            self.std_coe_matrix = self.coe_matrix.copy()


    def GenerateArray(self):
        '''genertare Array'''
        while True:
            dim_input = input("请输入系数矩阵的维数，用逗号分隔>>:").strip().split(',')
            if dim_input[0].strip().isdigit() and dim_input[1].strip().isdigit():
                dimx, dimy = int(dim_input[0]), int(dim_input[1])
                if dimx > 0 and dimy > 0: break
            print("请输入正确的数据！")
        i = 0
        data = []
        print("===输入矩阵元素请用空格或者逗号分隔===")
        while i < dimx:
            data_input = input("请输入第%d行的数据>>:"%(i+1)).strip()
            #输入用空格或者逗号分隔
            try:
                if ',' in data_input:
                    data_input = data_input.split(',')
                else :
                    data_input = data_input.split(' ')
                del_num = 0     #记录删除了多少数据
                for j in range(len(data_input)):
                    if data_input[j-del_num] == '':
                        del data_input[j-del_num]
                        del_num += 1
                        continue
                    if '.' in data_input[j-del_num]:
                        data_input[j - del_num] = float(data_input[j - del_num].strip())
                    else:
                        data_input[j-del_num] = int(data_input[j-del_num].strip())
                if len(data_input) != dimy:
                    print("数组维数必须是%d*%d" % (dimx,dimy))
                    continue
                data.append(data_input)
                i += 1
            except ValueError:
                print("非法输入，数组元素必须是int或float型")
        # if not dimx == dimy:
        #     Data = convert_to_square(Data)
        coe_matrix = np.array(data)     #系数矩阵
        print("生成系数矩阵:\n", coe_matrix)
        return coe_matrix


    def StdCoeMatrix(self ,type = "min"):
        '''Standardization the Coefficient Matrix'''
        if type not in ("min","max"):
            raise TypeError("The type must be 'max' or 'min'")
        self.std_coe_matrix = []
        for i in range(self.dimx):
            self.std_coe_matrix.append([])
            for j in range(self.dimy):
                self.std_coe_matrix[i].append(self.coe_matrix[i][j])
        dimx, dimy = self.coe_matrix.shape
        if dimx > dimy:
            add_num = dimx - dimy
            for i in range(dimx):
                for j in range(add_num):
                    self.std_coe_matrix[i].append(MIN if type == "max" else MAX)
        elif dimx < dimy:
            add_num = dimy - dimx
            for i in range(add_num):
                self.std_coe_matrix.append([])
                for i in range(dimy):
                    self.std_coe_matrix[-1].append(MIN if type == "max" else MAX)
        self.std_coe_matrix2 = np.array(self.std_coe_matrix)
        max_value = max(max(self.std_coe_matrix))
        if type == "max":
            for i in range(self.dimx):
                for j in range(self.dimy):
                    self.std_coe_matrix[i][j] = max_value - self.std_coe_matrix[i][j]
        self.std_coe_matrix = np.array(self.std_coe_matrix)


    def ViolentSolve(self, type = 'min'):
        '''
        using violent method to solve the assign problem
        and get the best solution and the best assign-vectors
        '''
        if type not in ("min","max"):
            raise TypeError("The type must be 'max' or 'min'")
            return
        dim = max(self.dimx, self.dimy)
        possible_choice = list(itertools.permutations(list(range(dim)), dim))
        for key in possible_choice:  # 将元组转换成列表
            possible_list = list(key)
            #Generate the sparse vector
            possible_matrix = np.zeros((dim, dim),dtype='int32')
            for i in range(dim):
                possible_matrix[i, possible_list[i] - 1] = 1
            result = (possible_matrix * self.std_coe_matrix2).sum()
            if type == "min":
                if result < self.best_result:
                    self.best_solve = [possible_matrix]
                    self.best_result = result
                elif result == self.best_result:
                    self.best_solve.append(possible_matrix)
            else:
                if result > self.best_result:
                    self.best_solve = [possible_matrix]
                    self.best_result = result
                elif result == self.best_result:
                    self.best_solve.append(possible_matrix)
        return self.best_solve, self.best_result


    def SetFlag(self, num=None):
        '''
        Initializing or setting the flag ,which is used to check if all the best solutions have been gotten.
        '''
        if num == None:
            self.flag = [0 for i in range(len(self.best_solve))]
            return 1
        elif num < len(self.flag):
            if self.flag[num] == 0:
                self.flag[num] = 1
                return 1
            else:
                return 2


    def IsBestSolution(self, solution):
        '''
        this method is used to check if the solution is one of the best solutions.
        '''
        if (solution * self.std_coe_matrix).sum() == self.best_result:
            for i in range(len(self.best_solve)):
                if (solution == self.best_solve[i]).all():
                    try:
                        if self.SetFlag(i) == 2:
                            # print("-->这个最优解已经在前面通过其他方法被求得")
                            return 2
                    except ValueError:
                        pass
                    return 1
        return False


    def IsOnlyZero(self, original_matrix, num, mode=0):
        '''
            Checking the whether the list or row only have one 0 element.
            :param original_matrix: 原始矩阵
            :param num: 行/列号
            :param mode: 模式选择，mode=0为行，mode=1为列
            :return: 只有一个0元素返回0元素的行列号列表, 否则返回False
        '''
        if mode == 0:
            if 0 in list(original_matrix[num]):
                if list(original_matrix[num]).count(0) == 1:
                    return [num, list(original_matrix[num]).index(0)]
        elif mode == 1:
            if 0 in list(original_matrix[:, num]):
                if list(original_matrix[:, num]).count(0) == 1:
                    return [list(original_matrix[:, num]).index(0), num]
        return False


    def IsFirstShown(self, solution_martix):
        '''
        Checking whether the selution
        :param original_martix: 变形后的稀疏矩阵，即operate()函数中的变量m
        :return: 没有出现过返回True并将它加入历史列表,出现过就返回False
        '''
        for i in range(len(self.all_mid_matrix)):
            if (solution_martix == self.all_mid_matrix[i]).all():
                return False
        self.all_mid_matrix.append(solution_martix)
        return True


    def MatrixConvert(self, original_matrix, row_list=[], line_list=[]):
        '''
            transfering the original_matrix to another form to help us to get the best method of assigning.
            :param original_matrix: 原始矩阵
            :param raw_list: 存储需变换的行的列表,默认为空
            :param line_list: 存储不需变换的列的列表,默认为空
            以上两个参数就是指派问题解法中打钩的行和列
            :return:m：变换后的矩阵
        '''
        dim = max(self.dimx, self.dimy)
        m = original_matrix.copy()
        # if len(row_list) == dim :    #没有行需要变换
        #     return
        if row_list == []:  # 这是第一次变换
            for i in range(dim):
                min_element = m[i].min()
                for j in range(dim):
                    m[i][j] = m[i][j] - min_element
        else:  # 这不是第一次变换
            min_element = MAX*10 # 足够大的一个数
            for i in range(dim):
                if i not in row_list: continue
                for j in range(dim):
                    if j in line_list: continue
                    if m[i][j] < min_element : min_element = m[i][j]
            # print(min_element)
            for i in range(dim):
                for j in line_list:
                    m[i][j] += min_element
                if i in row_list:
                    for j in range(dim):
                        m[i][j] -= min_element
        return m


    def Solver(self, original_matrix, m=None, row_list=None, line_list=None, parent=None ):
        '''
        用于对问题进行求解，对应教材中的"找0，画圈"动作。
        矩阵中的-1表示指派，-2表示不指派；分别对应《运筹学》教材中的“画圈”和“划去”操作。
        :param original_matrix: 初始矩阵
        :param m/row_list/line_list：这三个数据均为迭代求解多解时才会使用的参数，用于紧接着上一步继续求解。
                                    如果进行的是矩阵变换后的一次新的求解过程，则不需要对这三个数据赋值。
        :param parent: 用于记录当前步骤来自哪一步，方便用户阅读和分析理解。
        :return:
        '''
        try:
            if m == None:      #表明是一次完整处理的开始，那么就对这些数据进行初始化
                m = original_matrix.copy()
                row_list = list(range(max(self.dimx, self.dimy)))  # 不在row_list内的行的0元素被划去
                line_list = []  # 在line_list内的列的0元素被划去
                # print("=======第%d步======="%step)
                self.step += 1
                if parent == None:
                    self.transfer_count = 1
                    self.process_queue.put(1)
        except ValueError:
            pass
        op_count = 1
        if not (m == original_matrix).all():    #如果这并非第一次处理，那么先进行一次处理
            for i in range(max(self.dimx, self.dimy)):
                if i in row_list : continue
                for j in range(max(self.dimx, self.dimy)):
                    if m[i,j] == 0 : m[i,j] = -2
            for j in line_list:
                for i in range(max(self.dimx, self.dimy)):
                    if m[i,j] == 0 : m[i,j] = -2
        while 0 in m and op_count > 0:  #只要矩阵中还有未被圈出的0元素
            op_count = 0            #计数器初始化，代表操作（圈出）的0元素个数
            for row in range(max(self.dimx, self.dimy)):
                if row not in row_list : continue
                res = self.IsOnlyZero(m, row, 0)
                if res:
                    if res[1] not in line_list:
                        line_list.append(res[1])
                        m[res[0], res[1]] = -1
                        op_count += 1
            for line in range(max(self.dimx, self.dimy)):
                if line in line_list : continue
                res = self.IsOnlyZero(m, line, 1)
                if res:
                    if res[0] in row_list:
                        del row_list[row_list.index(res[0])]
                        m[res[0], res[1]] = -1
                        op_count += 1
            for i in range(max(self.dimx, self.dimy)):
                if i in row_list : continue
                for j in range(max(self.dimx, self.dimy)):
                    if m[i,j] == 0 : m[i,j] = -2
            for j in line_list:
                for i in range(max(self.dimx, self.dimy)):
                    if m[i,j] == 0 : m[i,j] = -2
        if op_count == 0:   #已经没有任何行或列仅有一个0元素，但存在仍未被圈出的0元素
            print("---此步存在有多种选择---")
            for i in range(max(self.dimx, self.dimy)):
                for j in range(max(self.dimx, self.dimy)):
                    if m[i,j] == 0:
                        m_multi = m.copy()
                        row_list_multi = row_list.copy()
                        line_list_multi = line_list.copy()
                        m_multi[i,j] = -1
                        line_list_multi.append(j)
                        del row_list_multi[row_list_multi.index(i)]
                        gevent.spawn(self.Solver, original_matrix, m_multi, row_list_multi, line_list_multi, parent)
                        self.process_queue.put(1)
                        # print("-->选择第%d行第%d列的元素：" % (i+1, j+1))
                        # gevent.sleep(0.01)
            self.process_queue.get()
        else:   #否则代表本次操作已经结束
            result_matrix = np.zeros((max(self.dimx, self.dimy), max(self.dimx, self.dimy)),dtype='int32')  #当前的指派矩阵
            count = 0
            for i in range(max(self.dimx, self.dimy)):
                for j in range(max(self.dimx, self.dimy)):
                    if m[i,j] == -1:
                        result_matrix[i,j] = 1
                        count += 1

            #输出编号
            if not self.transfer_count == 1 and not parent == None:
                print("当前编号：%d，"%self.transfer_count, "来自：%d"%parent)
            else:
                print("当前编号：%d" % self.transfer_count)
            #输出当前结果
            if self.IsBestSolution(result_matrix)==1:
                print("---得 到 最 优 解---")
            elif self.IsBestSolution(result_matrix)==2:
                print("---得到一个已经在之前被求得的最优解---")
            # elif len(row_list) <= len(line_list) and count < dim:   #这个判断条件仍有待验证
            #     print("---多选操作时选择错误，无法达到最优解---")
            else:
                if self.IsFirstShown(m):
                    if len(row_list) == 0 or len(line_list) == 5:
                        print("---这不是最优解---")
                    else:
                        print("---尚未达最优解，将继续迭代---")    #继续迭代
                        # print(matrix_convert(original_matrix, row_list, line_list))
                        print(row_list, line_list)
                        gevent.spawn(self.Solver, self.MatrixConvert(original_matrix, row_list, line_list), parent=self.transfer_count)
                        self.process_queue.put(1)
                else:
                    print("---尚未达到最优解，且当前解在前面已被求得---")
            print("变形后的系数矩阵为：")
            print(m)
            print("指派矩阵为：")
            print(result_matrix)
            print("------------------")
            self.transfer_count += 1
            self.process_queue.get()
        gevent.sleep(0.1)


if __name__ =="__main__":
    s = AssignSlover()
    s.coe_matrix = np.array([[6, 7, 8, 7, 15],
                             [10, 9, 6, 6, 15],
                             [7, 12, 9, 14, 15],
                             [12, 14, 6, 6, 15],
                             [8, 6, 0, 7, 15]])
    # s.coe_matrix = np.array([[37.7, 32.9, 33.8, 37.0],
    #                          [43.4, 33.1, 42.2, 34.7],
    #                          [33.3, 28.5, 38.9, 30.4],
    #                          [29.2, 26.4, 29.6, 28.5],
    #                          [30.1, 31.5, 29.2, 33.4],
    #                          ])
    # s.coe_matrix = np.array([[37.7, 32.9, 33.8, 37.0, 35.4], [43.4, 33.1, 42.2, 34.7, 41.8], [33.3, 28.5, 38.9, 30.4, 33.6]])
    # s.coe_matrix = s.GenerateArray()
    s.SetCoeMatrix(s.coe_matrix)
    s.__start__()



import gevent
import numpy as np
import itertools
import queue
M = 10000           #大M


def generate_array():
    '''生成数组'''
    #输入方阵维数
    global dimx, dimy

    while True:
        dim = input("请输入系数矩阵的维数，用逗号分隔>>:").strip()
        dim = dim.split(',')
        if dim[0].strip().isdigit() and dim[1].strip().isdigit():
            dimx, dimy = int(dim[0]), int(dim[1])
            if dimx > 0 and dimy > 0: break
        print("请输入正确的数据！")
    i = 0
    data = []
    print("===输入请用空格或者逗号分隔===")
    while i < dimx:
        data_input = input("请输入第%d行的数据>>:"%(i+1)).strip()
        #输入用空格或者逗号分隔
        try:
            if ',' in data_input:
                data_input = data_input.split(',')
            else : data_input = data_input.split(' ')
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
    if not dimx == dimy:
        data = convert_to_square(data)
    coe_matrix = np.array(data)     #系数矩阵
    print("生成系数矩阵:\n", coe_matrix)
    return coe_matrix


def convert_to_square(original_matrix):
    '''
    用大M法将矩阵转化为方阵
    :param original_matrix: 原始系数矩阵(list类型)
    :param dimx: 矩阵行数
    :param dimy: 矩阵列数
    :return: 方阵
    '''
    global dimx, dimy
    if dimx > dimy:
        add_num = dimx - dimy
        for i in range(dimx):
            for j in range(add_num):
                original_matrix[i].append(M)
    elif dimx < dimy:
        add_num = dimy - dimx
        for i in range(add_num):
            original_matrix.append([])
            for i in range(dimy):
                original_matrix[-1].append(M)
    return original_matrix


def violence_solve(coe_matrix):
    '''
        暴力求解指派问题的最优解
        :param coe_matrix:系数矩阵
        :return:best_solve:最优解矩阵
                best_result:最优解
    '''
    dim, dim = coe_matrix.shape
    best_solve = []
    best_result = 1000000     #足够大的一个值
    possible_choice = list(itertools.permutations(list(range(dim)), dim))
    for key in possible_choice:     #将元组转换成列表
        possible_list = list(key)
        #生成稀疏矩阵
        possible_matrix = np.zeros((dim,dim), dtype="int32")
        for i in range(dim):
            possible_matrix[i, possible_list[i]-1] = 1
        result = (possible_matrix * coe_matrix).sum()
        if result < best_result:
            best_solve = [possible_matrix]
            best_result = result
        elif result == best_result:
            best_solve.append(possible_matrix)
    return best_solve, best_result


def set_flag(num = None):
    '''
    用于初始化/修改最优解的标志位，可以判断是否求得全部最优解
    :param num: 本次操作第num个标志位/如果为None则表示初始化所有标志位
    :return: 1：成功设置标志位/2：这个标志位已经在前面过程中被置位
    '''
    global flag
    if num == None:
        flag = [0 for i in range(len(best_solve))]
        return 1
    elif num < len(flag):
        if flag[num] == 0:
            flag[num] = 1
            return 1
        else:
            return 2


def is_best_solution(solution):
    '''
    用于判断一个解是否是最优解
    :param solution: 解矩阵
    :return: 如果是新的最优解返回1，如果是以前已经求得的最优解则返回2，否则返回False
    '''
    if (solution*coe_matrix).sum() == best_result:
        for i in range(len(best_solve)):
            if (solution==best_solve[i]).all():
                try:
                    if set_flag(i) == 2:
                        # print("-->这个最优解已经在前面通过其他方法被求得")
                        return 2
                except ValueError:pass
                return 1
    return False


def matrix_convert(original_matrix, row_list=[], line_list=[]):
    '''
        矩阵变换函数
        :param original_matrix: 原始矩阵
        :param raw_list: 存储需变换的行的列表,默认为空
        :param line_list: 存储不需变换的列的列表,默认为空
        以上两个参数就是指派问题解法中打钩的行和列
        :return:m：变换后的矩阵
    '''
    dim, dim = original_matrix.shape
    m = original_matrix.copy()
    # if len(row_list) == dim :    #没有行需要变换
    #     return
    if row_list == []:      #这是第一次变换
        for i in range(dim):
            min_element = m[i].min()
            for j in range(dim):
                m[i][j] = m[i][j] - min_element
    else:                   #这不是第一次变换
        min_element = 1000000     #足够大的一个数
        for i in range(dim):
            if i not in row_list : continue
            for j in range(dim):
                if j in line_list:continue
                if m[i][j] < min_element : min_element = m[i][j]
        # print(min_element)
        for i in range(dim):
            for j in line_list:
                m[i][j] += min_element
            if i in row_list:
                for j in range(dim):
                    m[i][j] -= min_element
    return m


def is_only_zero(original_matrix, num, mode=0):
    '''
        判断此行/列是否只有一个0元素
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
        if 0 in list(original_matrix[:,num]):
            if list(original_matrix[:,num]).count(0) == 1:
                return [list(original_matrix[:, num]).index(0), num]
    return False


def how_many_zero(original_matrix, num, mode=0):
    '''
        判断此行/列是有多少个0元素
        一般只有当个数大于2时才会调用此函数
        :param original_matrix: 原始矩阵
        :param num: 行/列号
        :param mode: 模式选择，mode=0为行，mode=1为列
        :return: count：0元素的个数
        '''
    if mode == 0:
        return list(original_matrix[num]).count(0)
    elif mode == 1:
        return list(original_matrix[:, num]).count(0)


def is_first_shown(original_martix):
    '''
    用来判断这个矩阵是否在前面的求解步骤中出现过
    :param original_martix: 变形后的稀疏矩阵，即operate()函数中的变量m
    :return: 没有出现过返回True并将它加入历史列表,出现过就返回False
    '''
    global all_mid_matrix
    for i in range(len(all_mid_matrix)):
        if (original_martix==all_mid_matrix[i]).all():
            return False
    all_mid_matrix.append(original_martix)
    return True


def operate(original_matrix, m=None, row_list=None, line_list=None, parent=None ):
    '''
    用于对问题进行求解，对应教材中的"找0，画圈"动作。
    矩阵中的-1表示指派，-2表示不指派；分别对应《运筹学》教材中的“画圈”和“划去”操作。
    :param original_matrix: 初始矩阵
    :param m/row_list/line_list：这三个数据均为迭代求解多解时才会使用的参数，用于紧接着上一步继续求解。
                                如果进行的是矩阵变换后的一次新的求解过程，则不需要对这三个数据赋值。
    :param parent: 用于记录当前步骤来自哪一步，方便用户阅读和分析理解。
    :return:
    '''
    global step, transfer_count, process_queue
    dim, dim = original_matrix.shape
    try:
        if m == None:      #表明是一次完整处理的开始，那么就对这些数据进行初始化
            m = original_matrix.copy()
            row_list = list(range(dim))  # 不在row_list内的行的0元素被划去
            line_list = []  # 在line_list内的列的0元素被划去
            # print("=======第%d步======="%step)
            step += 1
            if parent == None:
                transfer_count = 1
                process_queue.put(1)
    except ValueError:
        pass
    dim, dim = original_matrix.shape
    op_count = 1
    if not (m == original_matrix).all():    #如果这并非第一次处理，那么先进行一次处理
        for i in range(dim):
            if i in row_list : continue
            for j in range(dim):
                if m[i,j] == 0 : m[i,j] = -2
        for j in line_list:
            for i in range(dim):
                if m[i,j] == 0 : m[i,j] = -2
    while 0 in m and op_count > 0:  #只要矩阵中还有未被圈出的0元素
        op_count = 0            #计数器初始化，代表操作（圈出）的0元素个数
        for row in range(dim):
            if row not in row_list : continue
            res = is_only_zero(m, row, 0)
            if res:
                if res[1] not in line_list:
                    line_list.append(res[1])
                    m[res[0], res[1]] = -1
                    op_count += 1
        for line in range(dim):
            if line in line_list : continue
            res = is_only_zero(m, line, 1)
            if res:
                if res[0] in row_list:
                    del row_list[row_list.index(res[0])]
                    m[res[0], res[1]] = -1
                    op_count += 1
        for i in range(dim):
            if i in row_list : continue
            for j in range(dim):
                if m[i,j] == 0 : m[i,j] = -2
        for j in line_list:
            for i in range(dim):
                if m[i,j] == 0 : m[i,j] = -2
    if op_count == 0:   #已经没有任何行或列仅有一个0元素，但存在仍未被圈出的0元素
        print("---此步存在有多种选择---")
        for i in range(dim):
            for j in range(dim):
                if m[i,j] == 0:
                    m_multi = m.copy()
                    row_list_multi = row_list.copy()
                    line_list_multi = line_list.copy()
                    m_multi[i,j] = -1
                    line_list_multi.append(j)
                    del row_list_multi[row_list_multi.index(i)]
                    gevent.spawn(operate, original_matrix, m_multi, row_list_multi, line_list_multi, parent)
                    process_queue.put(1)
                    # print("-->选择第%d行第%d列的元素：" % (i+1, j+1))
                    # gevent.sleep(0.01)
        process_queue.get()
    else:   #否则代表本次操作已经结束
        result_matrix = np.zeros((dim, dim),dtype='int32')  #当前的指派矩阵
        count = 0
        for i in range(dim):
            for j in range(dim):
                if m[i,j] == -1:
                    result_matrix[i,j] = 1
                    count += 1

        #输出编号
        if not transfer_count == 1 and not parent == None:
            print("当前编号：%d，"%transfer_count, "来自：%d"%parent)
        else:
            print("当前编号：%d" % transfer_count)
        #输出当前结果
        if is_best_solution(result_matrix)==1:
            print("---得 到 最 优 解---")
        elif is_best_solution(result_matrix)==2:
            print("---得到一个已经在之前被求得的最优解---")
        # elif len(row_list) <= len(line_list) and count < dim:   #这个判断条件仍有待验证
        #     print("---多选操作时选择错误，无法达到最优解---")
        else:
            if is_first_shown(m):
                print("---尚未达最优解，将继续迭代---")    #继续迭代
                # print(matrix_convert(original_matrix, row_list, line_list))
                # print(row_list, line_list)
                gevent.spawn(operate, matrix_convert(original_matrix, row_list, line_list), parent=transfer_count)
                process_queue.put(1)
            else:
                print("---尚未达到最优解，且当前解在前面已被求得---")
        print("变形后的系数矩阵为：")
        print(m)
        print("指派矩阵为：")
        print(result_matrix)
        print("------------------")
        transfer_count += 1
        process_queue.get()
    gevent.sleep(0.1)


def main():
    while True:
        user_choice = input("====请选择操作====\n   1.运行软件\n   2.打印结果\n   3.退出软件\n>>>").strip()
        if not user_choice.isdigit():
            print("无效选择")
            continue
        if int(user_choice) == 1:
            #定义一堆全局变量
            global dimx, dimy, flag, step, transfer_count, best_solve, best_result, coe_matrix, process_queue, all_mid_matrix
            #初始化全局变量
            # coe_matrix = np.array(
                # [[12, 7, 9, 7, 9], [8, 9, 6, 6, 6], [7, 17, 12, 14, 9], [15, 14, 6, 6, 10], [4, 10, 7, 10, 9]])
            dimx ,dimy = 5 , 5
            # coe_matrix = np.array(
            #     convert_to_square([[37.7, 32.9, 33.8, 37.0, 35.4], [43.4, 33.1, 42.2, 34.7, 41.8], [33.3, 28.5, 38.9, 30.4, 33.6], [29.2, 26.4, 29.6, 28.5, 31.1]]))
            coe_matrix = np.array(convert_to_square( [[ 4,8 ,7 ,15, 12],[ 7  ,9 ,17, 14, 10],[ 6  ,9 ,12 , 8 , 7],[ 6 , 7 ,14 , 6 ,10],[ 6 , 9 ,12 ,10 , 6]]))
            coe_matrix = generate_array()
            flag = []
            step = 1
            transfer_count = 1
            best_solve, best_result = violence_solve(coe_matrix)
            process_queue = queue.Queue()
            set_flag()
            all_mid_matrix = []
            #开始求解
            gevent.spawn(operate, matrix_convert(coe_matrix))
            print("===解题步骤如下===")
            while 0 in flag or not process_queue.empty():
                gevent.sleep(0.1)
            else:
                print("\n-->经检验，已经遍历得到全部最优解")
                print("-->最优解为%f"%(best_result-abs(dimx-dimy)*M))
                print("-->最优的指派矩阵为：")
                for obj in best_solve:
                    print(obj)
        elif int(user_choice) == 2:
            pass
        elif int(user_choice) == 3:
            break
        else:
            print("无效选择")
            continue

if __name__ == "__main__":
    gevent.joinall([gevent.spawn(main)])

    # coe_matrix = generate_array()
    # coe_matrix = np.array([[12,7,9,7,9],[8,9,6,6,6],[7,17,12,14,9],[15,14,6,6,10],[4,10,7,10,9]])   #方便调试
    # best_solve, best_result = violence_solve(coe_matrix)
    # print("---最优指派矩阵为---")
    # for key, obj in enumerate(best_solve):
    #     print(key,'\n',obj)
    # print("---最优解为:%d---"%best_result)
    # print(matrix_convert(matrix_convert(coe_matrix), [2,4], [0]))
    # print(is_only_zero(matrix_convert(coe_matrix), 2, 0))
    # gevent.spawn(operate, matrix_convert(coe_matrix))

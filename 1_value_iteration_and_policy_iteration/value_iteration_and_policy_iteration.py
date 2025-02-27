import numpy as np
import random
import time
import os



def get_reward(location, action, graph):
    r, c = len(graph), len(graph[0])
    reward = 0  # 默认奖励为0
    row, col = location
    # 根据移动的方向调整位置坐标  {0: '↑', 1: '↓', 2: '←', 3: '→', 4: 'O'}
    if action == 0:
        row = row - 1
    elif action == 1:
        row = row + 1
    elif action == 2:
        col = col - 1
    elif action == 3:
        col = col + 1
    
    # 根据新的位置（移动后）坐标计算reward    
    if row < 0 or row > r - 1 or col < 0 or col > c - 1:
        reward = -1
    elif graph[row][col] == '×':
        reward = -100
    elif graph[row][col] == '●':
        reward = 20
    
    # 控制边界约束
    row = max(0, row)
    row = min(r - 1, row)
    col = max(0, col)
    col = min(c - 1, col)
    # 返回下一个状态（位置）以及奖励
    return row, col, reward


class Solver(object):
    """基类"""
    def __init__(self, r: int, c: int):
        """
        :param r: 代表地图行数
        :param c: 代表地图列数
        """
        # 动作空间
        self.idx_to_action = {0: '↑', 1: '↓', 2: '←', 3: '→', 4: 'O'}
        # 地图行数、列数、动作个数
        self.r, self.c, self.action_nums = r, c, len(self.idx_to_action)
        self.state_value_matrix = np.random.randn(r, c)  # 状态价值矩阵初始化(5,5)
        self.action_value_matrix = np.random.randn(r, c, len(self.idx_to_action))  # 动作价值矩阵初始化(5,5,5)
        self.cur_best_policy = np.random.choice(len(self.idx_to_action), size=(r, c))  # 当前最优策略初始化(5,5)

    # 打印最优策略
    def show_policy(self):
        for i in self.cur_best_policy.tolist():
            print(*[self.idx_to_action[idx] for idx in i], sep=' ')

    # 显示地图
    def _show_graph(self, graph):
        for i in graph:
            print(*i, sep=' ')

    # 清空控制台
    def _clear_console(self):
        if os.name == 'nt':  # for Windows
            _ = os.system('cls')
        else:  # for Linux and Mac
            _ = os.system('clear')

    def show_point_to_point(self, start_point, end_point, graph):
        assert (0 <= start_point[0] < self.r) and (0 <= start_point[1] < self.c), f'The start_point is {start_point}, is out of range.'
        assert (0 <= end_point[0] < self.r) and (0 <= end_point[1] < self.c), f'The end_point is {end_point}, is out of range.'

        row, col = start_point
        i = 0
        while True:
            graph[row][col] = self.idx_to_action[self.cur_best_policy[row][col]]
            self._clear_console()  # 清空控制台
            self._show_graph(graph)  # 显示地图
            time.sleep(0.5)
            row, col, _ = get_reward((row, col), self.cur_best_policy[row][col], graph)
            if (row, col) == end_point or i > self.r * self.c:
                break
            i += 1

    # epsilon贪婪法，当epsilon=0时，完全贪婪法
    def get_epsilon_greedy_action(self, state, epsilon=0.):
        row, col = state
        # 找最优动作
        best_action = np.argmax(self.action_value_matrix[row][col]).item()
        # epsilon贪婪法，当epsilon != 0时，才有可能进入该if语句，否则直接返回最优动作
        if random.random() < epsilon * (self.action_nums - 1) / self.action_nums:
            actions = list(self.idx_to_action.keys())
            actions.remove(best_action)
            return random.choice(actions)
        return best_action


class ValueIterationSolver(Solver):
    """值迭代算法"""
    def __init__(self, r: int, c: int):
        super().__init__(r, c)

    def update(self, graph, gama=0.8, eps=1e-4):
        # 第1步：初始化：将所有状态的值设为0。
        last_state_value_matrix = np.ones_like(self.state_value_matrix)
        # 第2步：开始迭代更新   第三步：收敛判断：当两次迭代的差值小于阈值时停止。 
        while np.sum(np.abs(last_state_value_matrix - self.state_value_matrix)) > eps:
            last_state_value_matrix = self.state_value_matrix.copy()
            for row in range(self.r):
                for col in range(self.c):
                    # 第2步：对每个状态，计算所有可能动作的预期收益
                    for action in range(self.action_nums):
                        next_row, next_col, reward = get_reward((row, col), action, graph)
                        self.action_value_matrix[row][col][action] = reward + gama * self.state_value_matrix[next_row][next_col]
                    """策略更新"""
                    self.cur_best_policy[row, col] = np.argmax(self.action_value_matrix[row, col])
                    """值更新"""
                    # 第2步：取动作收益的最大值作为新状态值
                    self.state_value_matrix[row, col] = np.max(self.action_value_matrix[row, col])
            # """策略更新, 还可以最后在这里统一更新"""
            # self.cur_best_policy = np.argmax(self.action_value_matrix, axis=2)


class PolicyIterationSolver(Solver):
    def __init__(self, r: int, c: int):
        super().__init__(r, c)

    def update(self, graph, gama=0.8, eps=1e-4):
        last_best_policy = np.ones(shape=(r, c))
        i = 0
        # 第4步：重复评估和改进，直到策略稳定的不在变化（比如20次更新都没有改变）  
        while not np.array_equal(last_best_policy, self.cur_best_policy) or i < 20:
            last_state_value_matrix = np.ones_like(self.state_value_matrix)
            """ 第2步：策略评估，获取状态价值矩阵"""
            while np.sum(np.abs(last_state_value_matrix - self.state_value_matrix)) > eps:
                last_state_value_matrix = self.state_value_matrix.copy()
                for row in range(self.r):
                    for col in range(self.c):
                        action = self.cur_best_policy[row][col]
                        next_row, next_col, reward = get_reward((row, col), action, graph)
                        self.state_value_matrix[row][col] = reward + gama * self.state_value_matrix[next_row][next_col]
            """ 第三步：策略改进，获取改进策略"""
            last_best_policy = self.cur_best_policy.copy()
            for row in range(self.r):
                for col in range(self.c):
                    for action in range(self.action_nums):
                        next_row, next_col, reward = get_reward((row, col), action, graph)
                        self.action_value_matrix[row][col][action] = reward + gama * self.state_value_matrix[next_row][next_col]
                    self.cur_best_policy[row][col] = np.argmax(self.action_value_matrix[row, col])
            # 策略不更新了，但不一定是最优策略，要多迭代几次观察
            if np.array_equal(last_best_policy, self.cur_best_policy):
                i += 1
            else:
                i = 0

if __name__ == "__main__":
    graph = [['□', '□', '□', '□', '□', '□'],
             ['□', '×', '×', '□', '□', '×'],
             ['□', '□', '×', '□', '□', '×'],
             ['□', '×', '●', '×', '□', '×'],
             ['□', '×', '□', '□', '□', '×']]

    # graph = [['□', '□', '□'],
    #          ['□', '□', '×'],
    #          ['×', '□', '●']]

    # graph = [['□', '□', '□', '□', '□'],
    #          ['□', '×', '×', '□', '□'],
    #          ['□', '□', '×', '□', '□'],
    #          ['□', '×', '●', '×', '□'],
    #          ['□', '×', '□', '□', '□']]
    r = len(graph)
    c = len(graph[0])

    """值迭代算法"""
    # value_iterater = ValueIterationSolver(r, c)
    # value_iterater.update(graph)
    # value_iterater.show_policy()
    # value_iterater.show_point_to_point((0, 0), (3, 2), graph)

    """策略迭代算法, 其收敛速度比值迭代算法快"""
    policy_iterater = PolicyIterationSolver(r, c)
    policy_iterater.update(graph)
    policy_iterater.show_policy()
    policy_iterater.show_point_to_point((2, 1), (3, 2), graph)

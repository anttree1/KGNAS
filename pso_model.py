import random
import utils as ut
import fitness_model as fm
import math



def build_swarm(swarm_size, list_len):
    output_list = []
    for i in range(swarm_size):
        swarm_list = ''.join(random.choice(['0', '1']) for _ in range(list_len))
        output_list.append(swarm_list)
    return output_list


def evolving_k(k, k_size, list_len):
    list1 = list(range(list_len))
    random.shuffle(k)
    k = k[:k_size]
    new_elements = random.sample(list(set(list1) - set(k)), math.floor(k_size / 2))
    k.extend(new_elements)
    k.sort()
    return k


def rand_V(per):
    num = random.random()
    if num <= per:
        return 0
    else:
        return 1


def pso():
    list_len = 74
    swarm_size = 30
    iter_num = 200
    per = 0.6
    per_2 = 0.5
    M = 50
    k1 = random.sample(range(0, list_len), list_len)
    k1.sort()
    swarm_list = build_swarm(swarm_size, list_len)
    fitness_list = []
    for var_swarm in swarm_list:
        list_arch = ut.individual_list(var_swarm)
        fitness_list.append(fm.main_arch(list_arch))
    p_best = swarm_list
    g_best = swarm_list[fitness_list.index(max(fitness_list))]
    print('初始化的种群:', swarm_list)
    for var_iter in range(iter_num):
        swarm_iter_list = []
        print('iter', var_iter)
        for j, var_swarmlist in enumerate(swarm_list):
            V = ''
            for i1 in range(list_len):
                a = rand_V(per)  # a = r1
                if a == 0:
                    V += (p_best[j][i1])
                else:
                    V += (g_best[i1])
            individual_l = ''
            for i2, var2 in enumerate(var_swarmlist):
                b = rand_V(per_2)
                if b == 1 and i2 not in k1:
                    individual_l += V[i2]
                else:
                    individual_l += var2
            swarm_iter_list.append(individual_l)
        swarm_list_now = swarm_iter_list
        fitness_list_now = []
        for i, var_swarm_now in enumerate(swarm_list_now):
            list_arch_now = ut.individual_list(var_swarm_now)
            var_fit = fm.main_arch(list_arch_now)
            fitness_list_now.append(var_fit)
            print("第 ", i, ' 个体为: ', list_arch_now, ' 适应度为: ', var_fit)
        for var_num in range(len(swarm_list_now)):
            if fitness_list[var_num] < fitness_list_now[var_num]:
                fitness_list[var_num] = fitness_list_now[var_num]
                p_best[var_num] = swarm_iter_list[var_num]
        g_best = p_best[fitness_list.index(max(fitness_list))]
        swarm_list = swarm_list_now
        if var_iter % 5 == 0:
            k1_size = math.floor(M / math.sqrt(var_iter + 5))
            k1 = evolving_k(k1, k1_size, list_len)
        print('Iteration number:', var_iter, 'gBest: ', ut.individual_list(g_best))
        print('----------------------------')


pso()

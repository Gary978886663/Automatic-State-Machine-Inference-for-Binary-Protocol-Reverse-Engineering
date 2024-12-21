import re
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances, davies_bouldin_score
from sklearn.cluster import KMeans
from scapy.all import *
from scapy.layers.inet import IP, TCP, UDP
from collections import defaultdict, Counter    # 概率状态机构建
import matplotlib.pyplot as plt
import logging
        

# 检查list1是否为list2的子串, 考虑列表中元素的顺序关系，当两串相等的时候，不认为是子串
def is_subString(list1, list2):
    is_sublist = True
    # print("初始值: {}".format(is_sublist))
    if (len(list2) - len(list1)) <= 0:
        is_sublist = False
    else:
        for i in range(len(list2) - len(list1) + 1):
            for j in range(len(list1)):
                # print("list1: {}, list2: {}".format(list1[j], list2[j + i]))
                
                if list1[j] != list2[j + i]:
                    is_sublist = False
                    break
                # 执行到这个位置证明循环可以执行完成，证明已经匹配上了，即list1是list2的子串
                if j == len(list1) - 1:
                    is_sublist = True
            if is_sublist == True:
                break
    return is_sublist

# 去掉data中的空格、制表符和换行符
def msgs(data):
  lines = [re.sub("[ |\t|\n]+","",l,re.DOTALL) for l in data]                                                                                                                                                                                                                                                                                                                                                                                             
  return lines

# 将msg转变为16进制
def hexmsgs(data):
  lines = [re.sub("[ |\t|\n]+","",l,re.DOTALL) for l in data]
  # 一种简洁方式创建列表，后面for l in data是对data进行迭代，前面是对每一个元素进行的处理，然后返回一个新列表
  # 此处为删除所有的制表符和空格
  lines = [bytes.fromhex(m) for m in lines] #Turn it into hex
  # 转换为16进制表示的字节串
  return lines

# 获得两个int型列表的最长公共子序列
def lcs(X, Y):
    m = len(X)
    n = len(Y)

    # 创建一个二维数组，用于存储LCS的子问题解
    L = [[0] * (n + 1) for _ in range(m + 1)]

    # 动态规划填充L数组
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    # 回溯得到LCS
    i, j = m, n
    lcs = []
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs.append(X[i - 1])
            i -= 1
            j -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # 反转lcs列表，得到LCS
    return lcs[::-1]

# 求frequentItem相对于packet的隶属度
def fuzzMembership(packet, frequentItem):
    packetLen = len(packet)
    frequentItemLen = len(frequentItem)
    longestCommonSequence = lcs(packet, frequentItem)
    lcss = len(longestCommonSequence)
    
    # 模糊隶属度函数分为两种情况
    # if lcss <= (frequentItemLen / 2):
    #     return (1 - (lcss / frequentItemLen) * (lcss / frequentItemLen))
        
    # if lcss > (frequentItemLen / 2):
    #     return (lcss / frequentItemLen) * (lcss / frequentItemLen)
    return (lcss / frequentItemLen) * (lcss / frequentItemLen)

# database为数据集，格式为二维整数列表，每个数据包以一个int类型的列表表示，min_support为最小支持度，0-1之间。
def apriori_all(database, min_support):
    # 初始化频繁序列模式集
    frequent_sequences = {}   # 字典
    k = 1

    # 找出频繁1序列模式
    item_set = set()
    
    for sequence in database:
        for item in sequence:
            item_set.add(item)
        
    itemsets_list = [[[item], 0] for item in item_set]   # 3重列表，二层元素为频繁项，支持度

    print(f"Type of sequence: {type(sequence)}, Value: {sequence}")


    # 计算支持度并筛选频繁1序列模式, 筛选支持度符合要求的长度为1的频繁项
    for sequence in database:
        for item in itemsets_list:
            if is_subString(item[0], [element for element in sequence]):
                item[1] += 1    # 表示item这个key对应的值++, 此处value代表频数
    
    num_sequences = float(len(database))
    
    for item in itemsets_list:
        item[1] = item[1]/num_sequences
    
    frequent_sequences[k] = []
    
    for item in itemsets_list:
        if item[1] >= min_support:
            frequent_sequences[k].append(item)
        
    # 生成k+1序列模式
    while frequent_sequences[k]:   # 直到这个字典为空才停止循环
        k += 1                
        # 生成候选集
        candidates = []
        for item1 in frequent_sequences[k - 1]:
            for item2 in frequent_sequences[1]:
                union_list = list()
                union_list.append(item1[0] + item2[0])
                union_list.append(0)
                if len(union_list[0]) == k:
                    for sequence in database:
                        if is_subString(union_list[0], sequence) == True:
                            union_list[1] = union_list[1] + 1
                    union_list[1] =  union_list[1]/num_sequences
                    if union_list[1] >= min_support:
                        candidates.append(union_list)
        
        frequent_sequences[k] = [item for item in candidates]
    # 移除空集
    # 创建一个要删除的键的列表
    keys_to_delete = [key for key, value in frequent_sequences.items() if value in [None, '', []]]
    # 删除具有空值的项
    for key in keys_to_delete:
        del frequent_sequences[key]  
    # for item in frequent_sequences:
    #     print(item)
    return frequent_sequences

def needleman_distance(seq1, seq2, match_score=1, mismatch_penalty=-1, gap_penalty=-2):
    n = len(seq1)
    m = len(seq2)
    
    # 初始化矩阵
    score_matrix = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    
    # 初始化第一行和第一列
    for i in range(1, n + 1):
        score_matrix[i][0] = i * gap_penalty
    for j in range(1, m + 1):
        score_matrix[0][j] = j * gap_penalty
        
    # 填充矩阵
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score_matrix[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_penalty)
            delete = score_matrix[i-1][j] + gap_penalty
            insert = score_matrix[i][j-1] + gap_penalty
            score_matrix[i][j] = max(match, delete, insert)
    
    # 回溯路径
    align1, align2 = [], []
    i, j = n, m
    while i > 0 and j > 0:
        current_score = score_matrix[i][j]
        if current_score == score_matrix[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_penalty):
            align1.append(seq1[i-1])
            align2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif current_score == score_matrix[i-1][j] + gap_penalty:
            align1.append(seq1[i-1])
            align2.append('-')
            i -= 1
        else:
            align1.append('-')
            align2.append(seq2[j-1])
            j -= 1
    
    # 考虑剩余的元素
    while i > 0:
        align1.append(seq1[i-1])
        align2.append('-')
        i -= 1
    while j > 0:
        align1.append('-')
        align2.append(seq2[j-1])
        j -= 1
    
    align1.reverse()
    align2.reverse()
    
    # 计算相同位置相同值的个数
    match_count = sum(1 for a, b in zip(align1, align2) if a == b and a != '-')
    return match_count
    # return score_matrix[n][m], align1, align2, match_count

def kmedoids(data, k, distance_function, max_iter=300):
    n = len(data)
    print(f"n: {n}, k: {k}")
    # 随机选择初始的k个medoids
    medoids = random.sample(range(n), k)
    # 初始化标签
    labels = [-1] * n
    for _ in range(max_iter):
        # 分配每个点到最近的medoid
        new_labels = [np.argmin([distance_function(data[i], data[m]) for m in medoids]) for i in range(n)]
        # 如果标签没有改变，聚类完成
        if new_labels == labels:
            break
        labels = new_labels
        # 对每个medoid进行调整
        for i in range(k):
            cluster_indices = [index for index, label in enumerate(labels) if label == i]
            if not cluster_indices:
                continue
            # 找出新的medoid，使得其到所有其他点的距离和最小
            min_cost = float('inf')
            best_medoid = medoids[i]
            for index in cluster_indices:
                cost = sum([distance_function(data[index], data[j]) for j in cluster_indices])
                if cost < min_cost:
                    min_cost = cost
                    best_medoid = index
            medoids[i] = best_medoid
    return labels, medoids

# 自动选择最优类簇个数
def find_optimal_k(data, k_min=2, k_max=10):
    sse = []
    silhouette_scores = []
    
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        sse.append(kmeans.inertia_)  # SSE
        score = silhouette_score(data, kmeans.labels_)  # 轮廓系数
        silhouette_scores.append(score)
    
    # 绘制SSE（肘部法则）
    plt.plot(range(k_min, k_max + 1), sse, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal k')
    plt.show()

    # 绘制轮廓系数
    plt.plot(range(k_min, k_max + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.show()
    
    # 返回最佳k值
    best_k = range(k_min, k_max + 1)[silhouette_scores.index(max(silhouette_scores))]
    return best_k

class ProbabilisticStateMachine:
    def __init__(self):
        self.transition_counts = defaultdict(Counter)
        self.transition_probabilities = defaultdict(dict)
    
    def add_sequence(self, sequence):
        """Add a sequence of states to the state machine."""
        for i in range(len(sequence) - 1):
            current_state = sequence[i]
            next_state = sequence[i + 1]
            self.transition_counts[current_state][next_state] += 1
    
    def compute_probabilities(self):
        """Compute the transition probabilities from the transition counts."""
        for current_state, transitions in self.transition_counts.items():
            total_transitions = sum(transitions.values())
            for next_state, count in transitions.items():
                current_prob = count / total_transitions
                self.transition_probabilities[current_state][next_state] = [current_prob, 0.0]
    
    def compute_total_probabilities(self):
        """Compute the transition probabilities relative to all state transitions."""
        total_transitions = sum(sum(transitions.values()) for transitions in self.transition_counts.values())

        for current_state, transitions in self.transition_counts.items():
            for next_state, count in transitions.items():
                current_prob = self.transition_probabilities[current_state][next_state][0]
                total_prob = count / total_transitions
                self.transition_probabilities[current_state][next_state] = [current_prob, total_prob]
    
    def get_next_state_probability(self, current_state, next_state):
        """Get the transition probability from current_state to next_state."""
        return self.transition_probabilities.get(current_state, {}).get(next_state, [0, 0])
    
    def generate_sequence(self, start_state, length):
        """Generate a sequence of states based on the transition probabilities."""
        sequence = [start_state]
        current_state = start_state
        for _ in range(length - 1):
            next_states = list(self.transition_probabilities[current_state].keys())
            probabilities = [self.transition_probabilities[current_state][next_state][0] for next_state in next_states]
            if next_states:
                next_state = np.random.choice(next_states, p=probabilities)
                sequence.append(next_state)
                current_state = next_state
            else:
                break  # No valid transitions available
        return sequence

class Session:
    def __init__(self, src_ip, dst_ip, src_port, dst_port, protocol):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        self.packet_indices = []    # 该列表之中的每一个元素也是一个二元列表，第一个元素为数据包序号，第二个元素为该数据包被聚类的类簇号

    def add_packet_index(self, index):
        self.packet_indices.append([index, -1])

    def get_info(self):
        return {
            'src_ip': self.src_ip,
            'dst_ip': self.dst_ip,
            'src_port': self.src_port,
            'dst_port': self.dst_port,
            'protocol': self.protocol,
            'packet_indices': self.packet_indices
        }

def custom_distance_matrix(sequences):
    n = len(sequences)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            match_count = needleman_distance(sequences[i], sequences[j])
            # 计算距离，这里距离可以定义为匹配字符数的负值
            distance_matrix[i][j] = match_count
            distance_matrix[j][i] = distance_matrix[i][j]
    
    return distance_matrix

def dunn_index(X, labels):       # 越大越好
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # 计算所有点的 pairwise 距离
    distances = pairwise_distances(X)
    
    # 计算类间最小距离
    min_intercluster_dist = np.inf
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            points_in_cluster_i = np.where(labels == unique_labels[i])[0]
            points_in_cluster_j = np.where(labels == unique_labels[j])[0]
            intercluster_dist = np.min(distances[np.ix_(points_in_cluster_i, points_in_cluster_j)])
            if intercluster_dist < min_intercluster_dist:
                min_intercluster_dist = intercluster_dist
    
    # 计算类内最大距离
    max_intracluster_dist = -np.inf
    for i in range(n_clusters):
        points_in_cluster = np.where(labels == unique_labels[i])[0]
        intracluster_dist = np.max(distances[np.ix_(points_in_cluster, points_in_cluster)])
        if intracluster_dist > max_intracluster_dist:
            max_intracluster_dist = intracluster_dist

# X: 数据点矩阵 (n_samples, n_features)
# labels: 聚类后的标签 (n_samples, )
def calculate_davies_bouldin(X, labels):       # 越小越好
    return davies_bouldin_score(X, labels)

# 数据预处理
def dataProcess(content_lines):
    print("dataProcess() begin.")
    content_lines = msgs(content_lines)
    int_list = [[int(line[i:i + 2], 16) for i in range(0, len(line), 2)] for line in content_lines]
    # print("int_list")
    # lens = [len(item) for item in int_list]
    # 指定int_list输出文件的路径
    output_file_path = 'data\\testData\\int_list.txt'

    # 将二维整数列表写入文件
    with open(output_file_path, 'w') as file:
        for row in int_list:
            # 将每个整数转换为字符串，并用空格分隔，然后写入文件
            file.write(' '.join(map(str, row)) + '\n')
    print(f'二维整数列表已写入文件 {output_file_path}')
    print("dataProcess() end.")
    return int_list
    
# 提取最大频繁项集合
def extractMaxFrequentItemset(int_list, min_support):
    print("extractMaxFrequentItemset() begin.")
    frequentDic = apriori_all(int_list, min_support)
    frequentItemLength = 0
    frequentItem_list = list()
    # 打印频繁项模式
    for k, sequences in frequentDic.items():
        logging.info(f"frequent {k} item:")
        frequentItemLength += len(sequences)
        for item in sequences:
            logging.info(item)
            frequentItem_list.append(item)
    for item in frequentItem_list:
        logging.info(item)
    logging.info(frequentItemLength)
    # 合并子串
    for item1 in frequentItem_list:
        for item2 in frequentItem_list:
            # 如果item1是item2的子串
            if is_subString(item1[0], item2[0]) == True:
                item1[1] = 0
                break
    new_list = list()
    for item in frequentItem_list:
        if item[1] >= min_support:   # 只要长度大于1的最大频繁项
            new_list.append(item)
    logging.info("max frequent itemset")
    for item in new_list:
        logging.info(item)
    logging.info("The number of max frequent itemset: %d", len(new_list))
    logging.info(len(new_list))
    print("extractMaxFrequentItemset() end.")
    return new_list

# 协议格式聚类
def protocolFormatCLuster(maxFrequentItemSet):
    print("protocolFormatCLuster() begin.")
    membershipVector = []  # 二维列表，存储所有数据包的模糊隶属度向量
    itemVector = []  # 临时存储一个数据包的模糊隶属度向量，用完以后要进行清空处理
    # 初始化隶属度向量
    for i in int_list:
        # if len(itemVector) != len(frequentSet_and_support):
        #     print("error, 隶属度向量长度和最大频繁项列表长度不相等!")
        #     sys.exit()
        membershipVector.append([fuzzMembership(i, item[0]) for item in maxFrequentItemSet])
    
    # 打印隶属度向量
    # for item in membershipVector:
    #     print(item)
    
    lens_list = list()
    for item in membershipVector:
        lens_list.append(len(item))
    print("min: ", min(lens_list))
    print("max: ", max(lens_list))
    # 选择最优的类簇个数
    kmeansClusters1 = find_optimal_k(membershipVector, 2, kmeansClusters)
    logging.info("best cluster number: %d", kmeansClusters1)
    # 手动指定聚类中心
    # initial_centroids = [membershipVector[0], membershipVector[18], membershipVector[95]]
    # kmeans = KMeans(n_clusters = 10, init=initial_centroids)
    kmeans = KMeans(n_clusters = kmeansClusters1)
    logging.info("the number of cluster result: %d", kmeansClusters1)
    # # 随机聚类中心
    # kmeans = KMeans(n_clusters = 3)
    clusters = kmeans.fit_predict(membershipVector)
    # 将 clusters 转换为字符串形式，并用逗号间隔每个元素
    clusters_str = ', '.join(map(str, clusters))
    logging.info(clusters_str)
    num_samples = len(clusters)
    logging.info(num_samples)
    # 计算轮廓系数
    silhouette_avg = silhouette_score(membershipVector, clusters)
    logging.info("silhouette score: %f", silhouette_avg)

    return clusters

# 打印数据包与类簇的对应情况
def printPacketAndCLuster(kmeansClusters, packets, clusters):
    # 打印整个数据包的数据
    i = 0
    my_dict = {i: [] for i in range(0, kmeansClusters)}
    print(my_dict)

    i = 0
    for packet in packets:
        my_dict[clusters[i]].append([i, packet])
        i += 1
    # print(my_dict)
    print("数据包的个数: ", i)
    
    # 打开文件 'example.txt'，使用写入模式 ('w')，会覆盖已有内容
    with open('./output/example.txt', 'w') as file:
        # 向文件中写入内容
        for key in my_dict:
            file.write("cluster" + str(key) + ":\n")
            for value in my_dict[key]:
                
                # hex_str = hexdump(value[1], dump=True)
                # 提取原始数据
                packet_data = raw(value[1])

                # 将原始数据转换为十六进制
                hex_str = packet_data.hex()
                            
                file.write("    Packet Seq: " + "{:6}".format(value[0] + 1) + ", " + hex_str + '\n')
        
    file.close()
    
    
    
    
    # # 只打印应用层数据
    # i = 0
    # my_dict = {i: [] for i in range(0, kmeansClusters)}
    # print(my_dict)

    # i = 0
    # for packet in packets:
    #     my_dict[clusters[i]].append([i, packet])
    #     i += 1
    # # print(my_dict)

    # # 打开文件 'example.txt'，使用写入模式 ('w')，会覆盖已有内容
    # with open('example.txt', 'w') as file:
    #     # 向文件中写入内容
    #     for key in my_dict:
    #         file.write("cluster" + str(key) + ":\n")
    #         for value in my_dict[key]:
    #             # 提取应用层数据（这里假设你使用 Scapy 处理数据包）
    #             if value[1].haslayer('Raw'):  # 检查是否有应用层（原始）数据
    #                 app_layer_data = value[1]['Raw'].load  # 提取应用层负载

    #                 # 将应用层数据转换为十六进制字符串
    #                 hex_str = app_layer_data.hex()

    #                 file.write("    Packet Seq: " + "{:6}".format(value[0] + 1) + ", " + hex_str + '\n')
    #             else:
    #                 file.write("    Packet Seq: " + "{:6}".format(value[0] + 1) + ", " + '\n')
            
    # file.close()

    return 0

# 会话聚类
def sessionCluster(packets, session_cluster):
    # 会话切分
    # 存储会话的字典
    sessions = {}
    # 遍历所有数据包并标号
    for index, packet in enumerate(packets, start=1):
        # print("循环")
        if IP in packet:
            # print("if IP in packet")
            # 判断是否为TCP或UDP
            proto = None
            if TCP in packet:
                proto = 'TCP'
            elif UDP in packet:
                proto = 'UDP'
            if proto:
                # print("protocol:", proto)
                # 获取五元组
                src = packet[IP].src
                dst = packet[IP].dst
                sport = packet[proto].sport
                dport = packet[proto].dport
                # 会话标识符 (可以使用 tuple 或其他标识方法)
                session_key = (src, dst, sport, dport, proto)    # key
                reverse_session_key = (dst, src, dport, sport, proto)
                # 确定会话并添加数据包索引
                if session_key in sessions:
                    sessions[session_key].add_packet_index(index)
                elif reverse_session_key in sessions:
                    sessions[reverse_session_key].add_packet_index(index)
                else:
                    # print("新建一个会话.")
                    # 新建一个会话
                    new_session = Session(src, dst, sport, dport, proto)
                    new_session.add_packet_index(index)
                    sessions[session_key] = new_session
        # else: print("没有IP")
        
        
    # print("输出每个会话的信息")
    # # 输出每个会话的信息
    # for session_key, session in sessions.items():
    #     info = session.get_info()
    #     print(f"会话 {session_key}:")
    #     print(f"  源IP: {info['src_ip']}, 目的IP: {info['dst_ip']}")
    #     print(f"  源端口: {info['src_port']}, 目的端口: {info['dst_port']}")
    #     print(f"  协议: {info['protocol']}")
    #     print(f"  数据包索引: {info['packet_indices']}")
    #     print()
        
   
    # 根据协议格式聚类结果向每一个会话里面贴标签
    # 对所有的session进行遍历，将得到的类簇序号添加到每一个数据包序号的后面
    packet_number = 1   # 数据包序号
    flag = 0  # 标志位，初始值为0，如果找到了对应的序号并设置完毕，则将flag设置为1，主循环看到1则break，再将flag设置为0
    for item in clusters:
        # 遍历每一个session，找包序号
        for session_key, session in sessions.items():
            for index in session.packet_indices:
                if index[0] == packet_number:
                    index[1] =  item
                    flag = 1  # 标志位置1
                    break
                
            if flag == 1:
                flag = 0
                break
        
        packet_number += 1  # 序号自增

    # print("输出每个会话的信息")
    # # 输出每个会话的信息
    # for session_key, session in sessions.items():
    #     info = session.get_info()
    #     print(f"会话 {session_key}:")
    #     print(f"  源IP: {info['src_ip']}, 目的IP: {info['dst_ip']}")
    #     print(f"  源端口: {info['src_port']}, 目的端口: {info['dst_port']}")
    #     print(f"  协议: {info['protocol']}")
    #     print(f"  数据包索引: {info['packet_indices']}")
    #     print()
    # print("输出每个会话的信息")

    # 将每个会话中的类簇号作为一个列表提出来（不要包序号了），所有会话组成一个二维列表
    cluster_seqs = []
    for session_key, session in sessions.items():
        item_list = []  # 临时变量，用于存储cluster序列
        for index in session.packet_indices:
            item_list.append(index[1])
        cluster_seqs.append(item_list)
    logging.info("clusterSequence: %s",cluster_seqs)
    # 使用K-Medoids进行聚类,将NW算法进行序列对齐之后的相同位数作为二者的距离
    labels, medoids = kmedoids(cluster_seqs, session_cluster, needleman_distance)
    session_protocol_clusters = {}
    # 先根据标签数量创建一个字典
    for i in range(0, session_cluster):
        session_protocol_clusters[i] = []
    logging.info(session_protocol_clusters)
    i = 0
    for item in labels:
        session_protocol_clusters[item].append(cluster_seqs[i])
        i += 1
    logging.info(session_protocol_clusters)
    return session_protocol_clusters
    
# # 状态机推断
# def protocolStateMachineInference(session_protocol_clusters, fileName):
    
#     num = 1
    
#     # 在这个字典的每一个元素内进行概率状态机构建
#     for value in session_protocol_clusters.values():
#         # 构建概率自动状态机
#         psm = ProbabilisticStateMachine()
#         # 添加序列到状态机
#         for seq in value:
#             psm.add_sequence(seq)

#         # 计算转移概率
#         psm.compute_probabilities()
#         psm.compute_total_probabilities()

#         logging.info("\n")
#         logging.info(f"Protocol {num}:")
        
#         # 打印转移概率
#         for state, transitions in psm.transition_probabilities.items():
#             logging.info(f"Protocol Format Cluster {state}:")
#             for next_state, probability in transitions.items():    
#                 logging.info(f"    -> {next_state} with probability {probability[0]:.2f} total probability {probability[1]:.2f}")

#         num = num + 13
        
        # 示例：生成一个从状态1开始、长度为10的状态序列
        # generated_sequence = psm.generate_sequence(start_state=1, length=10)
        # print("根据概率生成可能序列:", generated_sequence)
        # print("\n\n\n")
        


# 状态机推断函数
def protocolStateMachineInference(session_protocol_clusters, file_name):
    num = 1

    # 打开文件以写入
    with open(file_name, "w") as file:
        # 在这个字典的每一个元素内进行概率状态机构建
        for value in session_protocol_clusters.values():
            # 构建概率自动状态机
            psm = ProbabilisticStateMachine()
            
            # 添加序列到状态机
            for seq in value:
                psm.add_sequence(seq)

            # 计算转移概率
            psm.compute_probabilities()
            psm.compute_total_probabilities()

            # 输出到控制台和文件
            print("\n", file=file)
            print("\n")

            print(f"Protocol {num}:", file=file)
            print(f"Protocol {num}:")

            # # 打印转移概率
            # for state, transitions in psm.transition_probabilities.items():
                
            #     print(f"Protocol Format Cluster {state}:", file=file)
            #     print(f"Protocol Format Cluster {state}:")
            #     for next_state, probability in transitions.items():
            #         line = (
            #             f"    -> {next_state} with probability {probability[0]:.2f} "
            #             f"total probability {probability[1]:.2f}"
            #         )
            #         print(line, file=file)
            #         print(line)
            
            # 打印转移概率
            for state, transitions in psm.transition_probabilities.items():
                
                # 仅打印两个概率均不为0的状态迁移
                valid_transitions = [
                    (next_state, probability)
                    for next_state, probability in transitions.items()
                    if probability[0] != 0 and probability[1] != 0
                ]

                if valid_transitions:
                    print(f"Protocol Format Cluster {state}:", file=file)
                    print(f"Protocol Format Cluster {state}:")

                    for next_state, probability in valid_transitions:
                        line = (
                            f"    -> {next_state} with probability {probability[0]:.2f} "
                            f"total probability {probability[1]:.2f}"
                        )
                        print(line, file=file)
                        print(line)

            # 更新协议编号
            num += 1
        
        
        

# 读取pcapng文件
def read_pcapng_to_hex(file_name, output_file):
    packets = rdpcap(file_name)  # 使用scapy读取pcapng文件, 读出一个packet格式的列表
    
    print(packets)

    # 打开输出文件
    with open(output_file, 'w') as f:
        # 遍历每个数据包
        for packet in packets:
            # 将数据包转换为十六进制字符串
            hex_dump = ''.join('%02X' % x for x in bytes(packet))     # join的作用是将一个序列（如列表、元组或字符串）中的元素连接成一个字符串，并且可以在这些元素之间插入一个指定的分隔符，前面的''里面可以写分隔符。
            # 将十六进制字符串写入文件，每个数据包一行
            f.write(hex_dump + '\n')

# 提取应用层txt文件
def extractApplicationMessages(packets, fileName):
    # 只保留应用层
    # 创建一个新的空列表来存储只保留应用层数据的包
    app_layer_packets = []
    for packet in packets:
        # 获取应用层数据
        if Raw in packet:
            app_layer_data = packet[Raw]
            app_layer_packets.append(app_layer_data)
    
    # 将只保留应用层数据的包写入新的 pcapng 文件
    wrpcap("./output/tmp.pcap", app_layer_packets)
    
    # 检查pcapng文件是否存在
    if not os.path.isfile("./output/tmp.pcap"):
        logging.error("Error: ./output/tmp.pcap does not exist.")
        exit()
    else: 
        logging.info("Successfully extracted application layer data!")
    
    # 读取pcapng文件并转换为hex格式存储
    read_pcapng_to_hex("./output/tmp.pcap", fileName)
    logging.info(f"Conversion completed. Output written to {fileName}")


if __name__ == "__main__":
    start_time = time.time()  # 获取开始时间
    
    # 配置日志
    logging.basicConfig(
        filename='./output/debug.log',  # 指定日志文件名
        level=logging.INFO,    # 设置日志级别为DEBUG, INFO
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 读取原始pcap文件
    packets = rdpcap('./data/dataTransition_test60.pcapng')
    # 打印数据包数量
    print(f"数据包数量: {len(packets)}")
    
    # 提取应用层信息
    extractApplicationMessages(packets, "./output/hex.txt")
    
    # 打开txt文件
    with open('./output/hex.txt', 'r') as file:     # 将open('output.txt', 'r')返回的对象赋值给file
        content_lines = file.readlines()
    
    # 参数设置
    min_support = 0.24         # 最小支持度
    kmeansClusters = 9        # k-means算法的类簇个数
    session_cluster = 1       # 会话聚类类簇个数
    
    # 数据预处理，将数据集转化为int类型的二位链表
    int_list = dataProcess(content_lines)
    # 提取最大频繁项集合
    maxFrequentItemSet = extractMaxFrequentItemset(int_list, min_support)
    # 协议格式聚类
    clusters = protocolFormatCLuster(maxFrequentItemSet)
    # 打印类簇号与数据包的对应情况，输出到output/example.txt中
    printPacketAndCLuster(kmeansClusters, packets, clusters)
    # 会话聚类
    session_protocol_clusters = sessionCluster(packets, session_cluster)
    # 协议状态机构建
    protocolStateMachineInference(session_protocol_clusters, "./output/result.txt")
    
    end_time = time.time()  # 获取结束时间
    print("程序执行时间：{}秒".format(end_time - start_time))
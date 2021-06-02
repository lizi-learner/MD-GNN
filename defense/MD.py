import scipy.sparse as sp
import numpy as np
import networkx as nx


def MD(features, adj, metric='order', threshold=0):
    # 判断metric是否合法
    metric_list = ['Cfs', 'Cfs1', 'Cfs2', 'Cfs3', 'Cfs4', 'Cs', 'Cs1', 'Jaccard1']
    if not (metric in metric_list):
        print("Error: metric is illegal!")
        return

    print('deleting edges...')
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)

    graph = nx.from_numpy_matrix(adj.A)

    # 取出稀疏矩阵上三角部分的元素
    adj_triu = sp.triu(adj, format='csr')
    removed_cnt = 0
    # 选择标准
    # 特征和结构结合
    if metric == 'Cfs':  # jaccard + cn, 也是原方案，其他都是对比方案
        modified_adj, removed_cnt = dropedge_order_jaccard(adj, adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=threshold)
    if metric == 'Cfs1':  # cosine + cn
        modified_adj, removed_cnt = dropedge_order_cosine(adj, adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=threshold)
    if metric == 'Cfs2':  # Euclidean + cn
        modified_adj, removed_cnt = dropedge_order_Euclidean(adj, adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=threshold)
    if metric == 'Cfs3':  # Manhattan + cn
        modified_adj, removed_cnt = dropedge_order_Manhattan(adj, adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=threshold)
    if metric == 'Cfs4':  # 将不产生单点的迭代条件换成了不生成不连通图 + jaccard + cn，效果相对于原方案没什么提升
        modified_adj, removed_cnt = dropedge_order_jaccard1(adj, graph, adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=threshold)

    # 只有特征或只有结构
    if metric == 'Cs':  # cn
        modified_adj, removed_cnt = dropedge_order(adj, adj_triu.data, adj_triu.indptr, adj_triu.indices)
    if metric == 'Cs1':  # cn with single nodes
        modified_adj, removed_cnt = dropedge_order1(adj, adj_triu.data, adj_triu.indptr, adj_triu.indices)
    if metric == 'Jaccard1':  # jaccard without single nodes
        modified_adj, removed_cnt = dropedge_jaccard(adj, adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=threshold)

    print('removed %s edges in the original graph' % removed_cnt)
    return modified_adj

def dropedge_order(adj_triu, A, iA, jA):

    removed_cnt = 0
    degrees = adj_triu.A.sum(0)
    S = np.dot(adj_triu.A, adj_triu.A) - np.diag(degrees)

    adj_triu1 = adj_triu.A
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):

            n1 = row
            n2 = jA[i]

            order = S[n1][n2]

            if order == 0:
                if degrees[n1] != 1 and degrees[n2] != 1:
                    adj_triu1[n1][n2] = 0
                    degrees[n1] -= 1
                    degrees[n2] -= 1
                    removed_cnt += 1


    adj_triu1 = sp.csr_matrix(adj_triu1)
    adj_triu1 = sp.triu(adj_triu1, format='csr')

    modified_adj = adj_triu1 + adj_triu1.transpose()

    return modified_adj, removed_cnt


def dropedge_order_cosine(adj_triu, A, iA, jA, features, threshold):
    removed_cnt = 0
    degrees = adj_triu.A.sum(0)
    S = np.dot(adj_triu.A, adj_triu.A) - np.diag(degrees)

    l1 = []
    l2 = []
    score = []
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):

            n1 = row
            n2 = jA[i]

            order = S[n1][n2]

            if order == 0:
                if degrees[n1] != 1 and degrees[n2] != 1:
                    a, b = features[n1], features[n2]
                    intersection = a.multiply(b).count_nonzero()
                    C = intersection * 1.0 / np.sqrt((a.count_nonzero() * b.count_nonzero()))
                    if threshold == 0:
                        l1.append(n1)
                        l2.append(n2)

                        score.append(C)
                        removed_cnt += 1
                    elif C < threshold:
                        l1.append(n1)
                        l2.append(n2)

                        score.append(C)
                        removed_cnt += 1

    score = np.array(score)
    adj_triu1 = adj_triu.A
    cnt = 0
    for i in range(removed_cnt):

        max_index = np.argmin(score)

        if degrees[l1[max_index]] != 1 and degrees[l2[max_index]] != 1:
            adj_triu1[l1[max_index]][l2[max_index]] = 0

            degrees[l1[max_index]] -= 1
            degrees[l2[max_index]] -= 1
            cnt += 1
        score[max_index] = 100

    adj_triu1 = sp.csr_matrix(adj_triu1)
    adj_triu1 = sp.triu(adj_triu1, format='csr')
    modified_adj = adj_triu1 + adj_triu1.transpose()

    return modified_adj, cnt

def dropedge_order_Euclidean(adj_triu, A, iA, jA, features, threshold):
    removed_cnt = 0
    degrees = adj_triu.A.sum(0)
    S = np.dot(adj_triu.A, adj_triu.A) - np.diag(degrees)

    l1 = []
    l2 = []
    score = []
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):

            n1 = row
            n2 = jA[i]

            order = S[n1][n2]

            if order == 0:
                if degrees[n1] != 1 and degrees[n2] != 1:
                    a, b = features.A[n1], features.A[n2]
                    C = np.linalg.norm(a - b, 2)

                    if threshold == 0:
                        l1.append(n1)
                        l2.append(n2)

                        score.append(C)
                        removed_cnt += 1
                    elif C < threshold:
                        l1.append(n1)
                        l2.append(n2)

                        score.append(C)
                        removed_cnt += 1

    score = np.array(score)
    adj_triu1 = adj_triu.A
    cnt = 0
    for i in range(removed_cnt):

        max_index = np.argmin(score)

        if degrees[l1[max_index]] != 1 and degrees[l2[max_index]] != 1:
            adj_triu1[l1[max_index]][l2[max_index]] = 0

            degrees[l1[max_index]] -= 1
            degrees[l2[max_index]] -= 1
            cnt += 1
        score[max_index] = 100

    adj_triu1 = sp.csr_matrix(adj_triu1)
    adj_triu1 = sp.triu(adj_triu1, format='csr')
    modified_adj = adj_triu1 + adj_triu1.transpose()

    return modified_adj, cnt

def dropedge_order_Manhattan(adj_triu, A, iA, jA, features, threshold):
    removed_cnt = 0
    degrees = adj_triu.A.sum(0)
    S = np.dot(adj_triu.A, adj_triu.A) - np.diag(degrees)

    l1 = []
    l2 = []
    score = []
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):

            n1 = row
            n2 = jA[i]

            order = S[n1][n2]

            if order == 0:
                if degrees[n1] != 1 and degrees[n2] != 1:
                    a, b = features[n1], features[n2]
                    intersection = a.multiply(b).count_nonzero()
                    C = a.count_nonzero() + b.count_nonzero() - intersection * 2

                    if threshold == 0:
                        l1.append(n1)
                        l2.append(n2)

                        score.append(C)
                        removed_cnt += 1
                    elif C < threshold:
                        l1.append(n1)
                        l2.append(n2)

                        score.append(C)
                        removed_cnt += 1

    score = np.array(score)
    adj_triu1 = adj_triu.A
    cnt = 0
    for i in range(removed_cnt):

        max_index = np.argmin(score)

        if degrees[l1[max_index]] != 1 and degrees[l2[max_index]] != 1:
            adj_triu1[l1[max_index]][l2[max_index]] = 0

            degrees[l1[max_index]] -= 1
            degrees[l2[max_index]] -= 1
            cnt += 1
        score[max_index] = 100

    adj_triu1 = sp.csr_matrix(adj_triu1)
    adj_triu1 = sp.triu(adj_triu1, format='csr')
    modified_adj = adj_triu1 + adj_triu1.transpose()

    return modified_adj, cnt


def dropedge_order_jaccard(adj_triu, A, iA, jA, features, threshold):
    removed_cnt = 0
    degrees = adj_triu.A.sum(0)
    S = np.dot(adj_triu.A, adj_triu.A) - np.diag(degrees)

    l1 = []
    l2 = []
    score = []
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):

            n1 = row
            n2 = jA[i]

            order = S[n1][n2]

            if order == 0:
                if degrees[n1] != 1 and degrees[n2] != 1:
                    a, b = features[n1], features[n2]
                    intersection = a.multiply(b).count_nonzero()
                    J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
                    if threshold == 0:
                        l1.append(n1)
                        l2.append(n2)

                        score.append(J)
                        removed_cnt += 1
                    elif J < threshold:
                        l1.append(n1)
                        l2.append(n2)

                        score.append(J)
                        removed_cnt += 1

    score = np.array(score)
    adj_triu1 = adj_triu.A
    cnt = 0
    for i in range(removed_cnt):

        max_index = np.argmin(score)

        if degrees[l1[max_index]] != 1 and degrees[l2[max_index]] != 1:
            adj_triu1[l1[max_index]][l2[max_index]] = 0

            degrees[l1[max_index]] -= 1
            degrees[l2[max_index]] -= 1
            cnt += 1
        score[max_index] = 100

    adj_triu1 = sp.csr_matrix(adj_triu1)
    adj_triu1 = sp.triu(adj_triu1, format='csr')
    modified_adj = adj_triu1 + adj_triu1.transpose()

    return modified_adj, cnt

def dropedge_order_jaccard1(adj_triu, graph, A, iA, jA, features, threshold):
    removed_cnt = 0
    degrees = adj_triu.A.sum(0)
    S = np.dot(adj_triu.A, adj_triu.A) - np.diag(degrees)


    l1 = []
    l2 = []
    score = []
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):

            n1 = row
            n2 = jA[i]

            order = S[n1][n2]

            if order == 0:
                if degrees[n1] != 1 and degrees[n2] != 1:
                    a, b = features[n1], features[n2]
                    intersection = a.multiply(b).count_nonzero()
                    J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
                    if threshold == 0:
                        l1.append(n1)
                        l2.append(n2)

                        score.append(J)
                        removed_cnt += 1
                    elif J < threshold:
                        l1.append(n1)
                        l2.append(n2)

                        score.append(J)
                        removed_cnt += 1

    score = np.array(score)
    adj_triu1 = adj_triu.A
    cnt = 0
    for i in range(removed_cnt):

        max_index = np.argmin(score)

        graph.remove_edge(l1[max_index], l2[max_index])

        if nx.has_path(graph,l1[max_index],l2[max_index]):
            adj_triu1[l1[max_index]][l2[max_index]] = 0
            cnt += 1
        else:
            graph.add_edge(l1[max_index], l2[max_index])
        score[max_index] = 100

    adj_triu1 = sp.csr_matrix(adj_triu1)
    adj_triu1 = sp.triu(adj_triu1, format='csr')
    modified_adj = adj_triu1 + adj_triu1.transpose()

    return modified_adj, cnt


def dropedge_order1(adj_triu, A, iA, jA):
    # 不考虑单个点的情况
    removed_cnt = 0
    degrees = adj_triu.A.sum(0)
    S = np.dot(adj_triu.A, adj_triu.A) - np.diag(degrees)

    adj_triu1 = adj_triu.A
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):

            n1 = row
            n2 = jA[i]

            order = S[n1][n2]

            if order == 0:
                adj_triu1[n1][n2] = 0
                removed_cnt += 1

    adj_triu1 = sp.csr_matrix(adj_triu1)
    adj_triu1 = sp.triu(adj_triu1, format='csr')

    modified_adj = adj_triu1 + adj_triu1.transpose()

    return modified_adj, removed_cnt


def dropedge_jaccard(adj_triu, A, iA, jA, features, threshold = 0.03):

    removed_cnt = 0
    degrees = adj_triu.A.sum(0)

    l1 = []
    l2 = []
    score = []
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]

            a, b = features[n1], features[n2]
            intersection = a.multiply(b).count_nonzero()
            J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)

            if J < threshold and degrees[n1] != 1 and degrees[n2] != 1:
                l1.append(n1)
                l2.append(n2)
                score.append(J)
                removed_cnt += 1
    print('removed_cnt: {}'.format(removed_cnt))
    score = np.array(score)
    adj_triu1 = adj_triu.A
    cnt = 0
    for i in range(removed_cnt):
        # 若去掉边不导致独立点，去掉分数最低的边
        max_index = np.argmin(score)
        if degrees[l1[max_index]] != 1 and degrees[l2[max_index]] != 1:
            adj_triu1[l1[max_index]][l2[max_index]] = 0
            # print(np.nonzero(np.array(adj_triu1))[0].size)
            degrees[l1[max_index]] -= 1
            degrees[l2[max_index]] -= 1
            cnt += 1
        score[max_index] = 100

    adj_triu1 = sp.csr_matrix(adj_triu1)
    adj_triu1 = sp.triu(adj_triu1, format='csr')
    modified_adj = adj_triu1 + adj_triu1.transpose()
    return modified_adj, cnt


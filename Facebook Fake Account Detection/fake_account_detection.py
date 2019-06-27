import json
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import copy

def SybilWalk(neighbors_list,benigns,sybils,eps=1e-3,T=200,proba_init = None):
    neighbors_list = copy.deepcopy(neighbors_list)
    # Chuẩn hóa để cho trọng số cạnh lớn nhất bằng 1
    w_max = 0
    for u in neighbors_list:
        for w in neighbors_list[u].values():
            if w>w_max:
                w_max=w
    # print(w_max)
    for u in neighbors_list:
        for v in neighbors_list[u]:
            neighbors_list[u][v] = neighbors_list[u][v]/w_max
    # Khởi tạo điểm xác suất ban đầu
    # p = {k:0.5 for k in neighbors_list}
    p = {}
    for u in neighbors_list:
    	p[u] = 0.5
    # Điểm xác suất SVM
    if proba_init != None:
        p.update(proba_init)

    # Thêm 2 điểm ls và lb
    p['ls'] = 1
    p['lb'] = 0
    for b in benigns:
        p[b] = 0
        neighbors_list[b]['lb'] = 1
    for s in sybils:
        p[s] = 1
        neighbors_list[s]['ls'] = 1
    
    # Tính tổng các trọng số cạnh của từng nốt
    d = {}
    for u in neighbors_list:
        d[u] = sum(neighbors_list[u].values())
    t = 1
    while t<T:
        p_old = p.copy()

        # Cap nhat diem xac suat
        for u in neighbors_list:
            if len(neighbors_list[u])>0:
                # p_u = 0
                # for v in neighbors_list[u]:
                #     p_u = p_u + neighbors_list[u][v]/d[u]*p_old[v]
                p[u] = sum(neighbors_list[u][v]/d[u]*p_old[v] for v in neighbors_list[u])
                # p[u] = p_u

        # Tinh sai so
        e = sum((p_old[u]-p[u])**2 for u in p)
        # for u in p:
        #     e +=  (p_old[u]-p[u])**2
        if e<eps:
            break
        t += 1
    print('Số vòng lặp:',t)
    return p

# Hàm xây dựng đồ thị kết nối giữa các tài khoản với đỉnh là tài khoản,
# cạnh nối giữa 2 đỉnh là số lượng bạn chung chia cho số lượng bạn bè lớn nhất
def build_graph(users):
    # Đọc danh sách bạn bè của các tài khoản
    friend_list = {}
    for user in users:
        file = 'friends/{0}.json'.format(user)
        friends = json.load(open(file))
        friend_list[str(user)] = set([f['id'] for f in friends])

    # Xây dựng đồ thị
    neighbors_list = {}
    for v in friend_list:
        neighbors_list[v] = {}
        for e in friend_list:
            # Tính trọng số cạch nối giữa các tài khoản
            w = len(friend_list[v] & friend_list[e]) / max(len(friend_list[v]), len(friend_list[e]), 1)
            if w > 0 and v != e:
                neighbors_list[v][e] = w
    return neighbors_list

if __name__ == '__main__':
    print("read data ...")
    data = pd.read_csv('data.csv')
    Y = np.floor(data['Sybil'])
    X_raw = data[['num_comment', 'num_comment_on_own_post', 'num_friend', 'num_join_group', 'num_like_on_own_post',
                  'num_own_post', 'num_post_on_wall', 'num_reaction', 'num_share_on_own_post', 'num_tagged_post',
                  'num_used_tags_commnet', 'num_user_tags', 'num_user_tags_comment']]

    ids = data['id']
    X_fillna = X_raw.fillna(np.mean(X_raw))
    X = np.log1p(X_fillna)

    X_train = X[data['is_train'] == True]
    X_test = X[data['is_train'] == False]
    y_train = Y[data['is_train'] == True]
    y_test = Y[data['is_train'] == False]
    ids_train = ids[data['is_train'] == True]
    ids_test = ids[data['is_train'] == False]

    model = SVC(probability=True)
    model.fit(X_train, y_train)
    print('svm train accuracy', model.score(X_train, y_train))
    print('svm test accuracy', model.score(X_test, y_test))
    y_pred = model.predict(X_test)

    print('Kết quả thuật toán SVM')
    print(classification_report(y_test, y_pred))

    sybils = []
    benigns = []
    for u, l in zip(ids_train, y_train):
        # for u,l in zip(ids,Y):
        if l == 1:
            sybils.append(str(u))
        else:
            benigns.append(str(u))
    print(len(sybils), len(benigns))
    
    # Xây dựng đồ thị
    # neighbors_list = build_graph(data['id'])
    # json.dump(neighbors_list,open('neighbors_list.json','w',encoding='utf-8'))
    neighbors_list = json.load(open('neighbors_list.json'))

    # Thực hiện thuật toán SybilWalk ko dùng SVM
    print('Kết quả phương pháp đồ thị')
    proba = SybilWalk(neighbors_list, benigns, sybils, eps=1e-10, T=200)

    y_pred_sybil = []
    for u, l in zip(ids_test, y_test):
        y_pred_sybil.append(0 if proba[str(u)] < 0.5 else 1)
    
    print(classification_report(y_test, y_pred_sybil))
    
    # Thực hiện thuật toán SybilWalk sử dụng SVM để khởi tạo điểm xác suất
    print('Kết quả phương pháp kết hợp cả đặc trưng và đồ thị')
    y_proba = model.predict_proba(X_test)
    proba_init = {str(k): v[1] for k, v in zip(ids_test, y_proba)}
    proba = SybilWalk(neighbors_list, benigns, sybils, eps=1e-10, T=200,
                      proba_init=proba_init)

    y_pred_sybil = []
    for u, l in zip(ids_test, y_test):
        y_pred_sybil.append(0 if proba[str(u)] < 0.35 else 1)

    print(classification_report(y_test, y_pred_sybil))

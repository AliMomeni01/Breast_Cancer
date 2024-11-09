from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier  
bc = load_breast_cancer()
#print (bc.DESCR)
#print (bc.target[500])
#print(bc.target.shape)
#print(bc.data[0])
#print(bc.data.shape)
x_train,x_test,y_train,y_test = train_test_split(bc.data, bc.target, test_size= 0.2)
#print(f"Feature => Train: {x_train.shape} - Test: {x_test.shape}")
#print(f"Label => Train: {y_train.shape} - Test: {y_test.shape}")
scaler = MinMaxScaler (feature_range= (0,1) )
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#print(x_train[0])
def calculate_metrics(y_train,y_test,y_pred_train,y_pred_test):
    acc_train = accuracy_score(y_true= y_train, y_pred= y_pred_train)
    acc_test  = accuracy_score(y_true= y_test, y_pred= y_pred_test)
    p = precision_score(y_true= y_test, y_pred= y_pred_test)
    r = recall_score(y_true= y_test, y_pred= y_pred_test)
    print(f"acc_train: {acc_train} - acc_test: {acc_test} - precision: {p} - recall: {r}")
    return acc_train, acc_test, p, r

gnb = GaussianNB()
gnb.fit(x_train,y_train)
#print(gnb)
y_pred_train = gnb.predict(x_train)
y_pred_test = gnb.predict(x_test)
acc_train_gnb,acc_test_gnb,p_gnb,r_gnb= calculate_metrics(y_train,y_test,y_pred_train,y_pred_test)

knn = KNeighborsClassifier(n_neighbors= 8 , algorithm= 'kd_tree', leaf_size= 28)
knn.fit(x_train,y_train)
#print(knn)
y_pred_train = knn.predict(x_train)
y_pred_test  = knn.predict(x_test)
acc_train_knn,acc_test_knn,p_knn,r_knn = calculate_metrics(y_train,y_test,y_pred_train,y_pred_test) 

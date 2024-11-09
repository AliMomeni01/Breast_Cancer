from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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

dt = DecisionTreeClassifier(criterion= 'entropy', max_depth= 114, min_samples_split=11)
dt.fit(x_train,y_train)
#print(dt)
y_pred_train = dt.predict(x_train)
y_pred_test = dt.predict(x_test)
acc_train_dt,acc_test_dt,p_dt,r_dt = calculate_metrics(y_train,y_test,y_pred_train,y_pred_test)

rf = RandomForestClassifier(n_estimators=1000, max_depth=32, min_samples_split=4)
rf.fit(x_train,y_train)
#print(rf)
y_pred_train = rf.predict(x_train)
y_pred_test = rf.predict(x_test)
acc_train_rf,acc_test_rf,p_rf,r_rf = calculate_metrics(y_train,y_test,y_pred_train,y_pred_test)

svm = SVC()
svm.fit(x_train,y_train)
#print(svm)
y_pred_train = svm.predict(x_train)
y_pred_test = svm.predict(x_test)
acc_train_svm,acc_test_svm,p_svm,r_svm = calculate_metrics(y_train,y_test,y_pred_train,y_pred_test)

lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred_train = lr.predict(x_train)
y_pred_test = lr.predict(x_test)
acc_train_lr,acc_test_lr,p_lr,r_lr = calculate_metrics(y_train,y_test,y_pred_train,y_pred_test)

ann = MLPClassifier(hidden_layer_sizes= 100, activation= "relu", solver= "adam")
ann.fit(x_train,y_train)
y_pred_train = ann.predict(x_train)
y_pred_test = ann.predict(x_test)
acc_train_ann,acc_test_ann,p_ann,r_ann = calculate_metrics(y_train,y_test,y_pred_train,y_pred_test) 
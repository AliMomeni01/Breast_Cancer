from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
bc = load_breast_cancer()
#print (bc.DESCR)
#print (bc.target[500])
#print(bc.target.shape)
#print(bc.data[0])
#print(bc.data.shape)
x_train,x_test,y_train,y_test = train_test_split(bc.data, bc.target, test_size= 0.2)
print(f"Feature => Train: {x_train.shape} - Test: {x_test.shape}")
print(f"Label => Train: {y_train.shape} - Test: {y_test.shape}")
scaler = MinMaxScaler (feature_range= (0,1) )
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train[0])

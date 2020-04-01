import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle

def average(y_pred):
    for i in range(len(y_pred)):
        if i % len(y_pred) == 0 or (i+1) % len(y_pred) == 0:
            pass
        else:
            average = float(y_pred[i-1] +  y_pred[i] + y_pred[i+1])/3
            if average >= 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
    return y_pred

df = pd.read_csv('feature_labels.csv')
df = df.drop(["Unnamed: 0"],axis=1)
df_alert = df[df["Y"]==0]
#df_alert_1 = df.iloc[0::360, :]

#Reordering the columns
df_means = df_alert[["EAR","MAR","Circularity","MOE"]].mean()
df_std = df_alert[["EAR","MAR","Circularity","MOE"]].std()


df["EAR_N"] = (df["EAR"]-df_means["EAR"])/ df_std["EAR"]
df["MAR_N"] = (df["MAR"]-df_means["MAR"])/ df_std["MAR"]
df["Circularity_N"] = (df["Circularity"]-df_means["Circularity"])/ df_std["Circularity"]
df["MOE_N"] = (df["MOE"]-df_means["MOE"])/ df_std["MOE"]
df.to_csv('totalwithmaininfo.csv',index=False)
#df = df.drop(df.columns[0],axis=1)
X = df[["EAR","MAR","Circularity","MOE","EAR_N","MAR_N","Circularity_N","MOE_N"]]
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
#print(X_train)
#print(y_train)
#-------------------------------------- MODEL -------------------------------

acc3_list = []
f1_score3_list = []
roc_3_list = []
for i in range(1,30):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)
    pred_KN = neigh.predict(X_test)
    pred_KN = average(pred_KN)
    y_score_3 = neigh.predict_proba(X_test)[:,1]
    acc3_list.append(accuracy_score(y_test, pred_KN))
    f1_score3_list.append(metrics.f1_score(y_test, pred_KN))
    roc_3_list.append(metrics.roc_auc_score(y_test, y_score_3))

print(acc3_list)
print(acc3_list.index(max(acc3_list)))
neigh = KNeighborsClassifier(n_neighbors=acc3_list.index(max(acc3_list))+1)
neigh.fit(X_train, y_train)
pred_KN = neigh.predict(X_test)
pred_KN = average(pred_KN)
y_score_3 = neigh.predict_proba(X_test)[:,1]
acc3 = accuracy_score(y_test, pred_KN)
f1_score_3 = metrics.f1_score(y_test, pred_KN)
roc_3 = metrics.roc_auc_score(y_test, y_score_3)
print([acc3,f1_score_3,roc_3])
print(confusion_matrix(y_test, pred_KN))
filename = 'knn.pkl'
pickle.dump(neigh, open(filename, 'wb'))

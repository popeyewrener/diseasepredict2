# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# %%


d1= pd.read_csv('training.csv')

# %%
d1.head()

# %%
d1.describe()

# %%
d1.shape

# %%
d1.isnull().sum()

# %%
name_value=(d1['prognosis'].unique())
print(name_value)

# %%



# %%
X=d1.iloc[:,:-1]
y=d1.iloc[:,-1:]
encoder=LabelEncoder()

y['prognosis']=encoder.fit_transform(y['prognosis'])
code_value=y['prognosis'].unique()
name_value=sorted(name_value)

m=[x  for x in range(len(name_value))]
name_maper={m[i]:name_value[i] for i in range(len(name_value))}


# %%
X_train, X_test, y_train, y_test= train_test_split(X,y,random_state=21,test_size=0.2,shuffle=True)

# %%
logreg = LogisticRegression()
finaltestedmodel=logreg.fit(X_train, y_train)
Y_predLR = logreg.predict(X_test)
print("Train Accuracy: ",round(accuracy_score(y_train, logreg.predict(X_train))*100,2))
print("Test Accuracy: ",round(accuracy_score(y_test, Y_predLR) * 100,2))

# %%
y_test.value_counts().sum()
print(type(Y_predLR))
df = pd.DataFrame(Y_predLR, columns = ['Result'])
print(type(df))

# %%
train_table= pd.DataFrame({
    'Expected result':[name_maper[p[0]] for p in df.values],
    'Predicted':[name_maper[m[0]] for m in df.values]

})
print(train_table)
#train_table.sort_values(ascending=False,by='Predicted')

# %%
import pickle


# %%
pickle.dump(finaltestedmodel,open('model.pkl','wb'))


# %%
model=pickle.load(open('model.pkl','rb'))

k=list(d1)
for i in range(len(k)):
    if k[i].__contains__('_'):
        k[i]=k[i].replace('_'," ")
l=open('l.txt','w')
l.write(str(k))
l.close()


# %%
symptoms = X.columns.values
symptom_index={}
for index, value in enumerate(symptoms):
    symptom = value
    symptom_index[index] = value
  
data_dict = {
    
    "predictions_classes":encoder.classes_,
    "symptom_index":symptom_index
}
def predictDisease(symptoms):
    symptoms = symptoms.split(",")

    
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = int(symptom)
        input_data[index] = 1
        
    
    input_data = np.array(input_data).reshape(1,-1)
    final_pred=model.predict(input_data)[0]
    
    return final_pred


print(name_maper[predictDisease("1,5,9,49,32,11")])    




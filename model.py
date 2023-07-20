import warnings
import pickle
warnings.filterwarnings('ignore')

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

df=pd.read_csv("SBAnational.csv")
df.dropna(subset=['Name', 'City', 'State', 'BankState', 'NewExist','RevLineCr', 'LowDoc', 'DisbursementDate', 'MIS_Status'], inplace=True)

df[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']] = \
df[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']].applymap(lambda x: x.strip().replace('$', '').replace(',', ''))

def clean_str(x):
    if isinstance(x, str):
        return x.replace('A', '')
    return x

df.ApprovalFY = df.ApprovalFY.apply(clean_str).astype('int64')

df = df.astype({'Zip': 'str', 'NewExist': 'int64', 'UrbanRural': 'str', 'DisbursementGross': 'float', 'BalanceGross': 'float',
                          'ChgOffPrinGr': 'float', 'GrAppv': 'float', 'SBA_Appv': 'float'})

df['Industry'] = df['NAICS'].astype('str').apply(lambda x: x[:2])

df['Industry'] = df['Industry'].map({
    '11': 'Ag/For/Fish/Hunt',
    '21': 'Min/Quar/Oil_Gas_ext',
    '22': 'Utilities',
    '23': 'Construction',
    '31': 'Manufacturing',
    '32': 'Manufacturing',
    '33': 'Manufacturing',
    '42': 'Wholesale_trade',
    '44': 'Retail_trade',
    '45': 'Retail_trade',
    '48': 'Trans/Ware',
    '49': 'Trans/Ware',
    '51': 'Information',
    '52': 'Finance/Insurance',
    '53': 'RE/Rental/Lease',
    '54': 'Prof/Science/Tech',
    '55': 'Mgmt_comp',
    '56': 'Admin_sup/Waste_Mgmt_Rem',
    '61': 'Educational',
    '62': 'Healthcare/Social_assist',
    '71': 'Arts/Entertain/Rec',
    '72': 'Accom/Food_serv',
    '81': 'Other_no_pub',
    '92': 'Public_Admin'
})

df.dropna(subset = ['Industry'], inplace = True)

df.loc[(df['FranchiseCode'] <= 1), 'IsFranchise'] = 0
df.loc[(df['FranchiseCode'] > 1), 'IsFranchise'] = 1

df = df[(df['NewExist'] == 1) | (df['NewExist'] == 2)]

df.loc[(df['NewExist'] == 1), 'NewBusiness'] = 0
df.loc[(df['NewExist'] == 2), 'NewBusiness'] = 1

df = df[(df.RevLineCr == 'Y') | (df.RevLineCr == 'N')]
df = df[(df.LowDoc == 'Y') | (df.LowDoc == 'N')]

df['RevLineCr'] = np.where(df['RevLineCr'] == 'N', 0, 1)
df['LowDoc'] = np.where(df['LowDoc'] == 'N', 0, 1)

df['Default'] = np.where(df['MIS_Status'] == 'P I F', 0, 1)
df['Default'].value_counts()

df[['ApprovalDate', 'DisbursementDate']] = df[['ApprovalDate', 'DisbursementDate']].apply(pd.to_datetime)

df['DaysToDisbursement'] = df['DisbursementDate'] - df['ApprovalDate']

df['DaysToDisbursement'] = df['DaysToDisbursement'].astype('str').apply(lambda x: x[:x.index('d') - 1]).astype('int64')

df['DisbursementFY'] = df['DisbursementDate'].map(lambda x: x.year)

df['StateSame'] = np.where(df['State'] == df['BankState'], 1, 0)

df['SBA_AppvPct'] = df['SBA_Appv'] / df['GrAppv']

df['AppvDisbursed'] = np.where(df['DisbursementGross'] == df['GrAppv'], 1, 0)

df = df.astype({'IsFranchise': 'int64', 'NewBusiness': 'int64'})

df.drop(columns=['LoanNr_ChkDgt', 'Name', 'City', 'Zip', 'Bank', 'NAICS', 'ApprovalDate', 'NewExist', 'FranchiseCode',
                      'ChgOffDate', 'DisbursementDate', 'BalanceGross', 'ChgOffPrinGr', 'SBA_Appv', 'MIS_Status'], inplace=True)

df['RealEstate'] = np.where(df['Term'] >= 240, 1, 0)

df['GreatRecession'] = np.where(((2007 <= df['DisbursementFY']) & (df['DisbursementFY'] <= 2009)) | 
                                     ((df['DisbursementFY'] < 2007) & (df['DisbursementFY'] + (df['Term']/12) >= 2007)), 1, 0)

df = df[df.DisbursementFY <= 2010]

df['DisbursedGreaterAppv'] = np.where(df['DisbursementGross'] > df['GrAppv'], 1, 0)

df = df[df['DaysToDisbursement'] >= 0]

def_ind = df.groupby(['Industry', 'Default'])['Industry'].count().unstack('Default')
def_ind['Def_Percent'] = def_ind[1]/(def_ind[1] + def_ind[0])

def_state = df.groupby(['State', 'Default'])['State'].count().unstack('Default')
def_state['Def_Percent'] = def_state[1]/(def_state[1] + def_state[0])

df = pd.get_dummies(df)

shuffled_subset=df.sample(n=10000, random_state=42)

y = shuffled_subset['Default']
X = shuffled_subset.drop('Default', axis = 1)

scale = StandardScaler()
X_scld = scale.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scld, y, test_size=0.25)


#for logistic regression classification
from sklearn.metrics import classification_report

lr = LogisticRegression(random_state = 42)

lr.fit(X_train, y_train)
y_pred = lr.predict(X_val)
report=classification_report(y_val, y_pred, digits = 3)
print(report)

print("Testing accuracy of logistic regression is:",accuracy_score(y_val, y_pred))
cm=confusion_matrix(y_val,y_pred)



#for random forest classifier

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(random_state=42)
rf.fit(X_train,y_train)

prediction_rf=rf.predict(X_val)

print(classification_report(y_val,prediction_rf,digits=3))
print("Testing accuracy of random forest is:",accuracy_score(y_val, prediction_rf))

# for knn classification

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

prediction_knn = knn.predict(X_val)

report = classification_report(y_val, prediction_knn, digits=3)
print(report)
print("Testing accuracy of KNN is:",accuracy_score(y_val, prediction_knn))

# for decision tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

prediction_dt = dt.predict(X_val)

report = classification_report(y_val, prediction_dt, digits=3)
print(report)
print("Testing accuracy of decision tree is:",accuracy_score(y_val, prediction_dt))

#for naive bayes

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

nb = GaussianNB()

nb.fit(X_train, y_train)
prediction_nb = nb.predict(X_val)

report = classification_report(y_val, prediction_nb, digits=3)
print(report)
print("Testing accuracy of naive bayes is:",accuracy_score(y_val, prediction_nb))

# with open('model.pkl', 'wb') as file:
#     pickle.dump(lr, file)

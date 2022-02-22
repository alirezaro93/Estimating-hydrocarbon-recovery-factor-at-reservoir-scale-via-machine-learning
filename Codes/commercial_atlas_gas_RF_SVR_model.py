import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from model_functions import GaussRankScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import shap
import operator
from sklearn.model_selection import GridSearchCV
import joblib
import plotly.graph_objs as go
import scipy as sp
import seaborn as sns
from numpy import asarray
from sklearn.svm import SVR
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer
from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')
import scipy.stats as stats
from yellowbrick.regressor import residuals_plot, ResidualsPlot
import sys
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr




print("Importing the databases.")

df = pd.read_csv("CA.csv", encoding = "ISO-8859-1",
                 engine = 'python')

df1 = pd.read_csv("independent.csv", encoding = "ISO-8859-1",
                 engine = 'python')

df = df.dropna(how='any', subset=['RECOVERY FACTOR (GAS ULTIMATE)'])

df1 = df1.dropna(how='any')

df = df.sample(frac=1, random_state = 2)

print("Comparing the distribution of the  raw features in the two databases.")

alpha =0.05

for col in list(df.columns.values):

    plt.figure(figsize = (10,6))
    sns.kdeplot(df[col] , bw_method = 0.5 , fill = True, color = 'red', label = 'CA')
    plt.axvline(np.mean(df[col]), color="red", linestyle="dashed", linewidth=1, label='CA mean')
    sns.kdeplot(df1[col] , bw_method = 0.5 , fill = True, color = 'blue', label = 'independent')
    plt.axvline(np.mean(df1[col]), color="blue", linestyle="dashed", linewidth=1, label = 'independent mean')
    plt.legend()
    plt.grid(False)
    plt.savefig('raw {}.png'.format(col), facecolor="w")
    stat, p = ttest_ind(df[col].sample(n=20, random_state=1), df1[col].sample(n=20, random_state=1), axis=0,
                        equal_var=False, alternative='two-sided',nan_policy='omit')
    print('Statistics = %.3f, p = %.3f' % (stat, p))
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
        
        
print("CA data clean up.")

df = df[(df['RECOVERY FACTOR (GAS ULTIMATE)'] >= 0) | (df['RECOVERY FACTOR (GAS ULTIMATE)'].isnull())]

df = df[(df['RECOVERY FACTOR (GAS ULTIMATE)'] <= 1) | (df['RECOVERY FACTOR (GAS ULTIMATE)'].isnull())]

df = df[(df['POROSITY (MATRIX AVERAGE)'] >= 0) | (df['POROSITY (MATRIX AVERAGE)'].isnull())]

df = df[(df['POROSITY (MATRIX AVERAGE)'] <= 1) | (df['POROSITY (MATRIX AVERAGE)'].isnull())]

df = df[(df['WATER SATURATION (AVERAGE)'] >= 0) | (df['WATER SATURATION (AVERAGE)'].isnull())]

df = df[(df['GOR (INITIAL mscfprb)'] >= 0) | (df['GOR (INITIAL mscfprb)'].isnull())]

df = df[(df['GOR (INITIAL mscfprb)'] <= 10e+9) | (df['GOR (INITIAL mscfprb)'].isnull())]

df = df[(df['WATER SATURATION (AVERAGE)'] <= 1) | (df['WATER SATURATION (AVERAGE)'].isnull())]

df = df[(df['RESERVES (ORIGINAL IN PLACE GAS MMscf)'] >= 0) | (df['RESERVES (ORIGINAL IN PLACE GAS MMscf)'].isnull())]

df = df[(df['RESERVES (ORIGINAL IN PLACE GAS MMscf)'] <= 4e+8) | (df['RESERVES (ORIGINAL IN PLACE GAS MMscf)'].isnull())]


clt = round(0.55*df.shape[1])

#rwt = round(0.70*df.shape[0])

df = df.dropna(axis=0, thresh=clt)

#df = df.dropna(axis=1, thresh=rwt)

df = df.reset_index(drop=True)

dfnum = df.select_dtypes(include=['number'])

print('CA correlation between features and features and the lable.')

corr = df.corr(method = 'spearman')

plt.figure(figsize = (20,12))
sns.heatmap(corr, annot = True)
plt.title("Spearman's rho values between features and features and the lable on CA database")
plt.savefig('CA_Corr.png', facecolor="w", bbox_inches='tight')
print("Splitting the data into train and test sets, normalize and standardizing.")  

df_x = dfnum.iloc[:,:-1]

df_y = dfnum.iloc[:,-1]

X_train, X_test, y_train, y_test= train_test_split(df_x, df_y,
                                                   test_size=0.1,
                                                   random_state=42)


print("imputing for the missing data.")

t = 0.10

n = 10

def mode_calculator(df):
    mo=df.mode()
    if mo.shape[0] == 1:
        zmo=mo[0]
    else:
        zmo=mo.mean(axis=0)
    return zmo

def missing_value_handler(i_start, i_end, df):
    temp = df[i_start:i_end]
    fr = temp.isnull().sum()/temp.shape[0]
    if fr > t:
        while fr > t:
            if df.shape[0] - i_end != 0:
                temp = df[i_start:i_end+1]
                i_end += 1
                fr = temp.isnull().sum()/temp.shape[0]
            zmo = mode_calculator(temp)
            temp = temp.replace(np.nan, zmo)
            return temp
        else:
                zmo = mode_calculator(temp)
                temp = temp.replace(np.nan, zmo)
                return temp
    elif fr <= t:
        zmo = mode_calculator(temp)
        temp = temp.replace(np.nan, zmo)
        return temp
    elif fr == 0:
        return temp

for col in list(X_train.columns.values):
    my_col = X_train[col]
    temp_df = pd.DataFrame()
    i_start = 0
    i_end = n
    
    while my_col.shape[0] - i_start >= n:
            curated_temp = missing_value_handler(i_start, i_end, my_col)
            temp_df = temp_df.append(curated_temp.to_frame())
            i_start = i_start+curated_temp.shape[0]
            i_end = i_start + n
            
    last_piece = missing_value_handler(i_start, my_col.shape[0], my_col)
    if last_piece is None:
        X_train[col+"_curated"] = temp_df
        X_train=X_train.drop([col], axis=1)

    else:
        temp_df = temp_df.append(last_piece.to_frame())
        X_train[col+"_curated"] = temp_df
        X_train=X_train.drop([col], axis=1)
        
pass

for col in list(X_train.columns.values):
    X_train[col] = X_train[col].fillna(mode_calculator(X_train[col]))
    
pass



for col in list(X_test.columns.values):
    my_col = X_test[col]
    temp_df = pd.DataFrame()
    i_start = 0
    i_end = n
    
    while my_col.shape[0] - i_start >= n:
            curated_temp = missing_value_handler(i_start, i_end, my_col)
            temp_df = temp_df.append(curated_temp.to_frame())
            i_start = i_start+curated_temp.shape[0]
            i_end = i_start + n
            
    last_piece = missing_value_handler(i_start, my_col.shape[0], my_col)
    if last_piece is None:
        X_test[col+"_curated"] = temp_df
        X_test=X_test.drop([col], axis=1)

    else:
        temp_df = temp_df.append(last_piece.to_frame())
        X_test[col+"_curated"] = temp_df
        X_test=X_test.drop([col], axis=1)
        
pass

for col in list(X_train.columns.values):
    X_test[col] = X_test[col].fillna(mode_calculator(X_test[col]))
    
pass




for col in list(X_train.columns.values):
    plt.figure(figsize=(10,6))
    count, bins, ignored = plt.hist(X_train[col], 100, density=False,
                                    align='mid', color='red')
    plt.title("Imputed Train Feature's Histogram")
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.savefig('Imputed {}.png'.format(col), bbox_inches='tight')


gauss_scaler = GaussRankScaler()

X_trainnum = gauss_scaler.fit_transform(X_train)

X_testnum = gauss_scaler.transform(X_test)

X_trainnum_columns = len(X_trainnum[0])

num_column_names = X_train.iloc[:, :].columns.values.tolist()

#%%

for col in range(0, X_trainnum_columns, 1):
    
    plt.figure(figsize=(10,6))
    count, bins, ignored = plt.hist(X_trainnum[:,col], 100, density=False,
                                    align='mid', color='green')
    plt.title("Gaussian Transformation of the Feature")
    plt.xlabel(num_column_names[col])
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.savefig('Gaussian Transformation of {}.png'.format(num_column_names[col]),
                bbox_inches='tight')
    
plt.figure(figsize=(10,6))
count, bins, ignored = plt.hist(y_train, 100, density=False,
                                align='mid', color='green')
plt.title("Target Variable's Histogram")
plt.xlabel('RECOVERY FACTOR (GAS ULTIMATE)')
plt.ylabel('Frequency')
plt.grid(False)
plt.savefig("Histogram of RECOVERY FACTOR (GAS ULTIMATE) 1",
            bbox_inches='tight')

scaler = preprocessing.MinMaxScaler()
    
X_trainnum = scaler.fit_transform(X_trainnum)

X_testnum = scaler.transform(X_testnum)

X_testnum = np.nan_to_num(X_testnum, nan=0)

X_trainnum = np.nan_to_num(X_trainnum, nan=0)

for col in range(0, X_trainnum_columns, 1):
    
    plt.figure(figsize=(10,6))
    count, bins, ignored = plt.hist(X_trainnum[:,col], 100, density=False,
                                    align='mid')
    plt.title("Processed Feature's Histogram")
    plt.xlabel(num_column_names[col])
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.savefig('Processed Histogram of {}.png'.format(num_column_names[col]),
                bbox_inches='tight')
    
plt.figure(figsize=(10,6))
count, bins, ignored = plt.hist(y_train, 100, density=False,
                                align='mid')
plt.title("Target Variable's Histogram")
plt.xlabel('RECOVERY FACTOR (GAS ULTIMATE)')
plt.ylabel('Frequency')
plt.grid(False)
plt.savefig('Histogram of RECOVERY FACTOR (GAS ULTIMATE) 2',
            bbox_inches='tight')
            
            
            
print("Visualizing the features after cleaning up and preprocessing.")

for col in range(0, X_trainnum_columns, 1):
    fig = plt.figure(constrained_layout=True,figsize=(10,6))
    fig.suptitle("CA Processed Features' Statistical graphs")
    fig.text(0.5, 0.01, num_column_names[col], ha='center')
    ax = fig.add_gridspec(1, 310)
    ax1 = fig.add_subplot(ax[0, 0:100])
    ax1.set_title('Histogram')
    plt.grid(False)
    sns.histplot(X_trainnum[:,col], ax=ax1, bins=100)
    ax1 = fig.add_subplot(ax[0:, 125:185])
    ax1.set_title('Box Plot')
    plt.grid(False)
    plt.boxplot(X_trainnum[:,col])
    ax1 = fig.add_subplot(ax[0:, 210:309])
    stats.probplot(X_trainnum[:,col], dist="norm", plot=plt)
    ax1.set_title('QQ Plot')
    plt.grid(False)
    plt.savefig('CA_Processed Statistical graphs of {}.png'.format(num_column_names[col]),
                bbox_inches='tight',facecolor='w')
    
# new=pd.DataFrame(X_trainnum,columns=num_column_names)
# plt.rcParams['font.family'] = "Times New Roman"

# plt.rcParams['axes.linewidth']=1
# plt.rcParams['axes.edgecolor']='black'

# new.hist(grid=False,figsize=(12,10),bins=130,ec='k',color='red')
# plt.tight_layout()
# plt.show()
    
test_rmse = pd.DataFrame(columns = ['Test', 'rmse_value'])

train_rmse = pd.DataFrame(columns = ['Train', 'rmse_value'])

independent_rmse = pd.DataFrame(columns = ['Train', 'rmse_value'])

test_r2 = pd.DataFrame(columns = ['Test', 'r2_value'])

train_r2 = pd.DataFrame(columns = ['Train', 'r2_value'])

independent_r2 = pd.DataFrame(columns = ['Train', 'r2_value'])

test_r = pd.DataFrame(columns = ['Test', 'r2_value'])

train_r = pd.DataFrame(columns = ['Train', 'r2_value'])

independent_r = pd.DataFrame(columns = ['Train', 'r_value'])

y_test_values = pd.DataFrame(columns = ['y_measured', 'y_estimated'])

y_train_values = pd.DataFrame(columns = ['y_measured', 'y_estimated'])

y_predindependent_values = pd.DataFrame(columns = ['y_measured', 'y_estimated'])

print("Constructing a learning curve of a SVR model fitted on the database and investigating the effect of adding more samples on the model's performance.")

modelSVR = SVR(C=1.0, epsilon=0.2)

train_error, val_error = [], []

for m in range (1,len(X_trainnum)):
    modelSVR.fit(X_trainnum[:m],y_train[:m].ravel())
    y_pred = modelSVR.predict(X_trainnum[:m])
    y_pred_test = modelSVR.predict(X_testnum)
    train_error.append(sqrt(mean_squared_error(y_train[:m], y_pred)))
    val_error.append(sqrt(mean_squared_error(y_test, y_pred_test)))

plt.figure(figsize=(20,12))   
plt.plot(train_error,"r",linewidth=1,label='Train')
plt.plot(val_error,"b",linewidth=1,label='Validation')
plt.title('SVR learning curve on CA')
plt.xlabel('Training set size')
plt.ylabel('RMSE')
plt.xticks(np.arange(0, len(X_train)+1, 250))
plt.yticks(np.arange(0, 0.25, 0.05))
plt.legend()
plt.annotate('%0.2f' % train_error[-1], xy=(1, train_error[-1]), xytext=(8, 0), 
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.annotate('%0.2f' % val_error[-1], xy=(1, val_error[-1]), xytext=(8, 0), 
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.yticks(np.arange(0, 0.4, 0.05))
plt.grid(False)
plt.savefig('SVR_learning_curve.png',
            bbox_inches='tight',facecolor='w')

print("Hyperparameter tuning.")

parameters = {'C': [0.001,0.01, 0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid'],
              'epsilon':[0.01,0.001,0.8,0.7,0.6,0.9,0.4,0.1,0.2,0.5,0.3]}
       
model = SVR()
       
grid = GridSearchCV(model,parameters, cv=3, scoring=
                   'neg_root_mean_squared_error', verbose=0,
                   n_jobs=-1, refit=True)
       
grid.fit(X_trainnum, y_train)

best_params = grid.best_params_

best_model=SVR(**best_params)

best_model.fit(X_trainnum, y_train)

joblib.dump(best_model,'best_SVMR_model_CA_Gas_RF')

y_pred = best_model.predict(X_testnum)

y_pred1 = best_model.predict(X_trainnum)

def corrcoeff(x, y):
    top_term = np.sum((x- np.mean(x)) * (y-np.mean(y)))
    term1 = np.sum((x- np.mean(x)))**2
    term2 = np.sum((y- np.mean(y)))**2
    bottom_term = np.sqrt(term1*term2)
    return round(top_term/bottom_term,5)

score=r2_score(y_test, y_pred)
score1=r2_score(y_train, y_pred1)

rmse = sqrt(mean_squared_error(y_test, y_pred))
rmse1 = sqrt(mean_squared_error(y_train, y_pred1))

corrtest, _ = pearsonr(y_test, y_pred)
print('Pearsons correlation: %.3f' % corrtest)

corrtrain, _ = pearsonr(y_train, y_pred1)
print('Pearsons correlation: %.3f' % corrtrain)

print('')
print("Test RMSE:", rmse)
print('')
print("Train RMSE:", rmse1)
print('')
print("Test R^2 is equal to", score)
print('')
print("Train R^2 is equal to", score1)
print('')

#%% plots

plt.figure(figsize=(8,4))
plt.scatter(y_test, y_pred, c='r', label='Test Measured RF')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 1, label="y = x")
plt.xlabel('Measured RF')
plt.ylabel('Esitmated RF')
plt.title('Measured RF Vs Estimated RF (Test Dataset)')
plt.legend()
plt.grid(False)
plt.savefig('Measured RF Vs Estimated RF (Test Dataset)',
            bbox_inches='tight')

plt.figure(figsize=(8,4))
plt.scatter(y_train, y_pred1, c='b', label='Train Measured RF')
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 1, label="y = x")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Measured RF')
plt.ylabel('Esitmated RF')
plt.title('Measured RF Vs Estimated RF (Train Dataset)')
plt.legend()
plt.grid(False)
plt.savefig('Measured RF Vs Estimated RF (Train Dataset)',
            bbox_inches='tight')

#%% feature importance

print('Feature Importance of the models fitted on the CA database')

explainerSVR = shap.KernelExplainer(best_model.predict, shap.sample(X_trainnum, 100))

shap_valuesSVR = explainerSVR.shap_values(shap.sample(X_trainnum, 100))

plt.figure(figsize=(10,6))
plt.title('CA ModelSVR Feature Importance')
shap.summary_plot(shap_valuesSVR, shap.sample(X_trainnum, 100), plot_type = "bar", 
                  feature_names = df_x.columns, show = False)
plt.savefig('CASVR_average_impact_on_model_output_magnitude.png', bbox_inches='tight',facecolor='w')

plt.figure(figsize=(10,6))
plt.title('CA ModelSVR Feature Importance')
shap.summary_plot(shap_valuesSVR, shap.sample(X_trainnum, 100), feature_names = df_x.columns,
                  show = False)
plt.savefig('CASVR_impact_on_model_output.png', bbox_inches='tight',facecolor='w')

#%%    

# print("Plotting the residuals")

# plt.figure(figsize=(10,6))
# residuals_plot(best_model, X_trainnum, y_train, X_testnum, y_test, size=(800, 500))
# plt.grid(False)

#%%
         

print("independent data clean up.")

df1 = df1[(df1['RECOVERY FACTOR (GAS ULTIMATE)'] >= 0) | (df1['RECOVERY FACTOR (GAS ULTIMATE)'].isnull())]

df1 = df1[(df1['RECOVERY FACTOR (GAS ULTIMATE)'] <= 1) | (df1['RECOVERY FACTOR (GAS ULTIMATE)'].isnull())]

df1 = df1[(df1['POROSITY (MATRIX AVERAGE)'] >= 0) | (df1['POROSITY (MATRIX AVERAGE)'].isnull())]

df1 = df1[(df1['POROSITY (MATRIX AVERAGE)'] <= 1) | (df1['POROSITY (MATRIX AVERAGE)'].isnull())]

df1 = df1[(df1['WATER SATURATION (AVERAGE)'] >= 0) | (df1['WATER SATURATION (AVERAGE)'].isnull())]

df1 = df1[(df1['WATER SATURATION (AVERAGE)'] <= 1) | (df1['WATER SATURATION (AVERAGE)'].isnull())]

df1 = df1[(df1['GOR (INITIAL mscfprb)'] >= 0) | (df1['GOR (INITIAL mscfprb)'].isnull())]

df1 = df1[(df1['GOR (INITIAL mscfprb)'] <= 10e+9) | (df1['GOR (INITIAL mscfprb)'].isnull())]

df1 = df1[(df1['RESERVES (ORIGINAL IN PLACE GAS MMscf)'] >= 0) | (df1['RESERVES (ORIGINAL IN PLACE GAS MMscf)'].isnull())]

df1 = df1[(df1['RESERVES (ORIGINAL IN PLACE GAS MMscf)'] <= 4e+8) | (df1['RESERVES (ORIGINAL IN PLACE GAS MMscf)'].isnull())]

print('independent data processing.')

Xindependent = df1.iloc[:, :-1]

Xindependent = gauss_scaler.fit_transform(Xindependent)

Xindependent = scaler.fit_transform(Xindependent)

yindependent = df1.iloc[:, -1]

bestmodel = joblib.load('best_SVMR_model_CA_Gas_RF')

y_predindependent = bestmodel.predict(Xindependent)

scoreindependent = r2_score(yindependent, y_predindependent)

rmseindependent = sqrt(mean_squared_error(yindependent, y_predindependent))

corrindependent, _ = pearsonr(yindependent, y_predindependent)
print('independent Pearsons correlation: %.3f' % corrindependent)

#%%    

print('')
print("independent Test RMSE =", rmseindependent)
print('')
print("independent Test R^2 =", scoreindependent)
print('')

plt.figure(figsize = (8, 4))
plt.scatter(yindependent, y_predindependent, c = 'r',
            label = 'independent Test Measured RF')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 1, label = "y = x")
plt.xlabel('Measured RF')
plt.ylabel('Esitmated RF')
plt.title('Measured RF Vs. Estimated RF (independent Test Dataset)')
plt.legend()
plt.grid(False)
plt.savefig('Measured RF Vs Estimated RF (independent Dataset)',
            bbox_inches='tight')

#%%

y_test_values = pd.concat([y_test_values,pd.DataFrame({"y_measured": y_test,
                            "y_estimated": y_pred})], ignore_index = True)
        
y_train_values = pd.concat([y_train_values,
                            pd.DataFrame({"y_measured": y_train,
                                          "y_estimated": y_pred1})],
                           ignore_index = True)
    
test_rmse = test_rmse.append({"Test": 'Test', "rmse_value" : rmse},
                             ignore_index = True)

train_rmse = train_rmse.append({"Train": "Train", "rmse_value" : rmse1},
                             ignore_index = True)

test_r2 = test_r2.append({"Test": "Test", "r2_value" : score},
                             ignore_index = True)

train_r2 = train_r2.append({"Train": "Train", "r2_value" : score1},
                             ignore_index = True)

independent_r2 = independent_r2.append({"Train": "Train", "r2_value" : scoreindependent},
                             ignore_index = True)

test_r = test_r.append({"Test": "Test", "r2_value" : corrtest},
                             ignore_index = True)

train_r = train_r.append({"Train": "Train", "r2_value" : corrtrain},
                             ignore_index = True)

independent_r = independent_r.append({"Train": "Train", "r2_value" : corrindependent},
                             ignore_index = True)

independent_rmse = independent_rmse.append({"Train": "Train", "rmse_value" : rmseindependent},
                             ignore_index = True)

best_parameters = pd.DataFrame.from_dict(best_params, orient='index').rename(
    columns={0: 'Value'})

x_set = pd.DataFrame(data = X_trainnum, columns = df_x.columns)

x_settest = pd.DataFrame(data = X_testnum, columns = df_x.columns)

x_setindependent = pd.DataFrame(data = Xindependent, columns = df_x.columns)

y_predindependent_values = pd.concat([y_predindependent_values,pd.DataFrame({"y_measured": yindependent,
                            "y_estimated": y_predindependent})], ignore_index = True)

with pd.ExcelWriter('model_data.xlsx', engine="openpyxl") as writer:
    
    y_test_values.to_excel(writer, sheet_name = 'y_test_values')
        
    y_train_values.to_excel(writer, sheet_name = 'y_train_values')
    
    y_predindependent_values.to_excel(writer, sheet_name = 'y_predindependent_values')
        
    test_rmse.to_excel(writer, sheet_name = 'test_rmse')
    
    train_rmse.to_excel(writer, sheet_name = 'train_rmse')
    
    independent_rmse.to_excel(writer, sheet_name = 'independent_rmse')
            
    test_r2.to_excel(writer, sheet_name = 'test_r2')
            
    train_r2.to_excel(writer, sheet_name = 'train_r2')
    
    independent_r2.to_excel(writer, sheet_name = 'independent_r2')
    
    test_r.to_excel(writer, sheet_name = 'test_r')
            
    train_r.to_excel(writer, sheet_name = 'train_r')
    
    independent_r.to_excel(writer, sheet_name = 'independent_r')
    
    best_parameters.to_excel(writer, sheet_name = 'best parameters')
    
    x_set.to_excel(writer, sheet_name = 'x train')
    
    x_settest.to_excel(writer, sheet_name = 'x test')
    
    x_setindependent.to_excel(writer, sheet_name = 'x independent')
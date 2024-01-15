# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:42:10 2023

"""
import sys,os, timeit, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB     #If data or column is real, decimal..
from sklearn.naive_bayes import BernoulliNB    #If data or column is binary (yes/no etc)
from sklearn.naive_bayes import MultinomialNB  # If data or colum in int numbers
from ucimlrepo import fetch_ucirepo

def vis(X, Y, W=None, b=None):
    indices_neg1 = (Y == -1).nonzero()[0]
    indices_pos1 = (Y == 1).nonzero()[0]
    plt.scatter(X[:,0][indices_neg1], X[:,1][indices_neg1], 
                c='blue', label='class -1')
    plt.scatter(X[:,0][indices_pos1], X[:,1][indices_pos1], 
                c='red', label='class 1')
    plt.legend()
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    
    if W is not None:
        # w0x0+w1x1+b=0 => x1=-w0x0/w1-b/w1
        w0 = W[0]
        w1 = W[1]
        temp = -w1*np.array([X[:,1].min(), X[:,1].max()])/w0-b/w0
        x0_min = max(temp.min(), X[:,0].min())
        x0_max = min(temp.max(), X[:,1].max())
        x0 = np.linspace(x0_min,x0_max,100)
        x1 = -w0*x0/w1-b/w1
        plt.plot(x0,x1,color='black')

    plt.show()

# Calculate accuracy given feature vectors X and labels Y.
def calc_accu(X, Y, classifier):
    
    # Hint: Use classifier.predict()
    Y_pred = classifier.predict(X)

    # Hint: Use accuracy_score().
    return( accuracy_score(Y, Y_pred))

def ohe(df, transform_list):
    #Feature engineering using OneHotEncoder and sklearn
    #Converts categories in feature space to numerical values
    #Input data set as pandas dataframe (df)
    #APPLY only for labels in transform_list
    #YOUR RESPONSIBILITY to check no category data left not transformed
    
    
    # One-hot encoding multiple columns
    from sklearn.preprocessing import OneHotEncoder as OHE
    from sklearn.compose import make_column_transformer

    #df = df[['workclass','education','marital-status','occupation',
       #      'relationship','race','native-country']]
     
    transformer = make_column_transformer(
         (OHE(sparse_output=False),transform_list), remainder='passthrough')
     
    #transformer = make_column_transformer(
    #   (OneHotEncoder(sparse_output=False), ['workclass','education','occupation','race','native-country']),
     #remainder='passthrough')
       
    transformed = transformer.fit_transform(df)
    transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
    #print(transformed_df.head())
    print('Dim before ohe:',np.shape(df))
    print('Dim after ohe:',np.shape(transformed_df))
    return(transformed_df)

def add_header(data_path,name_file,data_file):
    #Add header labels to csv data read as panadas dataframe (df)
    #df comes without header labels,  
    #Header labels are extracted from name_file
    #Specific to data file downloaded from ML repository of UCI website
    
    df = pd.read_csv(os.path.join(data_path,data_file)) 
    
    header=[]
    with open(os.path.join(data_path,name_file), 'r') as f:
       for line in [ x for x in f if x.strip()]:
           if line.split()[0].endswith(':'):
               header.append(line.split()[0].strip(':'))
            
    #print(header)
    header.append('target')
    df.columns = header
    #print(df.head())
    
    return(df,header) #Returns as dataframe with header

def deleted(df,delete_list):
    #Delete by columns from df by list
    print('before drop:',np.shape(df))
    print('Deleting:',delete_list)
    df_dropped=df.drop(delete_list, axis=1)
    print('after drop:',np.shape(df_dropped))
    return(df_dropped)


def prepare_data_adult():
    #Data downloaded from UCI https://archive.ics.uci.edu/dataset/2/adult
    #Downloaded data is csv file, converted to dataframe
    #Returns X(all features) and Y(all targets -1,1) data set as np arrays
    #Columns as per delete list are deleted from the data frame
    #Category columns as per transform_list converted to numercal matrices
    #Raw data is in pandas dataframe, without header row.
    #Header row is added from name_file
    #Returns X,Y as np arrays X dim:NxM, 
    #Y dim: Nx1 (N=number of data points, M=number of feature columns)
    #M increases many times after applying ohe transformation
    #ohe is a def, stands for One Hot Encoder of sklearn
    #Modified target to +1/-1 as Y = (y == '<50K') line as per data in last column to convert target to {-1,1}
    normalize=True
    data_path = 'C:/Users/prayu/LIDAR_UTILS/tmp/PROJECT/DATA/adult/'
    data_file = 'adult.data'
    name_file = 'adult.names'
    report_file = 'NORreport_adult.txt'
    report_file = os.path.join(data_path,report_file)
    
    delete_list = ['marital-status', 'relationship','education','sex']
    #delete_list = [ 'relationship','education']
    transform_list = ['workclass','occupation','race','native-country'] 
    
    report = ['Data file:',data_file,'\n',
              'Features not used:',delete_list,'\n',
              'Categories transformed:\n',transform_list,'\n','\n']
    
    df= pd.read_csv(os.path.join(data_path,data_file))
    #print(type(adult))

    #If df does not have header labels, run this, otherwise skip
    #Add column header labels from name file adult.names
    df,header = add_header(data_path,name_file,data_file)

    #Delete colum data that are not to be used by labels in delete_list
    df = deleted(df,delete_list)

    #Transform labelled categories to numercal matrices by labels in transform_list
    #MAKE SURE: delete_list + transform_list should cover all categories in raw data
    df = ohe(df,transform_list )

    #sys.exit()

    X= df.iloc[:,:-1] #Extract only features
    y= df.iloc[:,-1]  #Extract target
 

    #CHANGE HERE ACCORDING TO data file
    Y = (y == ' <=50K').astype(float)  #Convert to 1 if <= 50K, otherwise 0
    Y[Y == 0] = -1                   #Convert all 0 to -1 (48842x1)

    #ECI
    # Divide the data points into training set and test set.

    if normalize:
        d = preprocessing.normalize(X)
        X = pd.DataFrame(d, columns=X.columns)
    X=X.to_numpy(); Y=Y.to_numpy()

    #X_train = X[:5000][:,[0,2]] # Shape: (5000,2)
    #X = X[:5000][:] # 5000  data points
    #Y = Y[:5000]         
    #X_train = X; Y_train = Y  #All data points

    #x_train, x_test, y_train, y_test=train_test_split(X_train,Y_train, test_size=test_size, random_state=random_code)
    #print('Dim of train and test data used:',np.shape(X),np.shape(Y), np.shape(x_test), np.shape(y_test))
    return(X,Y,report,report_file)

def plot_CM(cm,C):
    import matplotlib.pyplot as plt
    import seaborn as sns
    class_names=[-1,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix [Model:LG, C='+str(C)+']',  y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
     
    
def run_Linear_SVC(x_train, y_train, x_test, y_test, C, max_iter):
    from sklearn.model_selection import cross_val_score
 
    # Create a linear SVM classifier.
    # Hints: You can use svm.LinearSVC()
    #        Besides, we use Hinge loss and L2 penalty for weights.
    #        The max iterations should be set to 100000.
    #        The regularization parameter should be set as C.
    #        The other arguments of svm.LinearSVC() are set as default values.

    classifier = svm.LinearSVC(loss='hinge', penalty='l2', C=C, max_iter=max_iter)

    # Use the classifier to fit the training set (use X_train, Y_train).
    # Hint: You can use classifier.fit().

    classifier.fit(x_train, y_train)

    # Obtain the weights and bias from the linear SVM classifier.
    W = classifier.coef_[0]
    b = classifier.intercept_[0]

    # Show decision boundary, training error and test error.
    print('######################################################')
    print('C = {}'.format(C))
    #print('Decision boundary: {:.3f}x0+{:.3f}x1+{:.3f}=0'.format(W[0],W[1],b))
    #vis(X_train, Y_train, W, b)
 
    #accu_training = calc_accu(x_train, y_train, classifier)
    accu_training = calc_accu(x_test, y_test, classifier)
    #e_training = 1 - accu_training
    #print('Training error: {}'.format(e_training))
    print('Training accuracy: {:.3f}'.format(accu_training))
    print('######################################################')
    print('\n\n')
    scores = cross_val_score(classifier, x_train, y_train, cv = 5)
    print('CV_Accuracy, sd=%0.2f %0.2f' % ( scores.mean(), scores.std()))
     
    return(accu_training,scores)

def run_Naive_Bayes(x_train, y_train, x_test, y_test, C, iterations):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    
    classifier = GaussianNB()
    classifier.fit(x_train,y_train)
    
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)
    scores = cross_val_score(classifier, x_train, y_train, cv = 5)
    print('CV_Accuracy, sd=%0.2f %0.2f' % ( scores.mean(), scores.std()))
     
    return(accuracy,scores)

def run_Logistic_Regression(x_train, y_train, x_test, y_test, C, iterations):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_score
    
    #classifier = LogisticRegression()
    classifier = LogisticRegression(solver='liblinear', C=C)
    classifier.fit(x_train,y_train)
    
    y_pred = classifier.predict(x_test)
    accuracy = classifier.score(x_test, y_test)
    conf_m = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print('report:', report, sep='\n')
    
    #Plot Confusion matrix 
    plot_CM(conf_m,C)
    
    scores = cross_val_score(classifier, x_train, y_train, cv = 5)
    print('CV_Accuracy, sd=%0.2f %0.2f' % ( scores.mean(), scores.std()))
    return(accuracy,scores,report)

def run_Decision_Tree(x_train, y_train, x_test, y_test, C, iterations):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_score
    from sklearn import tree
    
    #classifier = LogisticRegression()
    classifier =  tree.DecisionTreeClassifier()
    classifier.fit(x_train,y_train)
    
    y_pred = classifier.predict(x_test)
    accuracy = classifier.score(x_test, y_test)
    conf_m = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print('report:', report, sep='\n')
    
    #Plot Confusion matrix 
    plot_CM(conf_m,C)
    
    scores = cross_val_score(classifier, x_train, y_train, cv = 5)
    print('CV_Accuracy, sd=%0.2f %0.2f' % ( scores.mean(), scores.std()))
    return(accuracy,scores,report)

def run_GB(x_train, y_train, x_test, y_test, C, iterations):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    
    #classifier = LogisticRegression()
    classifier =  GradientBoostingClassifier()
    print('Flag1')
    classifier.fit(x_train,y_train)
    print('Flag2')
    print('score=',classifier.score(x_test,y_test))
    y_pred = classifier.predict(x_test)
    accuracy = classifier.score(x_test, y_test)
    print('Flag3')
    #accuracy2 = classifier_score(y_test, y_pred)
     
    #conf_m = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print('report:', report, sep='\n')
    
    #Plot Confusion matrix 
    #plot_CM(conf_m,C)
    
    scores = cross_val_score(classifier, x_train, y_train, cv = 5)
    #scores_test = cross_val_score(classifier, x_test, y_test, cv = 5)
    print('CV_Accuracy, sd=%0.2f %0.2f' % ( scores.mean(), scores.std()))
    return(accuracy,scores,report)

def write_report9(report_file,report) :
        with open(report_file,'w') as f:
            for line in report:
                f.write(str(line))
def write_report9(report_file,report) :
        with open(report_file,'w') as f:
            for line in report:
                f.write(str(line))
                
                
def prepare_data_coverType():
    #Data downloaded from: https://archive.ics.uci.edu/dataset/31/covertype
    # Direct download does not work, unzipping not succesful
    #Used python download:
    #Download once only
    download = None  # Set to True/None  
    normalize = True
    
    data_path = 'C:/Users/prayu/LIDAR_UTILS/tmp/PROJECT/DATA/covertype/'
    data_file = 'cover.csv'
    report_file = 'NORreport_cover.txt'
    
    data_file = os.path.join(data_path,data_file)
    report_file = os.path.join(data_path,report_file)

    if download:
        from ucimlrepo import fetch_ucirepo 
        # fetch dataset 
        covertype = fetch_ucirepo(id=31) 
        # data (as pandas dataframes) 
        data_csv = covertype.data.features 
        # metadata 
        print(covertype.metadata) 
        # variable information 
        print(covertype.variables) 
        print('data_csv:',np.shape(data_csv), type(data_csv))
        #Convert to csv file for saving as file
        data_csv.to_csv(data_file)
    
      
    #No need to delete or transform any, data is ready
    delete_list = []
    transform_list = []
    #delete_list = ['marital-status', 'relationship','education','sex']
    #transform_list = ['workclass','occupation','race','native-country'] 
    
    report = ['Data file:',data_file,'\n',
              'Features not used:',delete_list,'\n',
              'Categories transformed:\n',transform_list,'\n','\n']
    
    df= pd.read_csv(os.path.join(data_path,data_file))
    df = df.sample(frac=1)
    df = df.iloc[:50000]
    X= df.iloc[:,:-1] #Extract only features
    y= df.iloc[:,-1]  #Extract target  
    #Normalize data
    if normalize:
        d = preprocessing.normalize(X)
        X = pd.DataFrame(d, columns=X.columns)
        
    X = X.to_numpy()
    y = y.to_numpy()
    #Select only 50,000 data points
    #Disable true randomness (seed) and shuffle, everytime same sequence occurs
    #np.random.seed(1)
    #np.random.shuffle(df)
    #df = df[:50000]         #SELECTS ONLY 50000 data points
      
    #X= df.iloc[:,:-1] #Extract only features
    #y= df.iloc[:,-1]  #Extract target
    
    #X = df[:,:-1]
    #y = df[:,-1]
    
    print('type:',type(X))
     
        
    print('X:',np.shape(X))
    print('y:',np.shape(y))
    
    #summary = y.describe(include = 'all')
    #print(np.bincount(y))    
    #Find most frequently occuring number in y, use it to separate target {-1,1]}
    target_key= np.argmax(np.bincount(y))
    print('target key in y=',target_key)
    
    #from collections import Counter
    #b = Counter(y)
    #print('Alt y:',b.most_common(1))
    
    Y = (y == target_key).astype(float)  #Convert to 1 if == target_key, otherwise 0
    Y[Y == 0] = -1        
           #Convert all 0 to -1 
    #b=Counter(Y)
    #print(b.most_common(2))
        
    return(X,Y,report,report_file)

def prepare_data_letter():
    #Data downloaded from: https://archive.ics.uci.edu/dataset/59/letter+recognition
    #  
     
    #Download once only
    download = None  # Set to True/None  
    normalize=True
    
    data_path = 'C:/Users/prayu/LIDAR_UTILS/tmp/PROJECT/DATA/letter/'
    data_file = 'letter-recognition.data'
    report_file = 'NORreport_letter.txt'
    
    data_file = os.path.join(data_path,data_file)
    report_file = os.path.join(data_path,report_file)
    
    #No need to delete or transform any, data is ready
    delete_list = []
    transform_list = []
    
    report = ['Data file:',data_file,'\n',
              'Features not used:',delete_list,'\n',
              'Categories transformed:\n',transform_list,'\n','\n']
    
    df = pd.read_csv(data_file) 
    
    print('dim:',np.shape(df)  )  
    X= df.iloc[:,1:] #Extract only features
    y = df.iloc[:,0]
    
    #Normalize
    if normalize:
        d = preprocessing.normalize(X)
        X = pd.DataFrame(d, columns=X.columns)
    #Convert letters A-M to -1, N-Z to 1
    y = y.to_numpy() 
    X = X.to_numpy()
    for i in range(len(y)):
        #print(item)
        if re.match(r'[A-M]', y[i]):
            #print(i,y[i])
            y[i]= -1.0
        else:
            y[i]= 1.0
     
    #print('X:',np.shape(X))
    #print('y:',np.shape(y))
    #print('X=',X)
    
    Y = (y == 1.0).astype(float)
    Y[Y == 0] = -1 
    #print('X:',type(X),type(y))
    #print('Y=',Y)
    return(X,Y,report,report_file)

def prepare_data_drybean():
    #Data downloaded from:https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
     
    #Download once only
    download = None  # Set to True/None  
    normalize=True
    
    data_path = 'C:/Users/prayu/LIDAR_UTILS/tmp/PROJECT/DATA/drybean/'
    data_file = 'Dry_Bean_Dataset.xlsx'
    report_file = 'NORreport_drybean.txt'
    
    data_file = os.path.join(data_path,data_file)
    report_file = os.path.join(data_path,report_file)
    
    #No need to delete or transform any, data is ready
    delete_list = []
    transform_list = []
    
    report = ['Data file:',data_file,'\n',
              'Features not used:',delete_list,'\n',
              'Categories transformed:\n',transform_list,'\n','\n']
    
    df = pd.read_excel(data_file) 
    #print(df)
    print('dim:',np.shape(df)  )  
    X= df.iloc[:,:-1] #Extract only features
    y = df.iloc[:,-1]
    
    if normalize:
        d = preprocessing.normalize(X)
        X = pd.DataFrame(d, columns=X.columns)
        
    y = y.to_numpy() 
    X = X.to_numpy()
    
    #Find most frequently occuring number in y, use it to separate target {-1,1]}
    #target_key= np.argmax(np.bincount(y))
    unique,pos = np.unique(y, return_inverse=True)
    counts = np.bincount(pos)
    maxpos = counts.argmax()
     
    target_key = unique[maxpos]
    print('target key in y=',target_key,counts[maxpos] )
    
    Y = (y == target_key).astype(float)  #Convert to 1 if == target_key, otherwise 0
    Y[Y == 0] = -1     
    
    return(X,Y,report,report_file)


def main():
    X,Y, report,report_file = prepare_data_adult()
    #X,Y, report, report_file = prepare_data_coverType()
    #X,Y, report, report_file = prepare_data_letter()
    #X,Y, report,report_file = prepare_data_drybean()
    #sys.exit()
    #print('Xdim,Ydim:',np.shape(X), np.shape(Y))   
                 
    #C_list = [0.01,0.1,1,10,100]
    C = 1000.00
    iterations = 10000
    
    #Rename report file including c and iter values
    c = str(C).replace('.','d')
    report_file = report_file.replace('.','_C'+c+'_Iter'+str(iterations)+'.')
     
    test_size = [0.2, 0.5, 0.8] #Fraction of total data to be taken as test data
    #models = ['SVC', 'NB', 'LG', 'DT', 'GB']
    models = ['SVC',  'LG']
    #models = ['GB']

    for t_size in test_size:
        report.append('---------------------------------------------------\n'+
                      'test data fraction='+str(t_size)+'\n')
        for model in models:
            match model:
                case 'SVC':
                    tic= timeit.default_timer()
                    report.append('Model: Linear SVC\nC='+str(C)+' Iter='+str(iterations)+'\n')
                    # Visualize training set.
                    #vis(X, Y)
                    #nx = str(np.shape(x_train));nt = np.shape(x_test)
                    #write_report2(report_file,nx,nt)
               
                    #Run three times for different random sets
    
                    #Run model
                    ac=0
                    for i in range(3):
                        random_code = int(t_size *100) + i
                        
                        x_train, x_test, y_train, y_test = \
                            train_test_split(X,Y, test_size=t_size, random_state=random_code)
                        print('train dim:',np.shape(x_train),np.shape(y_train))
                        accuracy,CVscores = run_Linear_SVC(x_train, y_train, x_test, y_test, C, iterations)
                        #print('Flag1')
                        ac += round(accuracy,3)
                         
                        report.append('Training data:'+str(np.shape(x_train))+
                                       'Test data:'+str(np.shape(x_test))+'random_code:'+str(random_code)+'\n'+
                                       'Set'+str(i+1)+'='+str(round(accuracy,3))+'|'+
                                       'CV Accuracy(5)='+str(round(CVscores.mean(),3))+', CV s.d.='+str(round(CVscores.std(),3))+'\n')
                    accu_av = round(ac/3,3)
                    
                    print('Accuracy averaged over three random sets:{:.3f}'.format(accu_av))
                    toc=timeit.default_timer()
                    report.append('Average Accuracy='+str(accu_av)+'\n'+'Procesing time(sec)='+str(round(toc-tic,2))+'\n\n')
                     
                        
                case 'NB':
                    tic= timeit.default_timer()
                    report.append('Model: Naive Bayes\nC='+str(C)+' Iter='+str(iterations)+'\n')
                    ac=0
                    for i in range(3):
                        random_code = int(t_size *100) + i
                        x_train, x_test, y_train, y_test = \
                            train_test_split(X,Y, test_size=t_size, random_state=random_code)
                        accuracy,CVscores = run_Naive_Bayes(x_train, y_train, x_test, y_test, C, iterations)
                        ac += round(accuracy,3)
                        #write_report3(report_file,round(accuracy,3),i,CVscores)
                        report.append('Training data:'+str(np.shape(x_train))+
                                       'Test data:'+str(np.shape(x_test))+'random_code:'+str(random_code)+'\n'+
                                       'Set'+str(i+1)+'='+str(round(accuracy,3))+'|'+
                                       'CV Accuracy(5)='+str(round(CVscores.mean(),3))+', CV s.d.='+str(round(CVscores.std(),3))+'\n')
                        
                    accu_av = round(ac/3,3)
                    print('Accuracy averaged over three random sets:{:.3f}'.format(accu_av))
                    toc=timeit.default_timer()
                    report.append('Average Accuracy='+str(accu_av)+'\n'+'Procesing time(sec)='+str(round(toc-tic,2))+'\n\n')
                    #write_report4(report_file,accu_av,round(toc-tic,2))
                        
                case 'LG':  #Logistic Regression
                    tic= timeit.default_timer()
                    report.append('Model: Logistic Regression\nC='+str(C)+' Iter='+str(iterations)+'\n')
                    ac=0
                    for i in range(3):
                        random_code = int(t_size *100) + i
                        x_train, x_test, y_train, y_test = \
                            train_test_split(X,Y, test_size=t_size, random_state=random_code)
                        accuracy,CVscores, rep = run_Logistic_Regression(x_train, y_train, x_test, y_test, C, iterations)
                        ac += round(accuracy,3)
                        #write_report3(report_file,round(accuracy,3),i,CVscores)
                        report.append('Training data:'+str(np.shape(x_train))+
                                       'Test data:'+str(np.shape(x_test))+'random_code:'+str(random_code)+'\n'+
                                       'Set'+str(i+1)+'='+str(round(accuracy,3))+'|'+
                                       'CV Accuracy(5)='+str(round(CVscores.mean(),3))+', CV s.d.='+str(round(CVscores.std(),3))+'\n')           
                    accu_av = round(ac/3,3)
                    print('Accuracy averaged over three random sets:{:.3f}'.format(accu_av))
                    toc=timeit.default_timer()
                    report.append('Average Accuracy='+str(accu_av)+'\n'+'Procesing time(sec)='+str(round(toc-tic,2))+'\n\n')
                    #write_report4(report_file,accu_av,round(toc-tic,2))
                   
                case 'DT': #Decision Tree
                    tic= timeit.default_timer()
                    report.append('Model: Decision Tree\nC='+str(C)+' Iter='+str(iterations)+'\n')
                    ac=0
                    for i in range(3):
                        random_code = int(t_size *100) + i
                        x_train, x_test, y_train, y_test = \
                            train_test_split(X,Y, test_size=t_size, random_state=random_code)
                        accuracy,CVscores, rep = run_Decision_Tree(x_train, y_train, x_test, y_test, C, iterations)
                        ac += round(accuracy,3)
                        #write_report3(report_file,round(accuracy,3),i,CVscores)
                        report.append('Training data:'+str(np.shape(x_train))+
                                       'Test data:'+str(np.shape(x_test))+'random_code:'+str(random_code)+'\n'+
                                       'Set'+str(i+1)+'='+str(round(accuracy,3))+'|'+
                                       'CV Accuracy(5)='+str(round(CVscores.mean(),3))+', CV s.d.='+str(round(CVscores.std(),3))+'\n')           
                    accu_av = round(ac/3,3)
                    print('Accuracy averaged over three random sets:{:.3f}'.format(accu_av))
                    toc=timeit.default_timer()
                    report.append('Average Accuracy='+str(accu_av)+'\n'+'Procesing time(sec)='+str(round(toc-tic,2))+'\n\n')
                        
                case 'GB': #Gradient Boosting
                    tic= timeit.default_timer()
                    report.append('Model: Gradient Boosting\nC='+str(C)+' Iter='+str(iterations)+'\n')
                    ac=0
                    for i in range(3):
                        random_code = int(t_size *100) + i
                        x_train, x_test, y_train, y_test = \
                            train_test_split(X,Y, test_size=t_size, random_state=random_code)
                        accuracy,CVscores, rep = run_GB(x_train, y_train, x_test, y_test, C, iterations)
                        ac += round(accuracy,3)
                        #write_report3(report_file,round(accuracy,3),i,CVscores)
                        report.append('Training data:'+str(np.shape(x_train))+
                                       'Test data:'+str(np.shape(x_test))+'random_code:'+str(random_code)+'\n'+
                                       'Set'+str(i+1)+'='+str(round(accuracy,3))+'|'+
                                       'CV Accuracy(5)='+str(round(CVscores.mean(),3))+', CV s.d.='+str(round(CVscores.std(),3))+'\n')           
                    accu_av = round(ac/3,3)
                    print('Accuracy averaged over three random sets:{:.3f}'.format(accu_av))
                    toc=timeit.default_timer()
                    report.append('Average Accuracy='+str(accu_av)+'\n'+'Procesing time(sec)='+str(round(toc-tic,2))+'\n\n')
                             
                        
    write_report9(report_file,report)
           
            
#def main_cov_type():
   
     
    
       
if __name__ == '__main__':
    main()
    #main_cov_type()
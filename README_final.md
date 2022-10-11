# Company Acquistion status Prediction
in this project we analyzed the dynamics of how different set of attibutes like milestones and funding rounds etc. can impact the status od the company. 


## Understanding the Dataset
The dataset we are working on is Crunchbase-Companies data 

- The news dataset contains all the details related to the company including when it was created how much funding it got, how many relationships it had and much more.


- The label of the dataset is whether the company is **operating** ,  **closed**, **Acquired** or **ipo**.


## Preprocessing and EDA

We first **removed** all the irrelevant and redundant information by deleting all irrelevant columns and all those rows that were **duplicated**

Then we removed the noise and unreliable data from the dataset like removing outliers from **funding_rounds** and **funding_total_usd**
and removing all missing values from the dataset

We had to make some changes in the oiginal data like convert the **date** columns into **years**
and generalize the categorical data like **category_code** and **country_code**

We alson created a few new vaiables from already existing features to better understand the dataset 
- Create new feature isClosed from closed_at and status.
-Create new feature 'active_days'

We then replaced the null values of **Numerical Data** with mean value

Droped the remaining missing values

**Information of Dataset:**

Using countplot on target variable **status** we could see that **operating** has '38864' values, **Closed** has '2782' values, **Acquired** has '1183' values, **ipo** has '409' values. By this information we could conclude that there is imbalanced in the data and hence balancing of data is required.


## EDA
**Introduction:**

**Information of Dataset:**

Using countplot on target variable **status** we could see that **operating** has '38864' values, **Closed** has '2782' values, **Acquired** has '1183' values, **ipo** has '409' values. By this information we could conclude that there is imbalanced in the data and hence balancing of data is required.

**Analysis:**

Corelation graphs to tell if there were realtionship between how many days a company was active and its status proved that it is highly corelated.

boxplots to tell the outliers of the numerical variables o the dataset.

## Feature Engineering

We calcualted Mutual index scores to tell how much each feature was related to the the target variable then determined the 75th percentile of the scores of mutual index and considered only those above the 75th percentile 
value

We then used PCA to futher refine our datasets to get even better results but before applying PCA 
we blanced the dataset using SMOTE by oversampling the data 
Then we scaled the data using standad scaler to standarize our dataset
Then after that we finally applied PCA 
We then used MI_Scores again on the new features and only used the the ones with the highest scores and used the pca columns as our new features.

## Model Building

#### Metrics considered for Model Evaluation
**Accuracy , Precision , Recall and F1 Score**

- Accuracy: What proportion of actual positives and negatives is correctly classified?
- Precision: What proportion of predicted positives are truly positive ?
- Recall: What proportion of actual positives is correctly classified ?
- F1 Score : Harmonic mean of Precision and Recall


#### Feature for Input

Now, we splitted the new data to train of 80% and test of 20%, then scaled them using **StandardScaler**, too
We will take these ['founded_at', 'funding_rounds', 'funding_total_usd', 'milestones', 'relationships'] features as Input .


#### Ensemble Models

We use this anomaly detection techniques to first identify subset of majority class with high precision so the remaining subset has lower bias.
We will create single pipeline which main task is to ensemble two classifier Quadratic Discriminant Analysis(QDA) and Random Forest Classifier(RF) and run model.
Basically our major goal in ensembling part is to priortize accuracy on subset of data.
If we get 'Operating' as a output in QDA ,the final output will be same otherwise it will route to Random Forest Classifier and provide the respective inference according random forest.

                Accuracy: For LR accuracy is 89.34%

                precision    recall  f1-score   support

           0      0.273     0.005     0.011       560
           1      0.029     0.004     0.007       248
           2      0.375     0.120     0.182        75
           3      0.899     0.993     0.944      7765

    accuracy                          0.893      8648
   macro avg      0.394     0.281     0.286      8648
weighted avg      0.829     0.893     0.850      8648



#### Quadratic Discriminant Analysis(QDA)

**def** : QDA is a method you can use when you have a set of predictor variables and you’d like to classify a response variable into two classes.
We use QDA to first identify subset of 'Operating' classes so that the remaining data is more balanced.

                Accuracy: For LR accuracy is 88.84%

                precision    recall  f1-score   support

           0      0.230     0.083     0.122       808
           1      0.911     0.971     0.940      7840

    accuracy                          0.888      8648
   macro avg      0.571     0.527     0.531      8648
weighted avg      0.848     0.888     0.864      8648


                
#### Random Forest Classifier(RF)

**def** :The random forest is a classification algorithm consisting of many decision trees. It uses bagging and features randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree.
We use random forest classifier to classify remaining subset of data.

                 Accuracy: For LR accuracy is 88.6%

                 precision    recall  f1-score   support

           0      0.262     0.048     0.082       808
           1      0.837     0.005     0.010      7840
           2      0.000     0.000     0.000         0
           3      0.000     0.000     0.000         0

    accuracy                          0.009      8648
   macro avg      0.275     0.013     0.023      8648
weighted avg      0.783     0.009     0.017      8648



## Deployment

- After Finishing the pipelining and getting the best possible score, we move forward to the deployment phase.
- The first step for deployment was to Dump the best model through “pickle” (Library)
- Then, we created a basic website with Flask.
- To make our website on the internet we have deployed it with Heroku.


### Pickle

**def** :Pickle is a useful Python tool that allows you to save your models, to minimise lengthy re-training and allow you to share, commit, and re-load pre-trained machine learning models.
We trained our model and saved it in form of pickle files(.pkl).


### Flask 

**def** :We use flask because it has in-built tools, libraries, technologies which allow us to bulid web application easily. Website develop on your local computer, so to push it to live we setting up with heroku.
The basic approach was to load the Dumped model in Flask, get the user input and predict the output from that loaded model.
Our website contains two URLS (Home and Predict).
-The Home URL accepts all the input fields and the flow goes to Predict URL.
-Predict URL where we get the output after clicking the Predict button.


### Heroku

**def** :Heroku is a container-based cloud Platform as a Service (PaaS). Developers use Heroku to deploy, manage, and scale applications entirely in the cloud.
We deploy our flask application to heroku. In this way, we can share our app on the internet with others, we can access our app using this link [**status-prediction**](https://status-prediction-deploy.herokuapp.com/). 

We prepared the needed files to deploy our app sucessfully:
- Procfile: contains run statements for app file .
- requirements.txt: contains the libraries must be downloaded by Heroku to run app file successfully 
- ensemble.pkl: contains file which we used to combine both QDA and RF classifiers into single pipeline.
- qda.pkl :  contains our Quadratic Discriminant Analysis Classifier model.
- mod_rf.pkl : contains our Random Forest Classifier model that built by modeling part.




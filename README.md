# ML-Finding-Donors-Kaggle-competition

### Description
This is one of my first Kaggle competitions where I apply supervised learning techniques and an analytical mind on data collected for the U.S. census to help CharityML (a fictitious charity organization) identify people most likely to donate to their cause. I first explored the data to learn how the census data is recorded. Next, I applied a series of transformations and preprocessing techniques to manipulate the data into a workable format. I then evaluated several supervised learners of your choice on the data and consider which is best suited for the solution. Afterwards, I optimized the model I've selected and present it as my solution to CharityML. Finally, I explored the chosen model and its predictions under the hood, to see just how well it's performing when considering the data it's given.
The success of the model has been determined based on the models AUC or area under the curve associated with ROC curves.


### Files

- census.csv (45222, 14): this is the training data
- test_census (45222, 14): this is the data provided by Kaggle to test and score the model

### Libraries
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Sklearn
- XGBoost
- Optuna

### Wrangling and Cleaning

The target column has a significant imbalance in favor of the <=50K class, which will compromise the model training especially for the lower rapresented class.
To overcome this problem I can:
* Use learners capable of dealing with inbalances (i.e. XGBoost)
* Use oversampling techniques such us SMOTE to create an omogeneous trainig dataset

Columns *capital-gain* and *capital-loss* are highly skewed. I used a logarithmic transformation to significantly reduce the range of values caused by outliers. However, I first had to transoform the values by a small amount above 0 due to the fact that the logarithm of 0 is undefined.

Upon inspection of test_census.csv, I found an extra field named *Unamed: 0* and also that every other features contains missing values.
I dropped the extra column and filled the missing values with the training dataset most frequent value for that same column.

### Modeling

I check the performances of AdaBoost, Random Forest, XGBoost and SVM on the training data:
* **XGBoost:** is the best perfoming learner overall
* **AdaBoost:** has a better performance for the recall than the precision. Good also the ROC value
* **RandomForest:** is the second best performing learner overall. Similar scores to AdaBoost, just a tiny better
* **SVM:** the best recall score of the four learners, but the worse precision.Good value for the ROC, however the training takes significantly more time.

I then optimized AdaBoost, Random Forest, XGBoost with Optuna and stacked them for more accurate predictions.

### Conclusion

My model scored 0.94555 on the [Kaggle Leaderboard](https://www.kaggle.com/c/udacity-mlcharity-competition/leaderboard).

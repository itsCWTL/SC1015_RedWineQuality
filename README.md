# SC1015_RedWineQuality
Lab: FCS1<br>
Group: 5<br>

Members:
1. Christine Wong Tien Lee
2. Eileen Lim Jing Wen


# Dataset
Red wine quality

# Table of contents:
1. Problem Statement
2. Data Preparation and Cleaning
3. Exploratory Data Analysis/Visualization
4. Machine Learning Techniques
5. Data Driven Insights
6. References

# 1. Problem Statement
How can companies predict the quality of their red wine to prevent the sale of bad-quality red wines to customers, and which physicochemical factors have the most significant impact on determining the quality of red wine?

# 2. Data Preparation and Cleaning
**1. Checking missing values:** <br>
The returned results for each variables are 0, which shows that no empty cells were detected in the dataset. <br>
<br>
**2. Reclassifying quality:**<br>
Any value in ‘quality’ that contains a value of less then 7 is considered “bad” and any value that is more than or equal to 7 is considered “good”.<br>
<br>
# 3. Exploratory Data Analysis/Visualization
**1. Boxplot, Histogram, Violinplot** showing the distribution of data for each variable:<br>
We observed that the data contains a significant number of widely spread outliers. These outliers should be kept as part of the dataset, as these are natural variations in the population that are known as true outliers and could potentially represent a deeper cause that is relevant to the analysis. <br>
<br>
**2. Bar chart** showing frequency of “good” and “bad” wines: <br>
The bad wines makes up the majority of all the wines, or approximately 1400, while the remaining 200 are of good quality.<br>
<br>
**3. Boxplot and Scatterplot** comparing the good and bad wines:<br>
We constructed boxplots for each independent variable to showcase the different characteristics that make a wine good or bad. In here, we place the boxplot for the bad wine directly above the boxplot representing the good wine for each variable.<br>
<br>
**4. Correlation matrix** evaluating the relationship between all the variables except ‘quality’: <br>
Multicollinearity is detected! For example, “fixed acidity” has multiple moderately positive correlations with other variables such as “citric acid” and density. Besides, comparing “fixed acidity” to “pH” and comparing “volatile acidity” and “citric acid” also both give moderately negative correlations.<br>
<br>
**5. Pairplot** enabling the visualization of the relationship between each pair of variables except ‘quality’:<br>
Multicollinearity is detected! It allows us to spot and differentiate between the high-correlation relationships and low-correlation relationships faster, helping us to generate insights easier. For instance, there is some independent variables are moderately correlated with each other.<br>
<br>
# 4. Machine Learning Techniques
**1. Principal Component Analysis (PCA)**<br>
We decided to extract **5 components** with **variance explained around 85%**. The predictors for each principal component are described as follows:<br>
PC 1: fixed acidity, citric acid, density, and pH, which collectively represent the acid content of the wine.<br>
PC 2: free sulfur dioxide and total sulfur dioxide, which indicate the sulfur dioxide content of the wine.<br>
PC 3: volatile acidity and alcohol, which represent the volatile content of the wine.<br>
PC 4: chloride and sulphates, which indicate the chloride and sulphates content of the wine.<br>
PC 5: residual sugar, which represents the residual sugar content within the wine.<br>
<br>
**2. Classification** <br>
&emsp;**Step 1: Finding the best classification model**<br>
&emsp;&emsp;**i.Decision tree**<br>
&emsp;&emsp;Accuracy: 0.87<br>
&emsp;&emsp;Precision(Good): 0.51<br>
&emsp;&emsp;f1-score(Good): 0.54<br>
&emsp;&emsp;FPR: 0.086<br>
<br>
&emsp;&emsp;**ii. Random forest**<br>
&emsp;&emsp;Accuracy: 0.915<br>
&emsp;&emsp;Precision(Good): 0.83<br>
&emsp;&emsp;f1-score(Good): 0.59<br>
&emsp;&emsp;FPR: 0.014<br>
<br>
&emsp;&emsp;**iii. XGBoost**<br>
&emsp;&emsp;Accuracy: 0.9075<br>
&emsp;&emsp;Precision(Good): 0.71<br>
&emsp;&emsp;f1-score(Good): 0.59<br>
&emsp;&emsp;FPR: 0.032<br>
<br>
&emsp;&emsp;We selected the **Random Forest** model due to its superior performance among the three classification models, with an accuracy of 0.915. 
&emsp;&emsp;The precision for predicting good quality wine is 0.83, and the f1 score is 0.59. Additionally, the Random Forest model achieved the lowest 
&emsp;&emsp;false positive rate of 0.014. This indicates that the model is effective in minimizing the risk of incorrectly classifying bad quality wine as
good quality and subsequently selling it to customers.<br>
<br>
&emsp;**Step 2: Tuning**<br>
&emsp;&emsp;**i. Grid Search**<br>
&emsp;&emsp;Given the imbalance in the data, where the bad quality is significantly more prevalent than the good quality, there is a risk of overfitting. To \
&emsp;&emsp;mitigate this, we further improved the selected random forest model by utilizing Grid Search to optimize hyperparameters such as 
&emsp;&emsp;'n_estimators' and 'max_depth'.<br>
<br>
&emsp;&emsp;After hyperparameter tuning for 'n_estimators' and 'max_depth':<br>
<br>
&emsp;&emsp;Accuracy: 0.92<br>
&emsp;&emsp;Precision (Good): 0.89<br>
&emsp;&emsp;f1-score (Good): 0.60<br>
&emsp;&emsp;False Positive Rate (FPR): 0.009<br>
<br>
&emsp;&emsp;This indicates that after tuning, the Random Forest model can enhance both its accuracy and generalization performance, making it more 
&emsp;&emsp;robust to potential data drift.<br>
<br>

**3. Feature importance**<br>
Top 5 important features in wine quality are:<br>
-alcohol <br>
-sulphates<br>
-volatile acidity<br>
-citric acid<br>
-density<br>
<br>

# 5. Data Driven Insights<br>
Companies can use the tuned Random Forest model to accurately predict the quality of their red wine.By doing so, they can effectively prevent the sale of bad quality red wines to customers. This proactive approach not only safeguards the company's reputation for delivering good quality products but also enhances customer satisfaction. When customers receive consistent quality, it fosters trust and loyalty, leading to positive reviews. As a result, this can indirectly boost sales, as satisfied customers are more likely to make repeat purchases and recommend the company's wines to others.<br>
<br>
In order to produce good quality red wines, companies can focus more on the top 5 physicochemical factors. For instances, if a large quantity of bad quality wine is produced, the company can first examine these key factors to identify the issue. Nonetheless, the other physicochemical factors are also important, as even small variations can significantly influence the overall quality and taste of the wine. Therefore, a comprehensive understanding and careful monitoring of all these factors are essential for consistently producing good quality red wines.<br>
<br>

# 6. References
1. https://muthu.co/understanding-the-classification-report-in-sklearn/
2. https://medium.com/@utsavrastogi/red-wine-quality-prediction-a-beginners-guide-for-a-simple-classification-project-7183dae6101d
3. https://allysonf.medium.com/predict-red-wine-quality-with-svc-decision-tree-and-random-forest-24f83b5f3408
4. https://www.kaggle.com/code/rudolfjason/5-most-important-factors-in-red-wine-quality
5. https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
6. https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Clustering-Dimensionality-Reduction/Principal%20Component%20Analysis.ipynb
7. https://www.kaggle.com/code/rprkh15/red-wine-quality-xgboost-optuna-neural-networks



# Capstone Project

#### Machine Learning Engineer Nanodegree

Chenxiang Li

## Predicting Uber Rider Retention

## I. Definition
### Project Overview
Uber is interested in predicting rider retention. It would be very helpful for them to know what factors are most important for rider retention. To help explore this problem, they have provided a sample [dataset](https://www.dropbox.com/s/q5e3lqtma9dwmy6/uber_data_challenge.json?dl=0) of 50,000 users who signed up for an Uber account in January 2014. 

In this binary prediction project, I will apply classification algorithms in Python to predict Uber rider retention and explore feature importance. Eventually, I will provide data-driven suggestions to operationalize those insights to help Uber.

### Problem Statement
I would consider predicting rider retention as a supervised binary classification problem. The ultimate goal of this project is to find an machine learning algorithm to predict current rider retention and optimize retention rate by finding important features.

In this project, I will consider a rider retained if he/she was “active” (i.e. took a trip) in the preceding 30 days. Because the data was pulled several months later, I assumed the current date is `"2014-07-01"` and a user retained if the `last_trip_date` is after `"2014-06-01"`.

- I will complete data cleaning to fill missing values, remove outliers and also preprocess dataset for algorithm implementation.
- In exploratory phase, I will check basic statistics and rider segregation and train a Logistic Regression model as a benchmark.
- As for modeling, I will try Decision Tree, Random Forest and Support Vector Machine (SVM) classifiers to see which performs best on my training set. I will choose one algorithm for further reach to tune the respective parameters. 
- Finally, I will validate my model by cross-validation or on test set. Also, I will check the feature importance in the final model to provide suggestions for Uber.

### Dataset description:

* city: city this user signed up in
* phone: primary device for this user
* signup_date: date of account registration; in the form ‘YYYY­MM­DD’
* last_trip_date: the last time this user completed a trip; in the form ‘YYYY­MM­DD’ 
* avg_dist: the average distance *(in miles) per trip taken in the first 30 days after signup 
* avg_rating_by_driver: the rider’s average rating over all of their trips 
* avg_rating_of_driver: the rider’s average rating of their drivers over all of their trips 
* surge_pct: the percent of trips taken with surge multiplier > 1
* avg_surge: The average surge multiplier over all of this user’s trips 
* trips_in_first_30_days: the number of trips this user took in the first 30 days after signing up
* uber_black_user: TRUE if the user took an Uber Black in their first 30 days; FALSE otherwise
* weekday_pct: the percent of the user’s trips occurring during a weekday

### Metrics
As this is a binary classification problem with 50,000 samples, I will use Area Under the Receiver Operating Characteristic curve (AUROC) to measure performance of a model or result in this project. This score tells me the ability of my model to distinguish the two classes. Intuitively, given a random new user, AUROC is the probability that my model can predict correctly on it will be retained or not.

On the other side, AUROC is independent of the fraction of the test population which is class 0 or class 1. This makes AUROC not sensitive to unbalanced dataset. In this case, the retention rate is very likely something far below 50%, so AUROC will work well to evaluate model performance.

*In concept, ROC curve is plot of the true positive rate from confusion matrix VS the false positive rate as the threshold value for classifying an item as 0 or 1 is increased from 0 to 1: if the classifier is very good, the true positive rate will increase very quickly and the area under the curve will be close to 1. If the classifier is no better than random guessing, the true positive rate will increase linearly with the false positive rate and the area under the curve will be around 0.5, which is the probabilty of random guessing.*

![auroc align='center'](http://i.stack.imgur.com/9NpXJ.png)_

## II. Analysis
### Data Exploration
The original dataset is in JSON format. It was into Python and easily parsed into a `dataframe` object. It contains 50,000 rows and 12 columns as described above. Each row represents a user behavior. 

![df_head](images/df_head.png)

At the first glimpse of the dataset, there are 7 numerical variables, 3 categorical variables and two datetime stamps. 

|           | avg_dist | avg_rating_by_driver | avg_rating_of_driver | avg_surge | surge_pct | trips_in_first_30_days | weekday_pct |
| --------- | -------- | -------------------- | -------------------- | --------- | --------- | ---------------------- | ----------- |
| **count** | 50000    | 49799                | 41878                | 50000     | 50000     | 50000                  | 50000       |
| **mean**  | 5.796827 | 4.778158             | 4.601559             | 1.074764  | 8.849536  | 2.2782                 | 60.926084   |
| **std**   | 5.707357 | 0.446652             | 0.617338             | 0.222336  | 19.958811 | 3.792684               | 37.081503   |
| **min**   | 0        | 1                    | 1                    | 1         | 0         | 0                      | 0           |
| **25%**   | 2.42     | NaN                  | NaN                  | 1         | 0         | 0                      | 33.3        |
| **50%**   | 3.88     | NaN                  | NaN                  | 1         | 0         | 1                      | 66.7        |
| **75%**   | 6.94     | NaN                  | NaN                  | 1.05      | 8.6       | 3                      | 100         |
| **max**   | 160.96   | 5                    | 5                    | 8         | 100       | 125                    | 100         |

Then, I checked basic  statistics for all the 7 numerical variables. There are missing values in `avg_rating_by_driver` and `avg_rating_of_driver`. I also noticed that the *standard deviations* of `surge_pct` and `trips_in_first_30_days` are extremely large with respect to *means*, while other columns may still show abnomalities. I may need to deal with missing values and outliers in these columns.

**Missing value:** After counted missing values in all columns, the good thing is that there exists missing values only in `avg_rating_by_driver`(201), `avg_rating_of_driver`(8122) and `phone`(396). I filled missing values in `avg_rating_by_driver` and `avg_rating_of_driver` with respective `median` values and removed 396 samples with missing value in `phone`. 

**Outlier:** On boxplots of those 7 numerical variables, I confirmed my initial guess from the statistics above. To make it more straightforward, I visualized the data in two separate sets of boxplot on different scales.

![boxplot1](images/boxplot1.png) ![boxplot2](images/boxplot2.png)

I also counted the outliers out of 1.5×IQR as below. If I drop any sample with an outlier for one column, I may lost more than half the orginal training data. I decided to remove the 7805 samples considered outliers for more than one feature. 
```python
Outliers for 'avg_dist': 4477
Outliers for 'avg_rating_by_driver': 3922
Outliers for 'avg_rating_of_driver': 3106
Outliers for 'avg_surge': 8369
Outliers for 'surge_pct': 6768
Outliers for 'trips_in_first_30_days': 3153
Outliers for 'weekday_pct': 0
```

For other non-numerical variables, I checked unique values of `city` and `phone` and ranges for `signup_date` and `last_trip_date`. It shows all users are located in three different city: "King's Landing", "Astapor" and "Winterfell" and they all use "iPhone" or "Android" cellphone. As the project background, all the users signed up for an Uber account in January 2014 and took a last trip between 2014-01-01 and 2014-07-01.

**Target variable:** As mentioned before, I assume the current date is "2014-07-01" and a user retained if the `last_trip_date` is after "2014-06-01", which means this user is still considered an "active" user. Therefore, I created a boolean variable `active` as target variable indicating if the user is retained after signed up. Because this target variable was derived from `last_trip_date`, I droped this column.

### Exploratory Visualization

In this section, I will explore the relationships behind the data by some plots. First, I will generate a pair scatter plot to see if there is any multicollinearity or mutual dependencies among the predicting variables.

![pairplot.png](images/pairplot.png)

As can be seen from the pair plot above, there must be some correlation between `avg_surge` and `surege_pct`, `avg_rating_of_driver` and `surege_pct`.   A correlation matrix is more straightforward to tell the correlation between variables.

The pair plot also tells me that all these numerical variable don't follow a normal distribution. They are highly skewed. This indicates that the data mostly lies in a narrow range. 

![corrmatrix.png](images/corrmatrix.png)

From the correlation matrix above, it shows that `surge_pct` is correlated with `avg_surge`, which might be resulted from a Uber internal surge pricing algorithm. The good thing is that correlated features would not directly affect classification performance. However, considering the Curse of Dimensionality, these two correlated variables may potentially affect classifier performance. In the modeling phase, I may try a non-parametric classifier like **k-Nearest Neighbors (KNN)** into initial comparison.

For categorical variables, I can get a general idea on these three features. On the city plot, it shows that the differences among the retention rates of these three given cities are significant. The city “King’s Landing” might be a new market, which has the smallest amount of users. It owns a very good retention rate. “Astapor” might be similar to “Winterfell”, while it performs a litter worse than “Winterfell”, which might be a relative developed market.

As for `phone` and `uber_black_user`, the plots below show that most Uber rider use "iphone" to request a ride. At the same time, I guess the Uber APP on iPhone probabaly provides a better user experience to lead a user keeping use Uber. Uber black provides definitely a better riding experience. As expected, an Uber black user is more likely to be retained. 

![city.png](images/city.png) ![phone.png](images/phone.png)![black.png](images/black.png)

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?

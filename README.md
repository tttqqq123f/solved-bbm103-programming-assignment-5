Download Link: https://assignmentchef.com/product/solved-bbm103-programming-assignment-5
<br>



<a href="https://classroom.github.com/a/lqCwbS_H"><em>Click here to accept your Programming Assignment 5.</em></a>

<h1>1        Introduction</h1>

In this experiment, you will implement a program to solve a real wold machine learning problem with a real world data. In this experiment, you will get familiar Python data science libraries such as; Pandas, NumPy, and Matplotlib. At the end, you will build 3 types of machine learning models.

<h1>2        BackGround (Technical Debt)</h1>

Technical debt (TD) is a metaphor used to describe the lack of software quality in the product. It provides a valuable indicator to track software product quality throughout development and maintenance. It defines the invariance of delayed technical development activities to receive short-term repayments. The consequences related to these skipped activities are accumulated in the software, and it is called as ’debt.’ If there is too much technical debt that is accumulated, it causes low software quality, pointing to the initiation of design and code quality problems. Because development will slow down, maintainability of the software will be difficult. To fix these quality problems, it requires extra effort for their mitigation. TD is a measure of the effort needed for fixing these problems in the future and used as a measure of quality. Therefore, the higher value of TD for a software product means more unresolved quality problems included in, and lower overall quality. Since measuring TD directly can be difficult, we propose to analyze whether there is a relation between TD with internal and external metrics.

<h1>3        Dataset</h1>

In this assignment, we provide you a real world dataset <a href="https://zenodo.org/record/4265301#.X84QNLOhlPY"><em>(Click here to download the dataset.) </em></a>that consists of information about 50 open source Android project. In this dataset, you will have 16 different metrics of the each project. We divide these metrics as ”external”, ”internal”, and ”TD” metrics.

<strong>TD metrics that were measured:</strong>

<strong>Metric1</strong>: Code Duplication Ratio (CDR): Density of duplicated line of code

<strong>Metric2</strong>: Technical Debt (TD):Ratio between the cost to develop the code changed in the new code period and the cost of the issues linked to it

<strong>External metrics that were measured:</strong>

<strong>Metric3</strong>: Number of Bugs (NoB): It gives information about the number of bug

<strong>Metric4</strong>: Vulnerabilities (V): It gives information about the number of vulnerability

<strong>Metric5</strong>: Security Hotspots (SH): It is about number of security hotspots

Programming Assignment 5       1 <strong>Metric6</strong>: Code Smells (CS): Total count of code smell issues.

<strong>Internal metrics that were measured:</strong>

<strong>Metric7</strong>: Number of Children (NOC): number of subclasses that are linked to a class in the hierarchy

<strong>Metric8</strong>: Coupling between Object Classes (CBO): number of classes coupled to a given class

<strong>Metric9</strong>: Lack of Cohesion in Methods (LCOM): Let us assume that class C1 has a set of M1, M2,…Mn methods and that the set is a set of attribute variables used in the Mi method. In this case, LCOM is the number of discrete clusters that are the intersection of these n clusters.

<strong>Metric10</strong>: Fanin: a measure of how many other classes use the specific class. It indicates the number of classes to be affected if class changes. It can also be expressed as internal coupling. <strong>Metric11</strong>: FanOut: Indicates how many classes are coupled on the class being examined. It refers to the external coupling of a class.

<strong>Metric12</strong>: Response for a Class (RFC): number of different methods that can be executed when an object of that class receives a message (when a method is invoked for that object) <strong>Metric13</strong>: Depth of Inheritance Tree (DIT): maximum level of inheritance hierarchy of a class

<strong>Metric14</strong>: Weighted Method Per Class (WMC): sum of the complexity of all methods of a class

<strong>Metric15</strong>: Line of Code (LOC): It gives information about the size of the software. <strong>Metric16</strong>: Comment Lines of Code (CLOC): It is the number of comment lines written for the program.

Figure 1: A small part of dataset

The aim of this assignment is;

<strong>Part1</strong>– to find the relation between ”TD” metrics with ”internal” and ”external” metrics by using statistical correlation analysis.

<strong>Part2</strong>– making a TD estimation by using ML Regression models.

<h1>4        Packages</h1>

<ul>

 <li>numpy is the fundamental package for scientific computing with Python.</li>

 <li>matplotlib is a library to plot graphs in Python.</li>

 <li>pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.</li>

 <li>Scikit-learn is a free machine learning library for Python. It features various algorithms like support vector machine, random forests, and k-neighbours, and it also supports Python numerical and scientific libraries like NumPy and SciPy.</li>

 <li>Seaborn is a Python data visualization library based on matplotlib. It provides a highlevel interface for drawing attractive and informative statistical graphics</li>

</ul>

<h1>5        Problem</h1>

<h2>5.1       Part1 (Statistical Correlation Analysis)</h2>

Correlation analysis is a statistical method used to evaluate the strength of relationship between two quantitative variables. A high correlation means that two or more variables have a strong relationship with each other, while a weak correlation means that the variables are hardly related. We will examine two types of correlation analysis method: Spearman and Pearson. The fundamental difference between the two correlation coefficients is that the Pearson coefficient works with a linear relationship between the two variables whereas the Spearman Coefficient works with monotonic relationships as well. In this part, it is expected you to analyze the correlation of ”TD” with ”internal” and ”external” metrics. Before analyzing the correlation, firstly, you will find the distribution of data. After finding the distribution of data, you will decide which correlation type you will choose; Spearman or Pearson correlation analysis.

<strong>REQUIREMENTS FOR PART1</strong>

<strong>Step1: </strong>Show the distribution of 3 metrics that can be evidence for choosing appropriate correlation analysis type (Spearman or Pearson)

<strong>Step2: </strong>Since we have different data types and ranges in our data set, apply min-max normalization to all the data we have

<strong>Step3: </strong>Show the correlation matrix of all metrics

<strong>Step4: </strong>Show p values of correlation tables

<em>Note: The P-value is the probability that you would have found the current result if the correlation coefficient were in fact zero (null hypothesis). If this probability is lower than the conventional 5% (P0.05) the correlation coefficient is called statistically significant.</em>

<strong>Step4: </strong>Show heatmap of correlation matrix

<strong>Step5: </strong>Show correlation between External Metrics &amp; TD

<strong>Step6: </strong>Show correlation between Internal Metrics &amp; TD

<h2>5.2       Part2 (ML Modeling)</h2>

You will use the techniques of a subfield of computer science; which is machine learning. Machine learning can simlpy be defined as building a statistical model from a dataset in order to solve a real world problem. In this part, you will make a Technical Debt Estimation By Using ML Regression Models.

There are types of machine learning. However in this assignment, we expect you using 5 different ML Regression Models as shown below:

<ul>

 <li>Linear Regression: is a linear approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). The case of one explanatory variable is called simple linear regression.</li>

 <li>Support Vector Regression: SVR gives us the flexibility to define how much error is acceptable in our model and will find an appropriate line (or hyperplane in higher dimensions) to fit the data.</li>

 <li>Decision Tree Regression: builds regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes.</li>

 <li>Random Forest Tree Regression: is an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean/average prediction (regression) of the individual trees</li>

 <li>MultiLayer Perceptron Regression: is a supervised algorithm that models neural network.</li>

</ul>

<strong>REQUIREMENTS FOR PART 2</strong>

You will read the dataset with Pandas library. Pandas is a data structure and data analysis tool for Python.

<strong>Step1: </strong>Split Data Into Train and Test Sets: You are expected to split your dataset into 2 sets; train and test sets. The training set is used to train the model. In other words, your model will learn from the training set and tune the weights. For the evaluation purposes; you will use the test set. You will obtain accuracy results on test set. You will use 70% of the data for training set and 30% of the data for test set. <strong>Step2: </strong>Define the functions of 5 ML model

<em>def linearRegression(X train,y train,X test,y test): def svrRegression(X </em><em>train,y train,X test,y test): def decisionTreeRegression(X train,y train,X test,y test): def randomDecTreeRegression(X train,y train,X test,y test): def mlpRegressor(X train,y train,X test,y test)</em>

<strong>Step3: </strong>Show machine learning models that estimate Technical Debt using only internal metrics.

<strong>Step4: </strong>Show machine learning models that estimate Technical Debt using only external metrics.

<strong>Step5: </strong>Show machine learning models that estimate Technical Debt using all(external and internal) metrics.

<strong>Note: You can use the given code for all 2 parts of the assignment. It is important to note that the given code usage is optional.</strong>

The aim of this assignment is:

<ol>

 <li>Learning to use libraries</li>

 <li>Learning to understand and analyse a given problem</li>

 <li>The resulting accuracy percantage is not important. The important part is to understandthe problem and implementing it.</li>

</ol>



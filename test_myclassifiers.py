import numpy as np
import scipy.stats as stats 
import mysklearn.myutils as myutils

from mysklearn.myclassifiers import MySimpleLinearRegressor, MyKNeighborsClassifier, MyNaiveBayesClassifier, MyZeroRClassifier, MyRandomClassifier

# note: order is actual/received student value, expected/solution
def test_simple_linear_regressor_fit():
     # Test 1
    np.random.seed(0)
    x = list(range(0,100))
    y = [value * 2 + np.random.normal(0,25) for value in x]
    
    mslr = MySimpleLinearRegressor()
    mslr.fit(x,y)
    sklearn_line = stats.linregress(x, y)
    assert np.allclose(mslr.slope, sklearn_line.slope)
    assert np.allclose(mslr.intercept, sklearn_line.intercept)

    # Test 2
    np.random.seed(10)
    x2 = list(range(0,100))
    slope = np.random.normal(20,88)
    y2 = [value2 * slope + np.random.normal(0,25) for value2 in x2]
    
    mslr.fit(x2,y2)
    sklearn_line2 = stats.linregress(x2, y2)
    assert np.allclose(mslr.slope, sklearn_line2.slope)
    assert np.allclose(mslr.intercept, sklearn_line2.intercept)

def test_simple_linear_regressor_predict():
    mslr = MySimpleLinearRegressor()
    
    # Test 1, simple slope of 3
    x = [0, 1, 2, 3, 4, 5]
    y = [0, 3, 6, 9, 12, 15]
    mslr.fit(x,y)
    X_test = [[6],[7]]
    y_predicted = mslr.predict(X_test)
    assert(y_predicted == [18, 21]) 

    # Test 2, float slope test 
    x2 = [12, 45, 67, 34, 2]
    y2 = [3, 11, 23, 9, .6]
    mslr.fit(x2,y2)
    X_test2 = [[85],[44]]
    y_predicted2 = mslr.predict(X_test2)
    #print("y_predict:", y_predicted2)
    assert(y_predicted2 == [26.764032616753152, 13.269592290585619])

def test_kneighbors_classifier_kneighbors():
    mknc = MyKNeighborsClassifier()
    #print()
    # Test 1, 4 instance training set example traced in class on the iPad
    x = [
        [7,7], 
        [7,4], 
        [3,4], 
        [1,4]
    ]
    y = ["bad", "bad", "good", "good"]
    mknc.fit(x,y)
    test = [[3,7]]
    distances, indices = mknc.kneighbors(test)
    #print("distances: ", distances)
    #print("indices: ", indices)
    assert(distances == [[0.6666666666666667, 1.0, 1.0540925533894598]])
    assert(indices == [[0, 2, 3]])

    # Test 2, Use the 8 instance training set example from ClassificationFun/main.py
    x2 = [
        [3,2],
        [6,6],
        [4,1],
        [4,4],
        [1,2],
        [2,0],
        [0,3],
        [1,6]
    ]
    y2 =["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    mknc.fit(x2,y2)
    test2 = [[2, 3]]
    distances2, indices2 = mknc.kneighbors(test2)
    #print("distances2: ", distances2)
    #print("indices2: ", indices2)
    assert(distances2 == [[0.23570226039551584, 0.23570226039551587, 0.3333333333333333]])
    assert(indices2 == [[4, 0, 6]])

    # Test 3, Use Bramer 3.6 Self-assessment exercise 2
    x3 = [
        [.8,6.3],
        [1.4,8.1],
        [2.1,7.4],
        [2.6,14.3],
        [6.8,12.6],
        [8.8,9.8],
        [9.2,11.6],
        [10.8,9.6],
        [11.8,9.9],
        [12.4,6.5],
        [12.8,1.1],
        [14.0,19.9],
        [14.2,18.5],
        [15.6,17.4],
        [15.8,12.2],
        [16.6,6.7],
        [17.4,4.5],
        [18.2,6.9],
        [19.0,3.4],
        [19.6,11.1]
    ]
    y3 = ['-','-','-','+','-','+','-','+','+','+','-','-','-','-','-','+','+','+','-','+']
    mknc.__init__(5)
    mknc.fit(x3,y3)
    test3 = [[9.1, 11.0]]
    distances3, indices3 = mknc.kneighbors(test3)
    #print("distances3: ", distances3)
    #print("indices3: ", indices3)
    assert(distances3 == [[0.032355119842011795, 0.06579423870666472, 0.1171421039656662, 0.14903112474597766, 0.15507850784163038]])
    assert(indices3 == [[6, 5, 7, 4, 8]])

def test_kneighbors_classifier_predict():
    mknc = MyKNeighborsClassifier()
    #print()
    # Test 1, 4 instance training set example traced in class on the iPad
    x = [
        [7,7], 
        [7,4], 
        [3,4], 
        [1,4]
    ]
    y = ["bad", "bad", "good", "good"]
    mknc.fit(x,y)
    test = [[3,7]]
    y_predicted = mknc.predict(test)
    #print("y_predicted: ", y_predicted)
    assert(y_predicted == ['good'])

    # Test 2, Use the 8 instance training set example from ClassificationFun/main.py
    x2 = [
        [3,2],
        [6,6],
        [4,1],
        [4,4],
        [1,2],
        [2,0],
        [0,3],
        [1,6]
    ]
    y2 =["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    mknc.fit(x2,y2)
    test2 = [[2, 3]]
    y_predicted2 = mknc.predict(test2)
    #print("y_predicted2: ", y_predicted2)
    assert(y_predicted2 == ['no'])
    
    # Test 3, Use Bramer 3.6 Self-assessment exercise 2
    x3 = [
        [.8,6.3],
        [1.4,8.1],
        [2.1,7.4],
        [2.6,14.3],
        [6.8,12.6],
        [8.8,9.8],
        [9.2,11.6],
        [10.8,9.6],
        [11.8,9.9],
        [12.4,6.5],
        [12.8,1.1],
        [14.0,19.9],
        [14.2,18.5],
        [15.6,17.4],
        [15.8,12.2],
        [16.6,6.7],
        [17.4,4.5],
        [18.2,6.9],
        [19.0,3.4],
        [19.6,11.1]
    ]
    y3 = ['-','-','-','+','-','+','-','+','+','+','-','-','-','-','-','+','+','+','-','+']
    mknc.__init__(5)
    mknc.fit(x3,y3)
    test3 = [[9.1, 11.0]]
    y_predicted3 = mknc.predict(test3)
    #print("y_predicted3: ", y_predicted3)
    assert(y_predicted3 == ['+'])

def test_naive_bayes_classifier_fit():
    table = [
        [1,5,'yes'],
        [2,6,'yes'],
        [1,5,'no'],
        [1,5,'no'],
        [1,6,'yes'],
        [2,6,'no'],
        [1,5,'yes'],
        [1,6,'yes']
    ]
    myb = MyNaiveBayesClassifier()
    x_train = [row[:-1] for row in table]
    y_train = [row[-1] for row in table]
    myb.fit(x_train, y_train)
    assert myb.priors['yes'] == 5/8
    assert myb.priors['no'] == 3/8
    assert myb.posteriors['yes'][0][1] == 4/5
    assert myb.posteriors['no'][0][1] == 2/3
    assert myb.posteriors['yes'][0][2] == 1/5
    assert myb.posteriors['no'][0][2] == 1/3
    assert myb.posteriors['yes'][1][5] == 2/5
    assert myb.posteriors['no'][1][5] == 2/3
    assert myb.posteriors['yes'][1][6] == 3/5
    assert myb.posteriors['no'][1][6] == 1/3
 
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]
 
    x_train = [row[:-1] for row in iphone_table]
    y_train = [row[-1] for row in iphone_table]
    myb.fit(x_train, y_train)
    assert myb.priors['no'] == 1/3
    assert myb.priors['yes'] == 2/3
    assert myb.posteriors['yes'][0][1] == 2/10
    assert myb.posteriors['no'][0][1] == 3/5
    assert myb.posteriors['yes'][0][2] == 8/10
    assert myb.posteriors['no'][0][2] == 2/5
    assert myb.posteriors['yes'][1][1] == 3/10
    assert myb.posteriors['no'][1][1] == 1/5
    assert myb.posteriors['yes'][1][2] == 4/10
    assert myb.posteriors['no'][1][2] == 2/5
    assert myb.posteriors['yes'][1][3] == 3/10
    assert myb.posteriors['no'][1][3] == 2/5
    assert myb.posteriors['yes'][2]['fair'] == 7/10
    assert myb.posteriors['no'][2]['fair'] == 2/5
    assert myb.posteriors['yes'][2]['excellent'] == 3/10
    assert myb.posteriors['no'][2]['excellent'] == 3/5
 
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"], 
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"]
    ]
    x_train = [row[:-1] for row in train_table]
    y_train = [row[-1] for row in train_table]
    myb.fit(x_train, y_train)
 
    assert myb.priors['on time'] == 14/20
    assert myb.priors['late'] == 2/20
    assert myb.priors['very late'] == 3/20
    assert myb.priors['cancelled'] == 1/20
 
    assert myb.posteriors["on time"][0]["weekday"] == 9/14
    assert myb.posteriors["on time"][0]["saturday"] == 2/14
    assert myb.posteriors["on time"][0]["sunday"] == 1/14
    assert myb.posteriors["on time"][0]["holiday"] == 2/14
    assert myb.posteriors["on time"][1]["spring"] == 4/14
    assert myb.posteriors["on time"][1]["summer"] == 6/14
    assert myb.posteriors["on time"][1]["autumn"] == 2/14
    assert myb.posteriors["on time"][1]["winter"] == 2/14
    assert myb.posteriors["on time"][2]["none"] == 5/14
    assert myb.posteriors["on time"][2]["high"] == 4/14
    assert myb.posteriors["on time"][2]["normal"] == 5/14
    assert myb.posteriors["on time"][3]["none"] == 5/14
    assert myb.posteriors["on time"][3]["slight"] == 8/14
    assert myb.posteriors["on time"][3]["heavy"] == 1/14
    assert myb.posteriors["late"][0]["weekday"] == 1/2
    assert myb.posteriors["late"][0]["saturday"] == 1/2
    assert myb.posteriors["late"][1]["winter"] == 2/2
    assert myb.posteriors["late"][2]["high"] == 1/2
    assert myb.posteriors["late"][2]["normal"] == 1/2
    assert myb.posteriors["late"][3]["none"] == 1/2
    assert myb.posteriors["late"][3]["heavy"] == 1/2
    assert myb.posteriors["very late"][0]["weekday"] == 3/3
    assert myb.posteriors["very late"][1]["autumn"] == 1/3
    assert myb.posteriors["very late"][1]["winter"] == 2/3
    assert myb.posteriors["very late"][2]["high"] == 1/3
    assert myb.posteriors["very late"][2]["normal"] == 2/3
    assert myb.posteriors["very late"][3]["none"] == 1/3
    assert myb.posteriors["very late"][3]["heavy"] == 2/3
    assert myb.posteriors["cancelled"][0]["saturday"] == 1/1
    assert myb.posteriors["cancelled"][1]["spring"] == 1/1
    assert myb.posteriors["cancelled"][2]["high"] == 1/1
    assert myb.posteriors["cancelled"][3]["heavy"] == 1/1

def test_naive_bayes_classifier_predict():
    table = [
        [1,5,'yes'],
        [2,6,'yes'],
        [1,5,'no'],
        [1,5,'no'],
        [1,6,'yes'],
        [2,6,'no'],
        [1,5,'yes'],
        [1,6,'yes']
    ]
    myb = MyNaiveBayesClassifier()
    x_train = [row[:-1] for row in table]
    y_train = [row[-1] for row in table]
    myb.fit(x_train, y_train)
    myb.predict([[1,5]])

    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]
 
    x_train = [row[:-1] for row in iphone_table]
    y_train = [row[-1] for row in iphone_table]
    myb.fit(x_train, y_train)
    X_test = [
        [2,2,'fair'],
        [1,1,'excellent']
    ]
    y_test = myb.predict(X_test)
    assert y_test[0] == 'yes'
    assert y_test[1] == 'no'

    # Test 3:
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"], 
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"]
    ]
    x_train = [row[:-1] for row in train_table]
    y_train = [row[-1] for row in train_table]
    myb.fit(x_train, y_train)

    X_test = [
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "high", "heavy"],
        ["sunday", "summer", "normal", "slight"]
    ]
    y_test = myb.predict(X_test)
    assert y_test[0] == "very late"
    assert y_test[1] == "on time"
    assert y_test[2] == "on time"

def test_ZeroR_classifier():
    table = [
        [1,5,'yes'],
        [2,6,'yes'],
        [1,5,'no'],
        [1,5,'no'],
        [1,6,'yes'],
        [2,6,'no'],
        [1,5,'yes'],
        [1,6,'yes']
    ]
    myb = MyZeroRClassifier()
    x_train = [row[:-1] for row in table]
    y_train = [row[-1] for row in table]
    myb.fit(x_train, y_train)
    pred = myb.predict([[1,5]])
    assert pred[0] in y_train

def test_random_classifier():
    table = [
        [1,5,'yes'],
        [2,6,'yes'],
        [1,5,'no'],
        [1,5,'no'],
        [1,6,'yes'],
        [2,6,'no'],
        [1,5,'yes'],
        [1,6,'yes']
    ]
    myb = MyRandomClassifier()
    x_train = [row[:-1] for row in table]
    y_train = [row[-1] for row in table]
    myb.fit(x_train, y_train)
    prediction = myb.predict([[1,5]])
    assert prediction[0] in y_train
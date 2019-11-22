from sklearn.datasets import load_boston
import pandas as pd

boston = load_boston()

#Convert data to data frame using pandas
housing_data = pd.DataFrame(boston.data, columns = boston.feature_names)
#Price column
housing_data['MEDV'] = boston.target

#Function to find element influencing any other element
column = str()
def top_correlation(df, column):
    cor = df.corr().abs().unstack()
    cor = cor.sort_values(ascending=False)[column]
    cor = cor.nlargest(2)
    return cor

#User input
top_correlation(housing_data, 'MEDV')  #Obviously, MEDV is correlated with itself.
# we can see that housing price is highly correlated to LSTAT i.e.  Percentage of lower status of the population


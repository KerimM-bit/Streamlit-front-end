import streamlit as st
import pandas as pd
import base64

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import plotly.express as px
import plotly.graph_objects as go



header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
#model_training = st.beta_container()
linear_reg = st.beta_container()
interactive = st.beta_container()

st.markdown(
    """
    <style>
    .main {
        background-color: #F5F5F5;
        }
    </style>
    """,
    unsafe_allow_html=True)

background_color = '#F5F5F5'

@st.cache
def get_data(filename):
    data = pd.read_csv(filename)
    
    return data


with header:
    st.title('My first Streamlit project') 
    
with dataset:
    st.header('Movie dataset')
    st.text('You can access this dataset via my github or bla saytindan.')
    
    data = get_data("E:\Data analyst\Streamlit-data_viz\data\movie_metadata.csv")
    
    
    changed_ds = data.dropna(how='any', axis=0)
    changed_ds = changed_ds.drop(['color', 'facenumber_in_poster'], axis=1)
    changed_ds = changed_ds[changed_ds.loc[:]!=0].dropna()
    st.write(changed_ds)
    

    
    st.subheader('Data cleaning for Linear regression')
    
    new_ds = changed_ds[['director_facebook_likes', 'actor_1_facebook_likes',
                        'actor_2_facebook_likes', 'actor_3_facebook_likes', 'imdb_score']].copy()
    
    #new_ds.head(30)
    st.write(new_ds.head(5))
    
with interactive:
    st.title('A closer look into the data')
    
    st.header('IMDB links of most liked movies(Top 50)')
    sorted_changed_ds = changed_ds.sort_values(by='imdb_score', ascending=False,
                                               ignore_index=True)
    sorted_changed_ds = sorted_changed_ds[['movie_title', 'imdb_score', 'movie_imdb_link']].head(50)
    
    
    fig = go.Figure(data=go.Table(columnwidth=[2,1,3],
        header=dict(values=list(sorted_changed_ds[['movie_title','imdb_score',
        'movie_imdb_link']].columns),fill_color='#FD8E72', align='center'), 
        cells=dict(values=[sorted_changed_ds.movie_title, 
        sorted_changed_ds.imdb_score, sorted_changed_ds.movie_imdb_link],
                   fill_color='#E5ECf6',align='left')))
    
    fig.update_layout(margin=dict(l=5,r=5,b=10,t=10), 
                      paper_bgcolor=background_color)
    
    st.write(fig)
    
    #Sidebar- Country selection
    sorted_unique_country = sorted(changed_ds.country.unique())
    selected_country = st.sidebar.multiselect('Country', sorted_unique_country,sorted_unique_country)
    
    #Filtering data
    selected_ds = changed_ds[(changed_ds.country.isin(selected_country))]
    
    st.header('Display Movies bases on Countries')
    st.dataframe(selected_ds)
    
    figure = go.Figure()
    #Violin plot
    ratings = list(changed_ds.select_dtypes(['object', 'string']).columns)
    gross = list(changed_ds.select_dtypes(['float', 'int']).columns)
    
    #Scatter plot
    numeric_val = list(changed_ds.select_dtypes(['float', 'int']).columns)
    
    chart_plot = st.sidebar.selectbox(label='Select the chart type',
                                       options=['Violin plot', 'Scatterplots', 
                                                'Histogram'])
    if chart_plot == 'Violin plot':
        st.sidebar.subheader('Violin Plot Settings')
        try:
            x_values = st.sidebar.selectbox('X axis', options=ratings)
            y_values = st.sidebar.selectbox('Y axis', options=gross)
            plot = px.violin(data_frame=changed_ds, x=x_values, y=y_values, 
                             hover_data=changed_ds.columns)
            plot.update_layout(margin=dict(l=5,r=5,b=10,t=10), 
                      paper_bgcolor=background_color) 
            st.plotly_chart(plot)
        except Exception as e:
            print(e)
            
    elif chart_plot == 'Scatterplots':
        st.sidebar.subheader('Scatter Plot Settings')
        try:
            x_val = st.sidebar.multiselect(label='X axis', options=numeric_val)
            y_val = st.sidebar.selectbox(label='Y axis', options=numeric_val)
            plot_s = px.scatter(data_frame=changed_ds, x=x_val, y=y_val)
            st.plotly_chart(plot_s)
        except Exception as e:
            print(e)
            
    else: #histogram
        st.subheader('IMDB Scores of different movies' )
        imdb_dist = pd.DataFrame(changed_ds['imdb_score'].value_counts()).head(50)
        st.bar_chart(imdb_dist) 
        
with features:
    st.header('The features I created')
    
    st.markdown(" **first feature** I created `new_ds` dataframe in order to calculate Linear Regression without having to worry about categorical values.")
    st.markdown(" **second feature** Next step is to choose dependent and independent variables. In our new created dataframe `imdb_score` will be our dependent variable, other 3 columns will be our independent variables")

with linear_reg:
    st.subheader('MultiLinear Regression ')
    #sperate other attributes from the dependent attribute

    select, display = st.beta_columns(2)
    
    lr_X = new_ds[['director_facebook_likes', 'actor_1_facebook_likes',
                        'actor_2_facebook_likes', 'actor_3_facebook_likes']]
    #lr_X = lr_X.reshape(lr_X.shape[1:])
    lr_y = new_ds['imdb_score']
    
    #splitting the data
    X_train, X_test, y_train, y_test = train_test_split(lr_X, lr_y, test_size=0.25, random_state=42)
    
    #creating an object of LinearRegression class
    LR = LinearRegression()
    
    #fitting the training data
    LR.fit(X_train, y_train)
    
    y_prediction = LR.predict(X_test)
    
    select.text('In this model imdb_score chosen as dependent variable and other 3 independent variables')
    select.write(new_ds.columns)
    
    #predicting accuracy score 
    display.subheader('R square score is:')
    display.write(r2_score(y_test, y_prediction))
   
    display.subheader('Mean squared error is:' )
    display.write(mean_squared_error(y_test, y_prediction))
    
    display.subheader('Mean absolute error of the model is: ')
    display.write(mean_absolute_error(y_test,y_prediction))

 
download = st.button('Download CSV File')
if download:
    'Download started'
    #changed_ds
    csv = changed_ds.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    link = f'<a href="data:file/csv;base64,{b64}" download="myfile.csv">Download CSV File<a/>'
    st.markdown(link, unsafe_allow_html=True)    




    
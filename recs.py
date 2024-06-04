from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd

import plotly.graph_objects as go
import plotly.express as px

from surprise import Dataset                                                   
from surprise import Reader 
from surprise import KNNWithMeans

from surprise.model_selection import train_test_split

from collections import defaultdict

class DataProcess:

    def __init__(self, path):
        self.path = path 
    
    def load(self):
        df = pd.read_csv(self.path)

        #cleaned the image to be displayed in the web
        df['image'] = df['image'].apply(self.image_cleaned)

        #adding udemy url for which the title can be clicked and be directed to udemy website
        df['course_url'] = df['course_url'].apply(self.udemy_url)

        # renaming display_name to user_name
        df = df.rename(columns={'display_name' : 'user_name'})
        return df

    def unique(self, df):
    
        data = df.drop_duplicates(subset='title')
        return data[['course_id', 'title', 'headline', 'category', 
                     'subcategory', 'price', 'avg_rating', 'num_subscribers', 'course_url', 'image']]

    def image_cleaned(self, url):
        return url.replace("img-b", "img-c")

    def udemy_url(self, url):
        return 'https://www.udemy.com' + url
    

class DataExploratory:

    def __init__(self, df):
        self.df = df
        self.feature_dict = {
                                'id': 'Number of Courses Published', 
                                'num_subscribers': 'Total Subscribers', 
                                'num_comments': 'Total Comments', 
                                'num_reviews': 'Total Reviews',
                                'num_lectures': 'Total Lectures'
                            }
        self.inv_feature_dict = {value: key for key, value in self.feature_dict.items()}
    
    def most_popular(self):

        # Get top courses by categories according to highest num_subscribers
        most_popular_courses = self.df.sort_values(by='num_subscribers', ascending=False).groupby('category').head(1)
        most_popular = most_popular_courses[['course_id', 'title', 'category', 'subcategory', 'price', 'headline', 
                                            'avg_rating', 'num_subscribers', 'course_url', 'image']].reset_index(drop=True)
        most_popular.sort_values(by='num_subscribers', ascending=False)

        return most_popular

    def feature_count(self, column:str, n=10) -> pd.DataFrame:
        #counting a feature in interest
        counts = self.df.groupby(column).size().reset_index(name='count')
        return counts.sort_values(by='count', ascending=False)[:n]
    
    def feature_plot(self, column:str):
        feature = self.feature_count(column).sort_values(by='count', ascending=True)

        fig = px.bar(feature, y = column, x='count', text_auto=True)

        fig.update_layout(title_text=f'Top {len(feature)} courses by {column.capitalize()}',
                          yaxis_tickfont_size=14, yaxis=dict(title=''),
                          xaxis=dict(title='Count',titlefont_size=16,tickfont_size=14))
        return fig

    def subcategory_plot(self, category:str):

        df_cat = self.df[self.df['category']==category]
        feature = df_cat.groupby('subcategory').size().reset_index(name='count')
        feature = feature.sort_values(by='count', ascending=True)

        fig = px.bar(feature, y = 'subcategory', x = 'count', text_auto=True)

        fig.update_layout(title_text=f'Top courses in Category :  {category.capitalize()}',
                          yaxis_tickfont_size=14, yaxis=dict(title=''),
                          xaxis=dict(title='Count',titlefont_size=16,tickfont_size=14))
        
        return fig

    def instructor_plot(self, n=5):
        instructor_count = self.df.groupby('instructor_name')['num_subscribers'].sum().reset_index()
        top_instructor = instructor_count.sort_values(by='num_subscribers', ascending=False)[:n]

        fig = px.bar(top_instructor, x = 'instructor_name', y = 'num_subscribers', text_auto=True)

        fig.update_layout(title_text=f'Top {n} Performing Instructor based on Total Subscribers',
                          yaxis_tickfont_size=14, yaxis=dict(title=''),
                          xaxis=dict(title='',titlefont_size=16,tickfont_size=14))
        return fig

    def topic_plot(self, n=5):
        topic_subscribers = self.df.groupby('topic')['num_subscribers'].sum().reset_index()
        top_topic_subs = topic_subscribers.sort_values(by='num_subscribers', ascending=False)[:n]

        fig = px.bar(top_topic_subs, x = 'topic', y = 'num_subscribers', text_auto=True)

        fig.update_layout(title_text=f'Top {n} Topic Courses based on Total Subscribers',
                          yaxis_tickfont_size=14, yaxis=dict(title=''),
                          xaxis=dict(title='',titlefont_size=16,tickfont_size=14))
        return fig
    
    def yearly_feature(self):
        
        df_year = self.df
        df_year['published_time'] = pd.to_datetime(df_year['published_time'])
        df_year['published_year'] = df_year['published_time'].dt.year
        
        features = ['num_subscribers','num_reviews', 'num_comments', 'num_lectures']
        year = df_year.groupby('published_year')['id'].count().reset_index()
        for feature in features:
            feature_year = df_year.groupby('published_year')[feature].sum().reset_index()
            year = pd.concat([year, feature_year.iloc[: , 1:]], axis=1)
            
        return year

    def yearly_plot(self, feature):

        original_dict = self.feature_dict

        data_yearly = self.yearly_feature()
        
        year = data_yearly['published_year'].unique()
        colors = ['lightgray',] * len(year)

        highest_subs_index = data_yearly[feature].idxmax()
        colors[highest_subs_index ] = 'crimson'

        fig = go.Figure(data=[go.Bar(x = data_yearly['published_year'],
                                     y = data_yearly[feature], marker_color=colors)])
        
        fig.update_layout(title_text=f'{original_dict[feature]} from 2010 - 2022',
                          yaxis_tickfont_size=14, yaxis=dict(title=''),
                          xaxis=dict(title='',titlefont_size=16,tickfont_size=14))
        return fig
      

class ContentBased:
    
    def __init__(self, df):
        self.df = df 

    def compute_similarity(self, base:str = 'subcategory'):
        sub = self.df[base].str.split('|').explode()
    
        tfidf = TfidfVectorizer(ngram_range=(1,2))
        tfidf_matrix = tfidf.fit_transform(sub)
        
        similarity = cosine_similarity(tfidf_matrix)
        similarity_df = pd.DataFrame(similarity, index=self.df['title'], columns=self.df['title'])

        return similarity_df
    
    def get_top_n(self, title, sim=None, n=10):
        
        if sim == None:
            sim = self.compute_similarity()

        course_index = sim.index.get_loc(title)
        
        top_10 = sim.iloc[course_index].sort_values(ascending=False)[1:n+1]

        return top_10.index.tolist()


class CollaborativeFiltering:

    def __init__(self, df):
        self.df = df

    def data_selection(self):
        return self.df[['user_name', 'course_id', 'rate']]
    
    def data_splitting(self, test_size=0.3, random_state=42):

        course_data = self.data_selection()

        reader = Reader(rating_scale=(0,5))
        data = Dataset.load_from_df(course_data, reader)

        train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)  

        return train_data, test_data

    def model_calculation(self, k=5, similarity_measure='cosine', user_based=False):

        train_data, _ = self.data_splitting()

        # item-based approach
        sim_options = {'name' : similarity_measure,
                       'user_based' : user_based}

        # KNN with means; means of rating
        knn = KNNWithMeans(k=k, sim_options=sim_options, verbose=False)
        knn.fit(train_data) 

        return knn
    
    def predictions(self, model):

        _, test_data = self.data_splitting()

        pred = model.test(test_data)

        return pred
    
    def get_top_n(self, predictions, user_name, course_df, ratings_df, n = 10):

        '''Return the top N (default) movieId for a user,.i.e. userID and history for comparisom
        Args:
        Returns: 
    
        '''
        #Peart I.: Surprise docomuntation
        
        #1. First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        #2. Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key = lambda x: x[1], reverse = True)
            top_n[uid] = user_ratings[: n ]
        
        #Part II.: inspired by: https://beckernick.github.io/matrix-factorization-recommender/
        
        #3. Tells how many movies the user has already rated
        user_data = ratings_df[ratings_df.user_name == (user_name)]
        print('User {0} has already rated {1} courses.'.format(user_name, user_data.shape[0]))

        
        #4. Data Frame with predictions. 
        preds_df = pd.DataFrame([(id, pair[0],pair[1]) for id, row in top_n.items() for pair in row],
                                columns=["user_name" ,"course_id","rat_pred"])
        
        
        #5. Return pred_usr, i.e. top N recommended movies with (merged) titles and genres. 
        pred_usr = preds_df[preds_df["user_name"] == (user_name)].merge(course_df, how = 'left', 
                                                                        left_on = 'course_id', right_on = 'course_id')
                
        #6. Return hist_usr, i.e. top N historically rated movies with (merged) titles and genres for holistic evaluation
        hist_usr = ratings_df[ratings_df.user_name == (user_name) ].sort_values("rate", ascending = False).merge\
        (course_df, how = 'left', left_on = 'course_id', right_on = 'course_id')
        
        
        return hist_usr, pred_usr

    def retrieve_courses(self, df, model, course_id, k=10):
        
        id_dict = dict(zip(df['course_id'], df['title']))

        inner_id = model.trainset.to_inner_iid(course_id)
        neighbors_item = model.get_neighbors(inner_id, k=k)
        
        # Convert inner ids of the neighbors into names.
        neighbors_name = (
            model.trainset.to_raw_iid(iid) for iid in neighbors_item
        )
        
        course_neighbors = [id_dict[rid] for rid in neighbors_name]
        
        return course_neighbors
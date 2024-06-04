import streamlit as st
import pandas as pd
from recs import DataProcess, DataExploratory, ContentBased, CollaborativeFiltering
import hydralit_components as hc

# set layout wide
st.set_page_config(page_title='Udemy Recommendation System', layout="wide", page_icon=":computer:")

menu_data = [
        {'icon': "fas fa-tachometer-alt", 'label':"Dashboard", 'ttip':"Interactive Dashboard"},
        {'icon': "far fa-chart-bar", 'label':"Recommendation", 'ttip':"Course Recommendation and Prediction"},
        {'icon': "bi bi-hand-thumbs-up", 'label':"Summary", 'ttip':"Summary and Notes"},
        {'icon': "far fa-address-book", 'label':"Contact Me", 'ttip':"Contact Me"},
]

over_theme = {'txc_inactive': '#FFFFFF','menu_background':'#87CEFA','txc_active':'black','option_active':'white'}
menu_id = hc.nav_bar(menu_definition=menu_data, home_name='Home', override_theme=over_theme)



with st.sidebar:
    st.title('Udemy Course App')
    st.write('Made using **streamlit** by *Rasyid Sulaeman*')

    st.header('Background')
    st.info(
    '''
    Building a recommendation system for Udemy courses to enhance user experience by providing personalized course suggestions.
    '''
    )

    st.info(
    '''
    **Content-Based Filtering**: This method recommends courses based on the similarity between course content and the user's preferences. 
    '''
    )
    st.info(
    '''
    **Collaborative Filtering**: This approach recommends courses based on the behavior and preferences of similar users.
    '''
    )

path = 'dataset/udemy_sample_30.csv'

dp = DataProcess(path)
df = dp.load()
df_unique = dp.unique(df)

eda = DataExploratory(df)

if menu_id == 'Home':

    st.title('Udemy Course Recommendation System')
    st.divider()

    st.image('img/background.png', width=720, use_column_width='always')

    most_popular = eda.most_popular()

    with st.container():
        st.divider()
        st.subheader('Most Popular Course by Category')

        cols = st.columns(5)

        for i in range(len(cols)):
            cols[i].write(f"[{most_popular['title'][i]}](%s)" % most_popular['course_url'][i] )
            cols[i].image(most_popular['image'][i], 
                        caption='{} | {}'.format(most_popular['category'][i],
                                                most_popular['subcategory'][i]))
            caps = f"""
            *{most_popular['headline'][i]}* \n
            **Price** : $ {most_popular['price'][i]}\n
            **Rating** : {round(most_popular['avg_rating'][i],2)} :star: \n 
            **Total Subscribers** : {most_popular['num_subscribers'][i]:,} learners
            """
            cols[i].caption(caps)

    st.divider()

    st.subheader('Top Categories')

    category = eda.feature_count('category')[:4].reset_index(drop=True)

    with st.container():

        img_path = ['img/development.jpg', 'img/business.jpg', 'img/it_and_software.jpg', 'img/design.jpg']
        cols = st.columns(4)
        for i in range(len(cols)):
            cols[i].image(img_path[i], caption = f"{category['category'][i]} | {category['count'][i]:,} courses")
        
        st.divider()

if menu_id == 'Dashboard':

    st.title('Udemy Course Dashboard :computer:')
    st.divider()

    st.subheader('Best Udemy Courses by Category and Subcategory')

    features = ['category', 'subcategory']

    cols = st.columns(2)
    for i in range(len(cols)):
        fig = eda.feature_plot(features[i])
        cols[i].plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader('Top Courses in each Category')

    cat_unique = df['category'].unique()

    choose = st.selectbox('Select category you want to visualize!', options=cat_unique)
    subcat = eda.subcategory_plot(choose)
    st.plotly_chart(subcat, use_container_width=True)

    st.divider()

    cols = st.columns(2)

    instructor = eda.instructor_plot()

    cols[0].subheader('Top Performing Instructor')
    cols[0].plotly_chart(instructor, use_container_width=True)

    topic = eda.topic_plot()

    cols[1].subheader('Top Topic Courses')
    cols[1].plotly_chart(topic, use_container_width=True)

    st.divider()

    st.subheader('Yearly Feature Visualization')

    inv_feature_dict = eda.inv_feature_dict

    choose = st.selectbox('Select feature you want to visualize!', options=inv_feature_dict)

    yearly = eda.yearly_plot(inv_feature_dict[choose])

    st.plotly_chart(yearly, use_container_width=True)

if menu_id == 'Recommendation':
    st.title('Recommendation Engine')

    st.divider()

    tabs1, tabs2, tabs3 = st.tabs(['Content-Based Filtering', 'Collaborative Filtering - User Existing Prediction', 
                                   'Collaborative Filtering - Course Similarity Prediction'])

    df_index = df_unique.set_index('course_id')

    id_mapping = dict(zip(df_unique['title'], df_unique['course_id']))

    cb_recs = ContentBased(df_unique)

    title = tabs1.selectbox('Pick Udemy course you want to watch!', 
                            options=df_unique['title'].values, 
                            index=None,
                            placeholder="Select Your Preference's Course ...")
    
    with st.spinner('The model is calculated, it takes some time. Please wait ...'):

        if title == None:
            tabs1.warning("Please select the course you're interested in, and our system will provide the best recommendation for you.")

        else:

            title_id = id_mapping[title]

            tabs1.image(df_index['image'][title_id], caption='{} | {}'.format(df_index['category'][title_id],
                                                                            df_index['subcategory'][title_id]))
            tabs1.write(f'**Course Link** : [{title}](%s)' % df_index['course_url'][title_id])
            caps = f"""
                    *{df_index['headline'][title_id]}* \n
                    **Price** : $ {df_index['price'][title_id]}\n
                    **Rating** : {round(df_index['avg_rating'][title_id],2)} :star: \n 
                    **Total Subscribers** : {df_index['num_subscribers'][title_id]:,} learners
                    """
            tabs1.caption(caps)

            tabs1.divider()
            tabs1.info(f'Since you pick course about {title}, here are top 10 recommendation courses for you!')

            top_n = cb_recs.get_top_n(title)
            id_list = [id_mapping[title] for title in top_n]

            cols = tabs1.columns(5)

            for i in range(len(cols)):

                cols[i].write(f'[{top_n[i]}](%s)' % df_index['course_url'][id_list[i]])
                cols[i].image(df_index['image'][id_list[i]], caption='{} | {}'.format(df_index['category'][id_list[i]],
                                                                                    df_index['subcategory'][id_list[i]]))
                
                caps = f"""
                        *{df_index['headline'][id_list[i]]}* \n
                        **Price** : $ {df_index['price'][id_list[i]]}\n
                        **Rating** : {round(df_index['avg_rating'][id_list[i]],2)} :star: \n 
                        **Total Subscribers** : {df_index['num_subscribers'][id_list[i]]:,} learners
                        """
                cols[i].caption(caps)

            tabs1.divider()
            cols = tabs1.columns(5)

            for i in range(len(cols)):

                cols[i].write(f'[{top_n[len(cols) + i]}](%s)' % df_index['course_url'][id_list[len(cols) + i]])
                cols[i].image(df_index['image'][id_list[len(cols) + i]], 
                            caption='{} | {}'.format(df_index['category'][id_list[len(cols) + i]],
                                                    df_index['subcategory'][id_list[len(cols) + i]]))
                
                caps = f"""
                        *{df_index['headline'][id_list[len(cols) + i]]}* \n
                        **Price** : $ {df_index['price'][id_list[len(cols) + i]]}\n
                        **Rating** : {round(df_index['avg_rating'][id_list[len(cols) + i]],2)} :star: \n 
                        **Total Subscribers** : {df_index['num_subscribers'][id_list[len(cols) + i]]:,} learners
                        """
                cols[i].caption(caps)
        
            tabs1.divider()

        tabs2.header('User Existing Recommendation')
        
        tabs2.info(
        '''In this section, we are going to predict/ recommend the course for 
        existing user from their past preference and history watching. 
        ''')
        tabs2.write(
        ''' Select the **user name** to see how the algoritm works, 
            that eventually provide the top 10 course recommendation for them (*the user*)
        ''')
        
        cf_recs = CollaborativeFiltering(df)

        course_data = cf_recs.data_selection()

        model = cf_recs.model_calculation()
        pred = cf_recs.predictions(model)

        course_df = df_unique.copy()

        user_name = 'Yusuf'
        hist_user, pred_user = cf_recs.get_top_n(pred, user_name, course_df, course_data)

        tabs2.info(f'User {user_name} has already rated {hist_user.shape[0]} courses')
        tabs2.dataframe(hist_user)

        tabs2.dataframe(pred_user)
    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
if menu_id == 'Summary':
    pass

if menu_id == 'Contact Me':
    pass
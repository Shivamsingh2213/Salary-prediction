import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title("Salary Prediction App")

try:
    # Load the datasets
    survey_data = pd.read_csv('survey_results_public.csv')
    st.write("Data Loaded Successfully")
except Exception as e:
    st.write(f"Error loading data: {e}")

try:
    # Select relevant columns
    survey_data = survey_data[['Country', 'EdLevel', 'YearsCodePro', 'Employment', 'ConvertedCompYearly']]
    survey_data = survey_data.dropna(subset=['ConvertedCompYearly'])
    st.write("Relevant Columns Selected")

    # Handle the 'YearsCodePro' column
    survey_data['YearsCodePro'] = survey_data['YearsCodePro'].replace('Less than 1 year', 0)
    survey_data['YearsCodePro'] = survey_data['YearsCodePro'].replace('More than 50 years', 50)
    survey_data['YearsCodePro'] = survey_data['YearsCodePro'].astype(float)

    # Encode categorical variables
    survey_data_encoded = pd.get_dummies(survey_data, columns=['Country', 'EdLevel', 'Employment'], drop_first=True)
    st.write("Categorical Variables Encoded")
except Exception as e:
    st.write(f"Error processing data: {e}")

try:
    # Load the saved model and imputer
    model = joblib.load('salary_prediction_model.pkl')
    imputer = joblib.load('imputer.pkl')
    st.write("Model and Imputer Loaded Successfully")
except Exception as e:
    st.write(f"Error loading model or imputer: {e}")

try:
    # Define the input fields
    country_options = survey_data['Country'].unique()
    education_options = survey_data['EdLevel'].unique()
    employment_options = survey_data['Employment'].unique()

    country = st.selectbox('Country', country_options)
    education = st.selectbox('Education Level', education_options)
    employment = st.selectbox('Employment Type', employment_options)
    years_experience = st.slider('Years of Experience', 0, 50, 5)

    # Predict button
    if st.button('Predict Salary'):
        # Preprocess inputs
        input_data = pd.DataFrame([[country, education, employment, years_experience]],
                                  columns=['Country', 'EdLevel', 'Employment', 'YearsCodePro'])
        input_data['YearsCodePro'] = input_data['YearsCodePro'].astype(float)
        input_data = pd.get_dummies(input_data, columns=['Country', 'EdLevel', 'Employment'], drop_first=True)
        
        # Align input data with training data columns
        input_data = input_data.reindex(columns=survey_data_encoded.columns.drop('ConvertedCompYearly'), fill_value=0)
        
        # Impute any missing values in the input_data
        input_data = imputer.transform(input_data)
        
        # Predict salary
        prediction = model.predict(input_data)
        
        # Correct negative predictions
        if prediction[0] < 0:
            prediction[0] = abs(prediction[0])
        
        st.write(f'Predicted Salary: ${prediction[0]:,.2f}')

        # Visualizations
        # st.write("### Salary Distribution")
        # fig, ax = plt.subplots()
        # sns.histplot(survey_data['ConvertedCompYearly'], kde=True, ax=ax)
        # ax.set_xlabel('Salary')
        # ax.set_ylabel('Frequency')
        # st.pyplot(fig)

        # st.write("### Average Salary by Country")
        # country_avg_salary = survey_data.groupby('Country')['ConvertedCompYearly'].mean().sort_values()
        # fig, ax = plt.subplots(figsize=(10, 8))
        # country_avg_salary.plot(kind='barh', ax=ax)
        # ax.set_xlabel('Average Salary')
        # ax.set_ylabel('Country')
        # st.pyplot(fig)

        # st.write("### Salary by Education Level")
        # fig, ax = plt.subplots(figsize=(10, 8))
        # sns.boxplot(x='ConvertedCompYearly', y='EdLevel', data=survey_data, ax=ax)
        # ax.set_xlabel('Salary')
        # ax.set_ylabel('Education Level')
        # st.pyplot(fig)

        # st.write("### Years of Experience vs. Salary")
        # fig = px.scatter(survey_data, x='YearsCodePro', y='ConvertedCompYearly', trendline="ols", 
        #                  labels={'YearsCodePro': 'Years of Experience', 'ConvertedCompYearly': 'Salary'})
        # st.plotly_chart(fig)
        st.write("### Detailed Salary Distribution")
        fig, ax = plt.subplots()
        sns.histplot(survey_data['ConvertedCompYearly'], kde=True, ax=ax, bins=30)
        mean_salary = survey_data['ConvertedCompYearly'].mean()
        median_salary = survey_data['ConvertedCompYearly'].median()
        ax.axvline(mean_salary, color='r', linestyle='--', label=f'Mean Salary: ${mean_salary:,.2f}')
        ax.axvline(median_salary, color='g', linestyle='--', label=f'Median Salary: ${median_salary:,.2f}')
        ax.set_xlabel('Salary')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig)
        
        st.write("### Interactive Average Salary by Country")
        country_avg_salary = survey_data.groupby('Country')['ConvertedCompYearly'].mean().reset_index()
        fig = px.bar(country_avg_salary, x='Country', y='ConvertedCompYearly', title='Average Salary by Country',
                    labels={'ConvertedCompYearly': 'Average Salary'}, text='ConvertedCompYearly')
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(xaxis_title='Country', yaxis_title='Average Salary')
        st.plotly_chart(fig)
        
        st.write("### Detailed Salary by Education Level")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(x='ConvertedCompYearly', y='EdLevel', data=survey_data, ax=ax)
        ax.set_xlabel('Salary')
        ax.set_ylabel('Education Level')
        ax.set_title('Salary Distribution by Education Level')
        # Add mean markers
        mean_salaries = survey_data.groupby('EdLevel')['ConvertedCompYearly'].mean()
        for level, mean_salary in mean_salaries.items():
            ax.text(mean_salary, level, f'{mean_salary:,.2f}', va='center', ha='right', color='black')
        st.pyplot(fig)
        
        
        st.write("### Detailed Years of Experience vs. Salary")
        fig = px.scatter(survey_data, x='YearsCodePro', y='ConvertedCompYearly', trendline="ols", 
                        labels={'YearsCodePro': 'Years of Experience', 'ConvertedCompYearly': 'Salary'},
                        title='Years of Experience vs. Salary')
        fig.update_traces(marker=dict(size=7, opacity=0.6, line=dict(width=1, color='DarkSlateGrey')),
                        selector=dict(mode='markers+lines'))
        st.plotly_chart(fig)
        
        st.write("### Pairwise Relationships")
        fig = sns.pairplot(survey_data[['YearsCodePro', 'ConvertedCompYearly']], diag_kind='kde')
        st.pyplot(fig)
        
        st.write("### Correlation Heatmap")
        corr_matrix = survey_data[['YearsCodePro', 'ConvertedCompYearly']].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)


        
        
except Exception as e:
    st.write(f"Error making prediction or creating visualizations: {e}")

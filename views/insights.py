import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def render_insights():
    st.markdown("<h2 class='sub-header'>📊 Model Performance Insights</h2>", unsafe_allow_html=True)
    st.write("Compare the performance of different machine learning models trained on our dataset.")
    
    # Mock data for demonstration as requested
    models = ['Logistic Regression', 'Naive Bayes', 'Random Forest']
    accuracy = [0.88, 0.82, 0.94]
    precision = [0.85, 0.79, 0.93]
    recall = [0.80, 0.88, 0.91]
    
    # Accuracy Chart
    fig_acc = go.Figure(data=[
        go.Bar(name='Accuracy', x=models, y=accuracy, marker_color='#99f2c8')
    ])
    fig_acc.update_layout(
        title='Model Accuracy Comparison',
        xaxis_title='Algorithms',
        yaxis_title='Accuracy Score',
        template='plotly_white',
        yaxis=dict(range=[0, 1])
    )
    st.plotly_chart(fig_acc, use_container_width=True)
    
    # Radar Chart
    st.markdown("### Detailed Metrics Breakdown")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("The Random Forest algorithm is currently deployed because it provides the best balance of Precision and Recall, minimizing false positives.")
        
    with col2:
         # Radar chart for Random forest as an example
        fig_radar = go.Figure(data=go.Scatterpolar(
          r=[accuracy[2], precision[2], recall[2], 0.95],
          theta=['Accuracy','Precision','Recall', 'F1 Score'],
          fill='toself',
          marker_color='#1f4037'
        ))
        fig_radar.update_layout(
          polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
          ),
          showlegend=False,
          title="Random Forest Performance"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

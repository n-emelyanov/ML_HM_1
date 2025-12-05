import streamlit as st
import pickle
import pandas as pd
import numpy as np
from io import StringIO
import base64
import sklearn
import matplotlib.pyplot as plt

FEATURES = ['year',
 'km_driven',
 'mileage',
 'engine',
 'max_power',
 'torque',
 'seats',
 'max_torque_rpm',
 'fuel',
 'seller_type',
 'transmission',
 'owner']


# Загрузка модели
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Основное приложение
def main():
    st.title('🤖 ML Model Demo')
    st.write("Простое приложение для предсказаний с помощью ML модели")
    
    # Загружаем модель
    model = load_model()
    
    # Сайдбар для навигации
    page = st.sidebar.selectbox(
        "Выберите страницу",
        ["Главная", "Загрузка данных", "Предсказания", "Веса"]
    )
    
    if page == "Главная":
        show_home()
    elif page == "Загрузка данных":
        show_data_upload()
    elif page == "Предсказания":
        make_predictions(model)
    elif page == "Веса":
        weights(model)

def show_home():
    st.header("Добро пожаловать!")
    st.write("""
    ### Как использовать:
    1. Загрузите CSV файл с данными (Пример сохранен в репозитории - `data/sample.csv`)
    2. Перейдите на страницу предсказаний
    3. Нажмите - Предсказать
    """)

def show_data_upload():
    st.header("Загрузка данных")
    
    # Загрузка файла
    uploaded_file = st.file_uploader("Выберите CSV файл", type=['csv'])
    
    if uploaded_file is not None:
        # Чтение данных
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")
            
            # Показать первые строки
            st.subheader("Предпросмотр данных")
            st.dataframe(df.head())
            
            # Сохраняем данные в session state
            st.session_state['data'] = df
            st.session_state['features'] = df.columns.tolist()
            
        except Exception as e:
            st.error(f"Ошибка при чтении файла: {e}")

def make_predictions(model):
    st.header("Предсказания")
    
    if 'data' not in st.session_state:
        st.warning("Загрузите данные")
        return
    
    df = st.session_state['data']
    
    if st.button("Предсказать"):
        df['prediction'] = model.predict(df[FEATURES])
        
        st.dataframe(df)
            


def weights(model):
    st.header("Веса модели")

    # Извлекаем модель
    ridge_model = model.named_steps['model']
    coef = ridge_model.coef_

    # Получаем имена фич после всех преобразований
    preprocessor = model.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out()

    # Создаем DataFrame
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coef': coef,
        'abs_coef': np.abs(coef)
    })

    # Сортируем по важности
    coef_df = coef_df.sort_values('abs_coef', ascending=False)

    # Сортируем для графика
    coef_df_sorted = coef_df.sort_values(by='coef', ascending=True)
    
    # Создаем и отображаем график
    st.subheader("График важности признаков")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(coef_df_sorted['feature'], coef_df_sorted['coef'])
    ax.set_xlabel('Абсолютное значение коэффициента')
    ax.set_title('Важность признаков (абсолютные значения коэффициентов)')
    ax.grid(axis='x', alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Выводим таблицу
    st.subheader("Таблица коэффициентов")
    st.dataframe(coef_df_sorted)

if __name__ == "__main__":
    main()
import streamlit as st
from datetime import datetime
with st.sidebar:
    with st.form("form_tickers"):
        tickers_input = st.text_input('Adicione os tickers separados por vírgula')
        submitted = st.form_submit_button("Confirmar")
        
        if submitted and tickers_input:
            tickers_lista = [ticker.strip().upper() for ticker in tickers_input.split(',')]
            st.write('Tickers selecionados:', tickers_lista)
        
    with st.form("form_data"):
        data_inicial = st.date_input("Data inicial",
                                     min_value=datetime(1988, 1, 1),
                                     max_value=datetime.now().date()
        )
        data_final = st.date_input("Data final",
                                   min_value=datetime(1988, 1, 1),
                                   max_value=datetime.now().date()
        )
        submitted = st.form_submit_button("Confirmar")
        
        if submitted and data_inicial and data_final:
            st.write('Data inicial:', data_inicial)
            st.write('Data final:', data_final)
            
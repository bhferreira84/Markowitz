import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from Markowitz import obter_dados_ativos, calcular_retornos, calcular_variancia, calcular_retorno_portfolio, calcular_sharpe_ratio, otimizar_portfolio, otimizar_sharpe_ratio, fronteira_eficiente

# Variáveis de estado para armazenar os dados
if 'retorno_esperado_benchmark' not in st.session_state:
    st.session_state.retorno_esperado_benchmark = None
if 'variancia_benchmark' not in st.session_state:
    st.session_state.variancia_benchmark = None
if 'retorno_esperado_ativos' not in st.session_state:
    st.session_state.retorno_esperado_ativos = None
if 'matriz_cov' not in st.session_state:
    st.session_state.matriz_cov = None
if 'retornos_alvo' not in st.session_state:
    st.session_state.retornos_alvo = None
if 'volatilidades' not in st.session_state:
    st.session_state.volatilidades = None
if 'pesos' not in st.session_state:
    st.session_state.pesos = None
if 'pesos_max_sharpe' not in st.session_state:
    st.session_state.pesos_max_sharpe = None

# Criando duas colunas: uma para o sidebar e outra para os dados
sidebar = st.sidebar
dados_container = st.container()

with sidebar:
    with st.form("inputs"):
        tickers_input = st.text_input('Adicione os tickers separados por vírgula')
        benchmark_input = st.text_input('Adicione o benchmark')
        taxa_livre_risco = st.number_input('Adicione a taxa livre de risco', value=0.1)
        data_inicial = st.date_input("Data inicial",
                                     min_value=datetime(1988, 1, 1),
                                     max_value=datetime.now().date()
        )
        data_final = st.date_input("Data final",
                                   min_value=datetime(1988, 1, 1),
                                   max_value=datetime.now().date()
        )

        submitted = st.form_submit_button("Confirmar")
        
    if submitted and tickers_input and benchmark_input and data_inicial and data_final:
        try:
            #Captura os precos dos ativos e do benchmark
            dados_ativos, dados_benchmark = obter_dados_ativos(
                tickers_input,
                benchmark_input,
                start=data_inicial,
                end=data_final
            )
            
            #Calcula os retornos esperados e a covariância
            retorno_esperado_ativos, matriz_cov, retorno_esperado_benchmark, variancia_benchmark = calcular_retornos((dados_ativos, dados_benchmark))
            
            # Calcular fronteira eficiente
            retornos_alvo, volatilidades, pesos = fronteira_eficiente(retorno_esperado_ativos, matriz_cov)

            # Otimizar portfólio de máxima Sharpe
            pesos_max_sharpe = otimizar_sharpe_ratio(retorno_esperado_ativos, matriz_cov, taxa_livre_risco)

            # Armazenando os dados no session_state apenas se não houver erros
            st.session_state.retorno_esperado_benchmark = retorno_esperado_benchmark
            st.session_state.variancia_benchmark = variancia_benchmark
            st.session_state.matriz_cov = matriz_cov
            st.session_state.retorno_esperado_ativos = retorno_esperado_ativos
            st.session_state.retornos_alvo = retornos_alvo
            st.session_state.volatilidades = volatilidades
            st.session_state.pesos = pesos
            st.session_state.pesos_max_sharpe = pesos_max_sharpe
            st.success('Dados obtidos com sucesso!')

        except Exception as e:
            st.error(f'Erro ao obter dados: {str(e)}')

        # Mostrando os dados no container principal
        with dados_container:
            if st.session_state.retornos_alvo is not None:
                st.subheader('FRONTEIRA EFICIENTE')
                
                # Encontrando o portfólio de mínima variância
                volatilidades_series = pd.Series(st.session_state.volatilidades)
                idx_min_vol = volatilidades_series.argmin()
                retorno_min_vol = st.session_state.retornos_alvo[idx_min_vol]
                
                fig = go.Figure()

                fig.add_trace(go.Scatter(x=st.session_state.volatilidades, y=np.array(st.session_state.retornos_alvo)*100, mode='markers', name='Portfólio'))

                fig.add_trace(go.Scatter(
                    x=[st.session_state.volatilidades[idx_min_vol]], 
                    y=[st.session_state.retornos_alvo[idx_min_vol]*100], 
                    mode='markers', 
                    name='Portfólio de Mínima Variância', 
                    marker=dict(color='green', size=10)
                ))

                retorno_max_sharpe = calcular_retorno_portfolio(st.session_state.pesos_max_sharpe, st.session_state.retorno_esperado_ativos)
                volatilidade_max_sharpe = np.sqrt(calcular_variancia(st.session_state.pesos_max_sharpe, st.session_state.matriz_cov))
                
                fig.add_trace(go.Scatter(
                    x=[volatilidade_max_sharpe], 
                    y=[retorno_max_sharpe*100], 
                    mode='markers', 
                    name='Portfólio de Máxima Sharpe', 
                    marker=dict(color='red', size=10)
                ))

                volatilidade_benchmark = np.sqrt(st.session_state.variancia_benchmark[0])
                fig.add_trace(go.Scatter(
                    x=[volatilidade_benchmark], 
                    y=[st.session_state.retorno_esperado_benchmark[0]*100], 
                    mode='markers', 
                    name='Benchmark', 
                    marker=dict(color='yellow', size=10)
                ))

                fig.update_layout(
                    title='Fronteira Eficiente', 
                    xaxis_title='Volatilidade', 
                    yaxis_title='Retorno Esperado (%)'
                )
                st.plotly_chart(fig)
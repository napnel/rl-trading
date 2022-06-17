import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def body():
    df = pd.read_pickle("./data/BTCUSDT/candlesticks/15T.pkl")
    df = df[-100:]
    st.dataframe(df)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df.Open,
            high=df.High,
            low=df.Low,
            close=df.Close,
        )
    )
    st.plotly_chart(fig)
    # st.line_chart(df[["Open", "Close"]])
    df["Sell volume"] *= -1
    st.bar_chart(df[["Buy volume", "Sell volume"]])


if __name__ == "__main__":
    st.header("Crypto App")
    body()

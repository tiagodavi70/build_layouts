import altair as alt
from vega_datasets import data
import streamlit as st

source = data.iris()

c = alt.Chart(source).transform_window(
    index='count()'
).transform_fold(
    fold=['petalLength', 'petalWidth', 'sepalLength', 'sepalWidth'],
    as_=["k","value"]
).mark_line().encode(
    x='k:N',
    y='value:Q',
    color='species:N',
    detail='index:N',
    opacity=alt.value(0.5)
).properties(width=500)

st.write(c)

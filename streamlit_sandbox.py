import streamlit as st
import time
import numpy as np
import time

x = st.slider("x")
st.write(x, 'squared is', x * x)

add_selectbox = st.sidebar.selectbox(
    "selectbox", 
    ["a", "b"] # options
)

add_slider = st.sidebar.slider(
    "slider",
    0, 100, [0,20]
)

st.sidebar.markdown("# Header\n text")

@st.cache
def cached_function(num):
    time.sleep(3)
    return num

st.write("cached call: " , cached_function(5) )

progress_bar = st.progress(0)
status_text = st.empty()
chart = st.line_chart(np.random.randn(10, 2))

for i in range(100):
    # Update progress bar.
    progress_bar.progress(i)

    new_rows = np.random.randn(10, 2)

    # Update status text.
    status_text.text(
        'The latest random number is: %s' % new_rows[-1, 1])

    # Append data to the chart.
    chart.add_rows(new_rows)

    # Pretend we're doing some computation that takes time.
    time.sleep(0.1)

status_text.text('Done!')
st.balloons()
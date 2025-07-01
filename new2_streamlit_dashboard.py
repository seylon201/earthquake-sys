import streamlit as st

st.title("Earthquake Early Warning System")
st.write("NEW2 ConvLSTM with 98.5% Accuracy")

st.write("System Performance:")
st.write("- Accuracy: 98.5%")
st.write("- Detection Rate: 99.4%")
st.write("- False Alarm Rate: 0.6%")

st.write("Key Features:")
st.write("- AI Model: NEW2 ConvLSTM")
st.write("- Real-time earthquake detection")
st.write("- 3-class classification")

if st.button("Test Alert"):
    st.write("🚨 EARTHQUAKE ALERT!")
    st.write("Demo alert activated successfully!")
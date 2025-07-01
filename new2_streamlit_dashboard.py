import streamlit as st

st.set_page_config(page_title="Earthquake Detection System", page_icon="🌍")

st.title("🌍 Earthquake Early Warning System")
st.markdown("### NEW2 ConvLSTM with 98.5% Accuracy")

st.header("System Overview")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Accuracy", "98.5%")
with col2:
    st.metric("Detection Rate", "99.4%") 
with col3:
    st.metric("False Alarm Rate", "0.6%")

st.header("Key Features")
st.write("- AI Model: NEW2 ConvLSTM with 98.5% accuracy")
st.write("- Real-time monitoring: Earthquake vs Industrial vs Living vibrations")
st.write("- Web dashboard: Streamlit-based interface")
st.write("- 3-axis sensor visualization")

st.header("Performance Metrics")
data = {
    "Class": ["Earthquake", "Industrial Vibration", "Living Vibration"],
    "Accuracy": ["99.4%", "99.1%", "96.9%"],
    "Samples": ["540/543", "538/543", "526/543"]
}
st.table(data)

st.success("This system achieves 98.5% accuracy in earthquake detection!")

if st.button("Demo Earthquake Alert"):
    st.error("🚨 EARTHQUAKE ALERT!")
    st.markdown("**Magnitude 4.2 earthquake detected. Take cover immediately!**")
    st.balloons()
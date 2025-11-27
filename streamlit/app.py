import streamlit as st
from final import chatbot_response, generate_report, predict_conditions

st.set_page_config(page_title="Mental Health Chatbot", layout="wide")
st.title("💬 Mental Health Chatbot")

# Initialize conversation log in session state
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

def submit_input():
    """Function to handle user input submission."""
    user_input = st.session_state.input_text
    if user_input:
        try:
            emotion, response = chatbot_response(user_input)
        except Exception as e:
            st.error(f"Error in chatbot response: {e}")
            emotion, response = "error", "Oops, something went wrong!"

        st.session_state.chat_log.append({
            "text": user_input,
            "emotion": emotion,
            "response": response
        })

        # Clear input safely
        st.session_state.input_text = ""

# Sidebar for ending the conversation
if st.sidebar.button("End Conversation"):
    if st.session_state.chat_log:
        st.subheader("📊 Conversation Report")
        try:
            report, fig = generate_report(st.session_state.chat_log)
            st.text(report)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating report: {e}")

        try:
            conditions = predict_conditions(st.session_state.chat_log)
            st.subheader("⚠️ Possible Mental Health Conditions")
            for c in conditions:
                st.write(f"- {c}")
        except Exception as e:
            st.error(f"Error predicting conditions: {e}")

        st.session_state.chat_log = []
    else:
        st.info("No conversation to report yet.")

# Chat input with on_change callback
st.text_input(
    "You:", key="input_text", on_change=submit_input
)

# Display conversation
for entry in st.session_state.chat_log:
    st.markdown(f"**You:** {entry['text']}")
    st.markdown(f"**Chatbot [{entry['emotion'].upper()}]:** {entry['response']}")

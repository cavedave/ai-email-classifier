# Main Streamlit application
import streamlit as st

def main():
    st.set_page_config(page_title="Intro", page_icon="📬")  # <- This is the key line

    st.title("Email Classification System")
    st.write("Welcome to the Email Classification System!")
    st.write("Use the sidebar to navigate between different pages.")
    st.write("You can add Trianing examples.")
    st.write("classify new emails.")
    st.write("or look at what we have classified already.")

if __name__ == "__main__":
    main() 
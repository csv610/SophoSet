import streamlit as st
import pandas as pd

def main():
    st.title("CSV File Viewer")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Display the dataframe
        st.subheader("Data Preview")
        st.dataframe(df)

        # Display basic statistics
        st.subheader("Data Statistics")
        st.write(df.describe())

if __name__ == "__main__":
    main()

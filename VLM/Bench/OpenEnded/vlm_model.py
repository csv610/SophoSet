from llava_model import LlavaModel
import streamlit as st


@st.cache_resource
def load_vlm_model(model_name):
    vlm = LlavaModel(model_name)
    return vlm


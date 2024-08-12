import os
import time
import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults


def validate_and_set_keys(langchain_api_key, tavily_api_key, langchain_disabled, tavily_disabled):
    if langchain_disabled:
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "Multi-agent Collaboration"
        st.session_state.langchain_api_key_set = True
        st.session_state.langchain_api_key = langchain_api_key

        progress_text = "正在配置哦，请稍等..."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()

        st.info("LANGCHAIN_API_KEY 配置成功", icon='🌟')
        st.info("LANGSMITH 配置成功", icon='🌟')
        time.sleep(1)

    if tavily_disabled:
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        try:
            st.session_state.tavily_tool = TavilySearchResults(max_results=5)
            st.session_state.tavily_api_key_set = True
            st.session_state.tavily_api_key = tavily_api_key

            progress_text = "正在配置哦，请稍等..."
            my_bar = st.progress(0, text=progress_text)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()

            st.info("TAVILY_API_KEY 工具配置成功", icon='🌟')
            st.info(":red[注意：]请确保你的 TAVILY_API_KEY 有效，TAVILY_TOOL工具可以配置，但似乎只会在运行中报错，",
                    icon='🚫')
            time.sleep(1)

        except Exception as e:
            del os.environ["TAVILY_API_KEY"]
            st.session_state.tavily_api_key_set = False
            st.toast("TAVILY_API_KEY 配置失败: 无效的API密钥", icon='🚫')
            st.stop()

    st.session_state.show_rerun_button = True




@st.experimental_dialog("设置 API 密钥")
def set_api_keys():
    st.session_state.dialog_active = True
    st.session_state.langchain_toast_shown = False
    st.session_state.tavily_toast_shown = False
    st.session_state.show_rerun_button = False
    st.write("请输入你的 API 密钥:")
    langchain_disabled = st.checkbox("修改 LANGCHAIN_API_KEY", value=False)
    tavily_disabled = st.checkbox("修改 TAVILY_API_KEY", value=False)

    langchain_api_key = st.text_input("请输入你的 LANGCHAIN_API_KEY:", value=os.environ.get("LANGCHAIN_API_KEY", ""),
                                      disabled=not langchain_disabled)
    tavily_api_key = st.text_input("请输入你的 TAVILY_API_KEY:", value=os.environ.get("TAVILY_API_KEY", ""),
                                   disabled=not tavily_disabled)

    apply_button = st.button("应用", disabled=st.session_state.show_rerun_button, key='1')
    if apply_button:
        if langchain_disabled or tavily_disabled:
            validate_and_set_keys(langchain_api_key, tavily_api_key, langchain_disabled, tavily_disabled)

            time.sleep(1)
            st.session_state.show_dialog = False
            st.session_state.dialog_active = False
            st.rerun()


# 删除环境变量
def delete_environment_variable(key):
    if key in os.environ:
        del os.environ[key]
        if key == "LANGCHAIN_API_KEY":
            st.session_state.langchain_api_key_set = False
            st.session_state.langchain_toast_shown = False
            st.session_state.langchain_api_key = ""
        elif key == "TAVILY_API_KEY":
            st.session_state.tavily_api_key_set = False
            st.session_state.tavily_toast_shown = False
            st.session_state.tavily_api_key = ""
            st.session_state.tavily_tool = None
        st.rerun()


def set_environment_variable(key, value):
    os.environ[key] = value
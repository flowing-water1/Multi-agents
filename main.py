import os
import streamlit as st

from langchain_community.callbacks.manager import get_openai_callback
from graph import create_llm, create_agent, agent_node, StateGraph, END, HumanMessage, python_repl, AgentState, router
import functools

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage, AIMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from display import (
    extract_code_from_events,
    modify_code_for_streamlit,
    display_researcher_data,
    display_call_tool_data,
    display_chart_generator_data
)
from streamlit_check import (

    validate_and_set_keys,
    set_api_keys,
    delete_environment_variable,
    set_environment_variable
)
import sys
import io

# 智谱的
# b27f36511b74905f6adfd4f1e7cf1f72.r90Sh9TOsQ3JcQaf
# https://open.bigmodel.cn/api/paas/v4/

# using Tavily to research the question:'who is the richest person in this world now.', when you find it, finsish.ATTENTION:DO NOT USE 'chart_generator'
#Fetch the UK's GDP over the past 3 years, then draw a line graph of it. Once you code it up, finish.


# sk-9oYJRePIyAbz7wNj955dBbC98f0c44F8B91bF7779d38B131
# https://gtapi.xiaoerchaoren.com:8932
# https://gtapi.xiaoerchaoren.com:8932/v1
# https://gtapi.xiaoerchaoren.com:8932/v1/chat/completions

# https://tavily.com/
# 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
# https://smith.langchain.com/settings
# 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


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#

with st.sidebar:
    tab1, tab2 = st.tabs(["OPENAI", "智谱"])
    with tab1:
        openai_api_key = st.text_input("OpenAI API Key:", type="password")
        openai_api_base = st.text_input("OpenAI API Base:")
        # openai_port = st.text_input("VPN的端口：")
        st.markdown("[获取OpenAI API key](https://platform.openai.com/account/api-keys)")
        st.markdown("[OpenAI API文档](https://platform.openai.com/docs/api-reference/introduction)")
        st.markdown("要用直连原版的API话，要开VPN，端口设置为7890。")
        st.markdown("用中转的不用开VPN，已测试过中转的跟直连的效果一样。")

    with tab2:
        zhipu_api_key = st.text_input("智谱AI的API Key:", type="password")
        zhipu_api_base = st.text_input("智谱AI的API Base:")
        st.markdown("[获取智谱AI的API key](https://www.zhipuai.cn/)")
        st.markdown("国产的LLM模型基本上无法完成任务，但是可能可以通过修改prompt完成任务")

st.title("Mutil-Agents Collaboration\n(For Research and Chart purpose)")
st.divider()
with st.expander("**:red[使用前必看信息]**"):
    st.write('''**请注意**，Mutil-Agents是delicate的，因为LLM模型内部是:blue-background[黑箱].''')

    st.write(
        ''':blue-background[所以可能会有意想不到的bug。就算是同样的API，prompt和模型，可能结果也会有差距，这是不可避免的。]''')

    st.write('''***:red[模型推荐gpt-4/gpt-4o/gpt-4-1106-preview/claude-3]***.''')

    st.write('''实测这些模型可以完成任务（仍然会受prompt影响可能报错）''')

    st.write('''**:red[tavily是负责搜索的，langchain是监控流程和提供支持的，一定要填]**''')

    st.write(''':red[以下是运行成功的例子，可以参考]''')
    st.image("1.png",use_column_width=True)

    st.image("2.png",use_column_width=True)

    st.image("3.png",use_column_width=True)

    st.image("4.png",use_column_width=True)

    st.image("5.png",use_column_width=True)

    st.image("6.png",use_column_width=True)

    st.image("7.png",use_column_width=True)

    st.image("8.png",use_column_width=True)


st.divider()

# 初始化 session_state
if "tavily_api_key_set" not in st.session_state:
    st.session_state.tavily_api_key_set = bool(os.environ.get("TAVILY_API_KEY"))
    print("tavily_api_key_set:", st.session_state.tavily_api_key_set)

if "langchain_api_key_set" not in st.session_state:
    st.session_state.langchain_api_key_set = bool(os.environ.get("LANGCHAIN_API_KEY"))
    print("langchain_api_key_set:", st.session_state.langchain_api_key_set)

if "rerun" not in st.session_state:
    st.session_state.rerun = False


def set_environment_variable(key, value):
    os.environ[key] = value


# 检测并显示环境变量状态
column_check_tavily, column_check_langchain = st.columns([1, 1])

import os

# 初始化 session_state

if "langchain_api_key_set" not in st.session_state:
    st.session_state.langchain_api_key_set = bool(os.environ.get("LANGCHAIN_API_KEY"))

if "tavily_api_key_set" not in st.session_state:
    st.session_state.tavily_api_key_set = bool(os.environ.get("TAVILY_API_KEY"))

if "show_dialog" not in st.session_state:
    st.session_state.show_dialog = False

if "dialog_active" not in st.session_state:
    st.session_state.dialog_active = False

if "langchain_toast_shown" not in st.session_state:
    st.session_state.langchain_toast_shown = False

if "tavily_toast_shown" not in st.session_state:
    st.session_state.tavily_toast_shown = False

if "langchain_api_key" not in st.session_state:
    st.session_state.langchain_api_key = ""

if "tavily_api_key" not in st.session_state:
    st.session_state.tavily_api_key = ""

if "tavily_tool" not in st.session_state:
    st.session_state.tavily_tool = None

if "show_rerun_button" not in st.session_state:
    st.session_state.show_rerun_button = False

# 检测并显示环境变量状态
if not st.session_state.langchain_api_key_set and not os.environ.get(
        "LANGCHAIN_API_KEY"):
    langchain_info = st.info("未设置 LANGCHAIN_API_KEY")
elif os.environ.get(
        "LANGCHAIN_API_KEY") and not st.session_state.dialog_active and not st.session_state.langchain_toast_shown:
    st.toast("LANGCHAIN_API_KEY 已在环境变量中配置成功", icon='🌟')
    st.toast("LANGSMITH 配置成功", icon='🌟')
    st.session_state.langchain_toast_shown = True

if not st.session_state.tavily_api_key_set and not os.environ.get(
        "TAVILY_API_KEY"):
    tavily_info = st.info("未设置 TAVILY_API_KEY")
elif os.environ.get(
        "TAVILY_API_KEY") and not st.session_state.dialog_active and not st.session_state.tavily_toast_shown:
    st.toast("TAVILY_API_KEY 已在环境变量中成功", icon='🌟')
    st.toast(":red[注意：]请确保你的 TAVILY_API_KEY 有效，TAVILY_TOOL工具可以配置，但似乎只会在运行中报错，",
             icon='🚫')
    st.session_state.tavily_toast_shown = True

# 侧边栏按钮触发对话框
with st.sidebar:
    if st.button("设置 LANGCHAIN_API_KEY 和 TAVILY_API_KEY"):
        st.session_state.show_dialog = True
        set_api_keys()

    if st.button("删除环境变量中的 LANGCHAIN_API_KEY"):
        delete_environment_variable("LANGCHAIN_API_KEY")
    if st.button("删除环境变量中的 TAVILY_API_KEY"):
        delete_environment_variable("TAVILY_API_KEY")


# 捕获输出的函数
# 捕获输出的函数
class OutputCatcher(io.StringIO):
    def __init__(self):
        super().__init__()
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.initial_check_done = False

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, *args):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.seek(0)
        output = self.read()
        if "Failed to batch ingest runs: LangSmithError" in output:
            st.toast("再次提醒，你的Langchain_Api_Key是错误的，你可以忽略这个提醒，但是如果你要监控流程，务必使用正确的KEY",
                     icon="⚠️")
        self.close()

    def check_output(self):
        self.seek(0)
        output = self.read()
        if "Failed to batch ingest runs: LangSmithError" in output:
            st.toast("注意，你的Langchain_Api_Key是错误的，LangSmith无法配置，但是仍然可以运行", icon="⚠️")
        self.truncate(0)
        self.seek(0)


if not (openai_api_key or zhipu_api_key):
    st.info("请输入OpenAI API Key或者智谱AI的API key")

if openai_api_key and zhipu_api_key:
    st.info("有两个api，请选择一个使用即可")

if (openai_api_key or zhipu_api_key) and (
        not os.environ.get("TAVILY_API_KEY") and not os.environ.get("LANGCHAIN_API_KEY")):
    st.info("环境变量缺失:red[LANGCHAIN_API_KEY]和:red[TAVILY_API_KEY]")

if (
        openai_api_key and os.environ.get("LANGCHAIN_API_KEY") and os.environ.get("TAVILY_API_KEY") and not zhipu_api_key) or (
        zhipu_api_key and os.environ.get("LANGCHAIN_API_KEY") and os.environ.get("TAVILY_API_KEY") and not openai_api_key):
    column1, cloumn2 = st.columns([1, 1])
    with column1:
        research_agent_prompt = st.text_area('你可以在这修改:red[research_agent]的prompt(用于搜索数据)\n\n默认是：',
                                             "You should provide accurate data for use, "
        "and source code shouldn't be the final answer",
                                             height=150

                                             )
        st.write("如果你不清楚，尽量不要修改，上述的prompt已验证在:red[gpt-4/gpt-4-1106-preview]可运行")

    with cloumn2:
        chart_agent_prompt = st.text_area('你可以在这修改:red[chart_agent]的prompt（用于绘图）\n\n默认是：',
                                          "Create the python code to display the chart."

                                          ,
                                          height=150)
        st.write("如果你不清楚，尽量不要修改，上述的prompt已验证在:red[gpt-4/gpt-4-1106-preview]可运行")

    st.divider()
    model_name = st.text_input("输入你要使用的模型:",placeholder= "gpt-4-1106-preview")
    human_prompt = st.text_area("请输入你想要完成的搜索和绘制图表的任务",
                                placeholder=
                                "Fetch the UK's GDP over the "
                                "past 5 years, then draw a line "
                                "graph of it. Once you code it "
                                "up, finish.")
    start_button = st.button("开始")

    if start_button:
        if openai_api_key:
            api_key = openai_api_key
            api_base = openai_api_base

        else:
            api_key = zhipu_api_key
            api_base = zhipu_api_base

        # —————————————————————————————————————— 创建 llm 实例——————————————————————————————————————
        with st.spinner("创建LLM实例和Agents和Nodes中，请稍等..."):
            all_total_tokens = 0

            llm = create_llm(model_name, api_key, api_base)

            # 重新创建 agents 和 nodes
            research_agent = create_agent(
                llm,
                [st.session_state.tavily_tool],
                system_message=research_agent_prompt,
            )
            research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

            chart_agent = create_agent(
                llm,
                [python_repl],
                system_message=chart_agent_prompt,
            )
            chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")

            tools = [st.session_state.tavily_tool, python_repl]
            tool_node = ToolNode(tools)
            st.toast('创建Agents和Nodes成功啦~', icon='🌟')

        # ————————————————————————————————————创建 workflow——————————————————————————————————————
        with st.spinner("创建Workflow和Graph中，请稍等..."):

            workflow = StateGraph(AgentState)

            workflow.add_node("Researcher", research_node)
            workflow.add_node("chart_generator", chart_node)
            workflow.add_node("call_tool", tool_node)

            workflow.add_conditional_edges(
                "Researcher",
                router,
                {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},
            )
            workflow.add_conditional_edges(
                "chart_generator",
                router,
                {"continue": "Researcher", "call_tool": "call_tool", "__end__": END},
            )

            workflow.add_conditional_edges(
                "call_tool",
                # Each agent node updates the 'sender' field
                # the tool calling node does not, meaning
                # this edge will route back to the original agent
                # who invoked the tool
                lambda x: x["sender"],
                {
                    "Researcher": "Researcher",
                    "chart_generator": "chart_generator",
                },
            )
            workflow.set_entry_point("Researcher")
            st.toast('创建Workflow成功啦~', icon='🌟')
            graph = workflow.compile()
            total_tokens = 0  # Initialize token counter
            st.toast('创建Graph成功啦~', icon='🌟')

        # ——————————————————————————————————————运行Graph——————————————————————————————————
        try:
            with OutputCatcher() as output_catcher:
                with st.spinner("AI正在思考中，请稍等..."):

                    events = graph.stream(
                        {
                            "messages": [
                                HumanMessage(content=human_prompt)
                            ],
                        },
                        # Maximum number of steps to take in the graph
                        {"recursion_limit": 150},
                    )

                    with get_openai_callback() as cb:
                        event_list = []
                        code_detected = False
                        st.info("运行日志：")
                        for s in events:
                            event_list.append(s)

                            if not output_catcher.initial_check_done:
                                output_catcher.check_output()
                                output_catcher.initial_check_done = True

                            # st.write(s)
                            # st.write("————————————————————————————————————————————————————————————————————————")
                            display_researcher_data(event_list)  # 只显示Researcher部分
                            display_call_tool_data(event_list)  # 只显示call_tool部分
                            display_chart_generator_data(event_list)  # 只显示chart_generator部分
                            code = extract_code_from_events(event_list)
                            if code:
                                st.toast('提取代码成功啦~', icon='🌟')
                                modified_code = modify_code_for_streamlit(code)
                                exec(modified_code)
                                code_detected = True  # 检测到代码

                            all_total_tokens += cb.total_tokens

                        if not code_detected:
                            st.error("未找到包含代码的日志条目,你可能可以从日志中找到答案。")
                        st.info(f"本次运行消耗的Tokens: {all_total_tokens}")
        except Exception as e:
            st.toast(f"运行出错: {str(e)}", icon='🚫')
            st.error(f"运行出错: {str(e)}")

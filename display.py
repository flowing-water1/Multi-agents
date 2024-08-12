import json
import time

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage


def extract_code_from_events(events):
    final_answer_index = -1
    code = None

    # 找到包含'FINAL ANSWER'的位置，并提取之前的代码
    for i, s in enumerate(events):
        if 'FINAL ANSWER' in str(s):
            final_answer_index = i
            break

    # 提取包含代码的日志条目
    for s in events[:final_answer_index]:
        if 'chart_generator' in s:
            chart_data = s['chart_generator']
            for message in chart_data['messages']:
                if isinstance(message, AIMessage):
                    tool_calls = message.additional_kwargs.get('tool_calls', [])
                    for tool_call in tool_calls:
                        if tool_call['function']['name'] == 'python_repl':
                            code = json.loads(tool_call['function']['arguments'])['code']

                            st.code(code, language="python")
                            break
        elif 'call_tool' in s:
            call_tool_data = s['call_tool']
            for message in call_tool_data['messages']:
                if isinstance(message, ToolMessage) and 'code' in message.content:
                    code = json.loads(message.content)['code']
                    st.code(code, language="python")
                    break
    return code


def modify_code_for_streamlit(code):
    # 修改代码，将 plt.show() 改为 st.pyplot()
    code = code.replace("plt.show()", "st.pyplot(plt)")
    return code


def display_researcher_data(events):
    for s in events:
        if 'Researcher' in s:
            researcher_data = s['Researcher']
            for message in researcher_data['messages']:
                tool_info = {
                    "Tool": "Researcher",
                    "Tool Calls": [{"name": tool_call['function']['name']} for tool_call in
                                   message.additional_kwargs.get('tool_calls', [])],
                    "Content": message.content
                }
                st.json(tool_info)
                st.divider()

import re
def clean_json_string(json_str):
    # 移除控制字符
    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
    return json_str

def display_call_tool_data(events):
    for s in events:
        if 'call_tool' in s:
            call_tool_data = s['call_tool']
            for message in call_tool_data['messages']:
                if isinstance(message, ToolMessage):
                    if 'HTTPError' in message.content:
                        st.toast(f"tavily_tool调用失败：{message.content}，请检查你的tavily_key是否正确")
                        st.error(f"tavily_tool调用失败：{message.content}，请检查你的_tavily_key是否正确")
                        time.sleep(2)
                        st.stop()
                    else:
                        tool_info = {
                            "Tool": "call_tool",
                            "Tool Calls": message.name,
                            "Content": message.content
                        }
                        # 如果调用的工具是tavily_search_results_json的话，将url和content分别打印出来：
                        if tool_info["Tool Calls"] == "tavily_search_results_json":
                            tool_info["Entries"] = []
                            cleaned_content = clean_json_string(tool_info["Content"])
                            content_data = json.loads(cleaned_content)
                            for entry in content_data:
                                entry_info = {
                                    "url": entry.get('url'),
                                    "content": entry.get('content')
                                }
                                tool_info["Entries"].append(entry_info)
                            # 打印tool_info
                            tool_info_tavily = {
                                "Tool": "call_tool",
                                "Tool Calls": message.name,
                                "Content": tool_info["Entries"]
                            }
                            st.json(tool_info_tavily)
                            st.divider()
                        else:
                            st.json(tool_info)
                            st.divider()


def display_chart_generator_data(events):
    for s in events:
        if 'chart_generator' in s:
            chart_data = s['chart_generator']
            for message in chart_data['messages']:
                tool_info = {
                    "Tool": "chart_generator",
                    "Tool Calls": [{"name": tool_call['function']['name']} for tool_call in
                                   message.additional_kwargs.get('tool_calls', [])],
                    "Content": message.content
                }
                st.json(tool_info)
                st.divider()

import uuid
import tempfile
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage
from langgraph.types import Command
from backend import (
    chatbot,
    get_all_threads,
    ingest_rag_document,
    delete_thread_from_db,
)

# Map raw tool names to clean UI status labels
TOOL_LABELS = {
    "tavily_search_results_json": "Searching the web",
    "tavily_search": "Searching the web",
    "rag_tool": "Searching uploaded PDF documents",
    "calculator": "Calculating math expression",
    "get_stock_price": "Fetching live stock price",
    "purchase_stock": "Preparing stock purchase order",
    "get_weather": "Checking current weather",
}


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def add_thread(thread_id: str, title: str = "New Chat"):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
    st.session_state["chat_titles"][thread_id] = title


def reset_chat():
    new_thread_id = generate_thread_id()
    st.session_state["thread_id"] = new_thread_id
    st.session_state["message_history"] = []
    st.session_state["pending_hitl"] = None
    add_thread(new_thread_id, title="New Chat")


def load_conversation(thread_id: str):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def sync_ui_history(thread_id: str):
    """Converts saved LangChain messages into Streamlit session_state format."""
    messages = load_conversation(thread_id)
    temp_messages = []
    for message in messages:
        if isinstance(message, HumanMessage):
            temp_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage) and message.content:
            temp_messages.append({"role": "assistant", "content": message.content})
    st.session_state["message_history"] = temp_messages


def delete_chat(thread_id: str):
    """Deletes a conversation from SQLite and updates Streamlit session state."""
    delete_thread_from_db(thread_id)

    if thread_id in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].remove(thread_id)
    st.session_state["chat_titles"].pop(thread_id, None)

    if st.session_state.get("pending_hitl", {}) and st.session_state["pending_hitl"].get("thread_id") == thread_id:
        st.session_state["pending_hitl"] = None

    if st.session_state.get("thread_id") == thread_id:
        if st.session_state["chat_threads"]:
            next_thread = st.session_state["chat_threads"][-1]
            st.session_state["thread_id"] = next_thread
            sync_ui_history(next_thread)
        else:
            reset_chat()


def get_thread_title_from_state(thread_id: str) -> str:
    """Fallback helper to extract the first HumanMessage title for a thread."""
    for msg in load_conversation(thread_id):
        if isinstance(msg, HumanMessage) and msg.content:
            text = msg.content.strip()
            return text[:25] + ("..." if len(text) > 25 else "")
    return "New Chat"


def get_pending_interrupt(thread_id):
    """Return the first unresolved LangGraph interrupt for a thread, or None."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state_snapshot = chatbot.get_state(config)
        direct_interrupts = getattr(state_snapshot, "interrupts", ()) or ()
        if direct_interrupts:
            return direct_interrupts[0]

        tasks = getattr(state_snapshot, "tasks", ()) or ()
        for task in tasks:
            task_interrupts = getattr(task, "interrupts", ()) or ()
            if task_interrupts:
                return task_interrupts[0]
    except Exception:
        return None
    return None


def save_pending_interrupt(thread_id, interrupt_object):
    """Save the pending interrupt information inside Streamlit state."""
    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value),
    }


def sync_pending_interrupt(thread_id):
    """Synchronize Streamlit HITL state with the LangGraph checkpoint."""
    pending_interrupt = get_pending_interrupt(thread_id)
    if pending_interrupt is not None:
        save_pending_interrupt(thread_id, pending_interrupt)
    else:
        current_pending = st.session_state.get("pending_hitl")
        if current_pending is not None and current_pending.get("thread_id") == thread_id:
            st.session_state["pending_hitl"] = None


def resume_hitl_execution(decision: str):
    """
    Resume an interrupted LangGraph execution with 'yes' or 'no'.
    """
    pending_hitl = st.session_state.get("pending_hitl")
    if not pending_hitl:
        st.warning("There is no pending action to approve or reject.")
        return

    interrupted_thread_id = pending_hitl["thread_id"]
    resume_config = {
        "configurable": {"thread_id": interrupted_thread_id},
        "metadata": {"thread_id": interrupted_thread_id},
        "run_name": "hitl_resume_trace",
    }

    try:
        with st.chat_message("assistant"):
            status_holder = {
                "box": st.status("🔄 Resuming the requested action...", expanded=True)
            }

            def resumed_ai_only_stream():
                for message_chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages",
                ):
                    node_name = metadata.get("langgraph_node", "")

                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        status_holder["box"].update(
                            label=f"🔧 Completed `{tool_name}`",
                            state="running",
                            expanded=True,
                        )
                        status_holder["box"].write(f"✅ Executed `{tool_name}`")

                    if isinstance(message_chunk, (AIMessage, AIMessageChunk)) and node_name == "chat_node":
                        if message_chunk.content:
                            yield message_chunk.content

            resumed_ai_message = st.write_stream(resumed_ai_only_stream())

            next_interrupt = get_pending_interrupt(interrupted_thread_id)
            if next_interrupt is not None:
                save_pending_interrupt(interrupted_thread_id, next_interrupt)
                status_holder["box"].update(
                    label="⚠️ Another approval is required",
                    state="complete",
                    expanded=False,
                )
            else:
                st.session_state["pending_hitl"] = None
                status_holder["box"].update(
                    label="✅ Action completed",
                    state="complete",
                    expanded=False,
                )

        if resumed_ai_message:
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": resumed_ai_message,
            })

        st.rerun()

    except Exception as error:
        st.error(f"Could not resume the requested action: {error}")


# =========================== Session State Initialization ==========================

st.title("Agentic Chatbot with LangGraph")

if "chat_titles" not in st.session_state or not isinstance(st.session_state["chat_titles"], dict):
    raw_threads = get_all_threads()
    st.session_state["chat_titles"] = {
        tid: get_thread_title_from_state(tid) for tid in raw_threads
    }

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = list(st.session_state["chat_titles"].keys())

if "thread_id" not in st.session_state:
    if not st.session_state["chat_threads"]:
        initial_thread = generate_thread_id()
        st.session_state["thread_id"] = initial_thread
        add_thread(initial_thread, title="New Chat")
        st.session_state["message_history"] = []
    else:
        st.session_state["thread_id"] = st.session_state["chat_threads"][-1]
        sync_ui_history(st.session_state["thread_id"])

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

# Keep HITL state synced with LangGraph checkpoint on every reload/thread switch
sync_pending_interrupt(st.session_state["thread_id"])


# =========================== Sidebar (Threads & Delete) ============================

with st.sidebar:
    st.title("My Conversations")

    if st.button("➕ New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.divider()

    for thread_id in st.session_state["chat_threads"][::-1]:
        button_label = st.session_state["chat_titles"].get(thread_id, "New Chat")
        is_active = thread_id == st.session_state["thread_id"]

        col_chat, col_del = st.columns([4, 1])

        with col_chat:
            btn_type = "primary" if is_active else "secondary"
            if st.button(
                button_label,
                key=f"select_{thread_id}",
                use_container_width=True,
                type=btn_type,
            ):
                st.session_state["thread_id"] = thread_id
                sync_ui_history(thread_id)
                sync_pending_interrupt(thread_id)
                st.rerun()

        with col_del:
            if st.button(
                "🗑️",
                key=f"del_{thread_id}",
                help="Delete this conversation",
                use_container_width=True,
            ):
                delete_chat(thread_id)
                st.rerun()


# =========================== Main Chat Interface ===================================

# Render existing conversation messages
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Render HITL Approval Card if the current thread is paused on an interrupt
pending_hitl = st.session_state.get("pending_hitl")
if pending_hitl and pending_hitl.get("thread_id") == st.session_state["thread_id"]:
    with st.container(border=True):
        st.warning(f"⚠️ **Human Approval Required**\n\n{pending_hitl['prompt']}")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("✅ Approve (Yes)", type="primary", use_container_width=True):
                resume_hitl_execution("yes")
        with col_no:
            if st.button("❌ Reject (No)", use_container_width=True):
                resume_hitl_execution("no")

# Chat input with direct PDF attachment enabled (disabled while awaiting approval)
chat_submission = st.chat_input(
    "Approve or reject the pending action above..."
    if (pending_hitl and pending_hitl.get("thread_id") == st.session_state["thread_id"])
    else "Type here or attach a PDF...",
    accept_file=True,
    file_type=["pdf"],
    disabled=bool(pending_hitl and pending_hitl.get("thread_id") == st.session_state["thread_id"]),
)

if chat_submission:
    user_input = chat_submission.text
    uploaded_files = chat_submission.files

    # If the user attached a PDF directly in the chat bar, index it automatically
    if uploaded_files:
        for pdf_file in uploaded_files:
            with st.status(f"📄 Indexing attached PDF: **{pdf_file.name}**...", expanded=False) as pdf_status:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_file.read())
                    tmp_path = tmp.name
                num_chunks = ingest_rag_document(tmp_path)
                pdf_status.update(
                    label=f"📄 Indexed `{pdf_file.name}` ({num_chunks} chunks)",
                    state="complete",
                )
        if not user_input:
            user_input = f"I have uploaded {uploaded_files[0].name}. Summarize its main points."

    if user_input:
        current_thread = st.session_state["thread_id"]

        # Update title on first user message
        if st.session_state["chat_titles"].get(current_thread, "New Chat") == "New Chat":
            clean_input = user_input.strip()
            truncated_title = clean_input[:25] + ("..." if len(clean_input) > 25 else "")
            st.session_state["chat_titles"][current_thread] = truncated_title

        # Append and display user message
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        CONFIG = {
            "configurable": {"thread_id": current_thread},
            "metadata": {"thread_id": current_thread},
            "run_name": "chat_trace",
        }

        # Stream assistant response and tool status
        with st.chat_message("assistant"):
            status_box = st.status("Thinking...", expanded=True)
            message_placeholder = st.empty()

            full_response = ""
            used_tools = []
            seen_tool_calls = set()

            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                node_name = metadata.get("langgraph_node", "")

                # 1. Detect tool selection
                if isinstance(message_chunk, AIMessageChunk) and message_chunk.tool_call_chunks:
                    for tc in message_chunk.tool_call_chunks:
                        tool_name = tc.get("name")
                        tc_id = tc.get("id")
                        if tool_name and (tc_id or tool_name) not in seen_tool_calls:
                            seen_tool_calls.add(tc_id or tool_name)
                            friendly_name = TOOL_LABELS.get(tool_name, f"Running `{tool_name}`")
                            used_tools.append(friendly_name)

                            status_box.update(
                                label=f"Using tool: **{friendly_name}**...",
                                state="running",
                            )
                            status_box.write(f"🔧 Selected tool: `{tool_name}`")

                # 2. Detect tool completion
                elif isinstance(message_chunk, ToolMessage):
                    status_box.write(f"✅ Completed `{message_chunk.name}`")

                # 3. Stream final answer tokens from chat_node
                elif (
                    isinstance(message_chunk, AIMessageChunk)
                    and message_chunk.content
                    and node_name == "chat_node"
                ):
                    full_response += message_chunk.content
                    message_placeholder.markdown(full_response + "▌")

            # Check if the stream stopped due to a human-in-the-loop interrupt
            pending_interrupt = get_pending_interrupt(current_thread)

            if pending_interrupt is not None:
                save_pending_interrupt(current_thread, pending_interrupt)
                status_box.update(
                    label="⏸️ Waiting for human approval...",
                    state="running",
                    expanded=False,
                )
                message_placeholder.empty()
            else:
                message_placeholder.markdown(full_response)
                if used_tools:
                    unique_tools = ", ".join(dict.fromkeys(used_tools))
                    status_box.update(
                        label=f"Used tools: {unique_tools}",
                        state="complete",
                        expanded=False,
                    )
                else:
                    status_box.update(
                        label="Answered from base knowledge",
                        state="complete",
                        expanded=False,
                    )

                if full_response:
                    st.session_state["message_history"].append(
                        {"role": "assistant", "content": full_response}
                    )

        st.rerun()
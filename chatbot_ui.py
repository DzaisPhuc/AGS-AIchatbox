#  import necessary libraries
import sys
import os
import threading #cho phép chạy các tác vụ song song
import tkinter as tk #thư viện giao diện người dùng
from tkinter import scrolledtext #cho phép tạo vùng văn bản cuộn
from src import state #file tự tạo để chia sẻ trạng thái hội thoại giữa AI và UI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))) #thêm src vào đg dẫn tìm kiếm

from ags_ai_agent.crew import AgsAiAgent #import lớp trung tâm gọi AI crew hoạt động

root = tk.Tk()
root.title("Crew AI Chatbox")
root.geometry("700x500")

chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', font=("Arial", 11))
chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True) #tạo vùng văn bản cuộn để hiển thị hội thoại

user_input = tk.Entry(root, font=("Arial", 12))
user_input.pack(padx=10, pady=5, fill=tk.X) #tạo ô nhập liệu cho người dùng

def send_message():
    msg = user_input.get().strip()
    if not msg:
        return

    if not state.awaiting_input:
        chat_area.config(state='normal')
        chat_area.insert(tk.END, "⚠️ AI chưa sẵn sàng nhận phản hồi. Vui lòng đợi câu hỏi.\n")
        chat_area.config(state='disabled')
        chat_area.see(tk.END)
        return

    # Nếu sẵn sàng nhận phản hồi
    chat_area.config(state='normal') #cho phép chỉnh sửa vùng văn bản
    chat_area.insert(tk.END, f"You: {msg}\n") #hiển thị tin nhắn người dùng trong vùng văn bản
    chat_area.config(state='disabled') #cập nhật vùng văn bản với tin nhắn người dùng tức dòng cuối cùng
    chat_area.see(tk.END) #cuộn xuống cuối để hiển thị tin nhắn mới

    state.last_response = msg #lưu phản hồi của người dùng vào biến toàn cục
    state.awaiting_input = False #đặt trạng thái chờ phản hồi của AI về False
    user_input.delete(0, tk.END) #xóa nội dung ô nhập liệu sau khi gửi


def run_crew():
    def task():
        try:
            AgsAiAgent().crew().kickoff(inputs={"topic": ""})
        except Exception as e:
            chat_area.config(state='normal')
            chat_area.insert(tk.END, f"Bot: Lỗi: {e}\n")
            chat_area.config(state='disabled')
            chat_area.see(tk.END)
    threading.Thread(target=task).start()

user_input.bind("<Return>", lambda event: send_message())
tk.Button(root, text="Gửi", command=send_message).pack(pady=5)
def check_bot_prompt():
    if state.last_prompt:
        chat_area.config(state='normal')
        chat_area.insert(tk.END, f"Bot: {state.last_prompt}\n")
        chat_area.config(state='disabled')
        chat_area.see(tk.END)
        state.last_prompt = ""
    root.after(500, check_bot_prompt)

check_bot_prompt()
run_crew()

def on_close():
    print("🔴 Đang thoát chương trình...")
    root.destroy()         # Đóng cửa sổ giao diện
    os._exit(0)            # Dừng tất cả luồng, kể cả CrewAI đang chạy nền

root.protocol("WM_DELETE_WINDOW", on_close)  # Bắt sự kiện tắt cửa sổ

root.mainloop()

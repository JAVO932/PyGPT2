import tkinter as tk
from tkinter import ttk, messagebox
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import threading
import gc
def open_about():
    about_win = tk.Toplevel(root)
    about_win.title("About PyGPT2")
    about_win.geometry("400x400")
    about_win.resizable(False, False)
    about_win.configure(bg="#222222")  # Siyah arka plan

    # Scrollable Text widget kullanmak daha iyi olur uzun metin için
    text_frame = tk.Frame(about_win, bg="#222222")
    text_frame.pack(expand=True, fill="both", padx=10, pady=10)

    scrollbar = tk.Scrollbar(text_frame)
    scrollbar.pack(side="right", fill="y")

    about_text = tk.Text(
        text_frame,
        font=("Consolas", 10),
        bg="#222222",
        fg="white",
        wrap="word",
        yscrollcommand=scrollbar.set
    )
    about_text.pack(expand=True, fill="both")
    scrollbar.config(command=about_text.yview)

    about_text.insert("1.0", """MIT License

Copyright (c) 2025 Ege Önder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.""")
    about_text.config(state="disabled")  # Sadece okunabilir

    tk.Button(
        about_win,
        text="Close",
        command=about_win.destroy,
        font=("Consolas", 11, "bold"),
        bg="#4CAF50",
        fg="white"
    ).pack(pady=10)

# --- Tekrarları temizleme ---
def clean_repetitions(text, max_repeat=20, ignore_words={"and", "or", "but", "is"}):
    words = text.split()
    cleaned = []
    count = 0
    last_word = None
    for w in words:
        if w in ignore_words:
            count = 1
        elif w == last_word:
            count += 1
        else:
            count = 1
        if count <= max_repeat:
            cleaned.append(w)
        last_word = w
    return " ".join(cleaned)

# --- Başlangıç değerleri ---
MODEL_NAME = "gpt2-medium"
DEVICE_NAME = "CPU"

# --- Model yükleme ---
def load_model(model_name, device_name):
    global tokenizer, model, MODEL_NAME, DEVICE_NAME
    try:
        output_box.configure(state="normal")
        output_box.insert("end", f"[System] Loading {model_name} on {device_name}...\n")
        output_box.see("end")
        output_box.configure(state="disabled")
        root.update()

        # Önceki modeli temizle
        try:
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass

        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)

        device = torch.device("cpu")
        if "GPU" in device_name:
            gpu_index = int(device_name.split()[1])
            device = torch.device(f"cuda:{gpu_index}")
        model.to(device)

        MODEL_NAME = model_name
        DEVICE_NAME = device_name

        output_box.configure(state="normal")
        output_box.insert("end", f"[System] {model_name} loaded successfully on {device_name}!\n\n")
        output_box.see("end")
        output_box.configure(state="disabled")
    except Exception as e:
        messagebox.showerror("Error", f"Model load failed: {e}")

# --- Prompt gönderme (thread'li) ---
def send_prompt_threaded():
    threading.Thread(target=send_prompt, daemon=True).start()

def send_prompt():
    user_input = prompt_entry.get().strip()
    if not user_input:
        return

    output_box.configure(state="normal")
    output_box.insert("end", f"> {user_input}\n")
    output_box.see("end")
    output_box.configure(state="disabled")
    prompt_entry.delete(0, "end")

    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)  # maskeyi manuel ekle
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = reply[len(user_input):].strip()
    reply = clean_repetitions(reply, max_repeat=3)

    output_box.configure(state="normal")
    output_box.insert("end", f"GPT-2: {reply}\n\n")
    output_box.see("end")
    output_box.configure(state="disabled")

# --- Settings menüsü (model ve device ayrı, siyah arka plan) ---
def open_settings():
    settings_win = tk.Toplevel(root)
    settings_win.title("Settings")
    settings_win.geometry("360x250")
    settings_win.resizable(False, False)
    settings_win.configure(bg="#222222")  # Siyah arka plan

    tk.Label(settings_win, text="Settings", font=("Consolas", 14, "bold"),
             bg="#222222", fg="white").pack(pady=10)  # Beyaz yazı

    # Model seçimi
    tk.Label(settings_win, text="Select Model:", font=("Consolas", 11),
             bg="#222222", fg="white").pack(anchor="w", padx=20)
    model_var = tk.StringVar(value=MODEL_NAME)
    model_menu = ttk.Combobox(
        settings_win, textvariable=model_var,
        values=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        state="readonly", font=("Consolas", 11)
    )
    model_menu.pack(fill="x", padx=20, pady=5)

    # Device seçimi
    tk.Label(settings_win, text="Select Device:", font=("Consolas", 11),
             bg="#222222", fg="white").pack(anchor="w", padx=20)
    gpu_count = torch.cuda.device_count()
    devices = ["CPU"] + [f"GPU {i} ({torch.cuda.get_device_name(i)})" for i in range(gpu_count)]
    device_var = tk.StringVar(value=DEVICE_NAME)
    device_menu = ttk.Combobox(
        settings_win, textvariable=device_var,
        values=devices,
        state="readonly",
        font=("Consolas", 11)
    )
    device_menu.pack(fill="x", padx=20, pady=5)

    def apply_settings():
        chosen_model = model_var.get()
        chosen_device = device_var.get()
        load_model(chosen_model, DEVICE_NAME)  # sadece model değiştirme
        load_model(MODEL_NAME, chosen_device)  # sadece device değiştirme
        settings_win.destroy()

    apply_button = tk.Button(settings_win, text="Apply", command=apply_settings,
                             font=("Consolas", 11, "bold"), bg="#4CAF50", fg="white")
    apply_button.pack(pady=15)
    device_menu.pack(fill="x", padx=20, pady=5)

    def apply_settings():
        # Model ve device ayrı ayrı yüklenebilir
        chosen_model = model_var.get()
        chosen_device = device_var.get()
        load_model(chosen_model, DEVICE_NAME)  # sadece model değiştirme
        load_model(MODEL_NAME, chosen_device)  # sadece device değiştirme
        settings_win.destroy()

    apply_button = tk.Button(settings_win, text="Apply", command=apply_settings,
                             font=("Consolas", 11, "bold"), bg="#4CAF50", fg="white")
    apply_button.pack(pady=15)

# --- Tkinter UI ---
root = tk.Tk()
root.title("PyGPT2")
root.geometry("750x550")
root.configure(bg="#222222")

# Menü bar
menubar = tk.Menu(root)
root.config(menu=menubar)
settings_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Settings", menu=settings_menu)
settings_menu.add_command(label="Model / Device...", command=open_settings)
menubar.add_cascade(label="About", command=open_about)

# Çıktı ekranı
output_frame = tk.Frame(root, bg="#222222")
output_frame.pack(expand=True, fill="both", padx=10, pady=10)
output_box = tk.Text(output_frame, wrap="word", font=("Consolas", 11),
                     bg="#1e1e1e", fg="#00FF00", bd=2, relief="sunken")
output_box.pack(expand=True, fill="both", side="left")
scrollbar = tk.Scrollbar(output_frame, command=output_box.yview)
scrollbar.pack(side="right", fill="y")
output_box.config(yscrollcommand=scrollbar.set, state="disabled")

# Prompt alanı
frame = tk.Frame(root, bg="#222222")
frame.pack(fill="x", padx=10, pady=5)

prompt_label = tk.Label(frame, text="Prompt:", font=("Consolas", 11), bg="#222222", fg="white")
prompt_label.pack(side="left", padx=5)

prompt_entry = tk.Entry(frame, font=("Consolas", 11), bg="#333333", fg="white", insertbackground="white")
prompt_entry.pack(side="left", expand=True, fill="x", padx=5, pady=5)

send_button = tk.Button(frame, text="Send", command=send_prompt_threaded,
                        font=("Consolas", 11, "bold"), bg="#4CAF50", fg="white")
send_button.pack(side="right", padx=5)

root.bind("<Return>", lambda event: send_prompt_threaded())

# İlk model yükle
output_box.configure(state="normal")
output_box.insert("end", f"[System] Loading {MODEL_NAME} on {DEVICE_NAME}...\n")
output_box.see("end")
output_box.configure(state="disabled")
root.update()
load_model(MODEL_NAME, DEVICE_NAME)

root.mainloop()

# PyGPT2

**PyGPT2** is a lightweight Python application that provides a simple GUI for interacting with GPT-2 models locally. It is designed for users who want to experiment with GPT-2 text generation without the need for complex setups or cloud services. The application allows selecting different GPT-2 models, switching between CPU and GPU devices, and sending prompts for AI text generation with basic repetition cleaning.

---

## Features

- **Model Selection:** Easily switch between the available GPT-2 variants:
  - `gpt2`
  - `gpt2-medium`
  - `gpt2-large`
  - `gpt2-xl`
- **Device Selection:** Choose the computation device:
  - CPU
  - Any available GPU device
- **Threaded Prompt Handling:** Prompts are sent in a separate thread to prevent GUI freezing.
- **Repetition Cleaning:** Automatically reduces repeated words to make outputs more readable.
- **Dark-Themed GUI:** A comfortable dark interface with green-on-black text output.
- **Prompt History:** See previous inputs and AI responses in the scrollable output box.
- **Easy-to-Use:** Minimal installation requirements and intuitive interface.

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/EgeOnderX/PyGPT2.git
cd PyGPT2
```

2. Install dependencies (Python 3.8+ recommended):

```bash
pip install torch transformers tkinter
```

3. Run the application:

```bash
python main.py
```

---

## Usage

1. **Launch the app:** The main window shows a prompt entry box and an output box.
2. **Send a prompt:** Type your text into the prompt entry box and press `Enter` or click `Send`.
3. **Settings:** Open the `Settings` menu to change the GPT-2 model or switch between CPU and GPU.
4. **Read outputs:** Responses from GPT-2 appear in the scrollable output box with cleaned repetitions.

---

## Contributing

Contributions are welcome! Feel free to:

- Open issues for bugs or feature requests.
- Submit pull requests for improvements or new features.
- Suggest model optimizations or UI enhancements.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/) – GPT-2 model and tokenizer
- Python community – for libraries and support
- Tkinter – GUI framework for Python

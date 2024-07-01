# working 6-30-2024
# this script uses torch/cuda to run the model on gpu (if available), and lets you know in a print statement which one it's using

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch  # Ensure torch is imported
import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
import time
import threading
from torch.quantization import quantize_dynamic


# --- Module: Application Setup ---
class LLMGUI:
    def __init__(self, root):
        self.root = root
        self.setup_main_window()
        self.create_widgets()
        self.timer_running = False
        self.start_time = 0
        self.max_length = 50
        self.num_return_sequences = 1 # model generates this many sequences. 2 = 2x computation required
        self.no_repeat_ngram_size = 3 # prevents the model from reapeating any x-word phrases. Improves quality.
        self.top_k = 2 # limits the sampling to the top k most likely next tokens
        self.top_p = 0.9 # nucleus sampling. Lower = faster but lower quality
        self.temperature = 0.6  # controls the randomness of predictions lower = confident,less diverse; higher = less confident, more diverse
        self.skip_special_tokens = True # skip special tokens in the output
        print("initializing the model")
        self.tokenizer, self.model, self.device = intialize_model_mod()
        self.inputs = ""


    def setup_main_window(self):
        self.root.title("I am a llama farmer")
        self.root.geometry("900x600")  # Width x Height

    def create_widgets(self):
        # Input and output frames setup
        self.input_frame = tk.Frame(self.root, width=400, height=500, bg='lightgray')
        self.input_frame.grid(row=0, column=0, sticky="nsew")
        self.output_frame = tk.Frame(self.root, width=400, height=500, bg='white')
        self.output_frame.grid(row=0, column=1, sticky="nsew")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Input and output text boxes
        self.input_text = scrolledtext.ScrolledText(self.input_frame, wrap=tk.WORD, font=('Helvetica', 12), bg='white', fg='black')
        self.input_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.submit_button = tk.Button(self.input_frame, text="Submit", font=('Helvetica', 12), command=self.start_process)
        self.submit_button.pack(padx=10, pady=10, ipadx=5, ipady=5)
        self.output_text = scrolledtext.ScrolledText(self.output_frame, wrap=tk.WORD, font=('Helvetica', 12), bg='white', fg='darkblue')
        self.output_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def start_process(self):
        prompt = self.input_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showinfo("Error", "Please enter some text before submitting.")
            return
        if len(prompt) > (self.max_length):
            messagebox.showinfo("Error", "Please reduce input length")
            return
        self.timer_running = True
        threading.Thread(target=self.process_input, args=(prompt,)).start()
        self.start_time = time.time()
        threading.Thread(target=self.update_timer).start()

        

    def process_input(self, prompt):
        self.inputs = input_token_mod(self.tokenizer, prompt, self.device)
        simulated_output = self.simulate_llm_response(self.inputs)
        elapsed_time = time.time() - self.start_time
        self.timer_running = False
        self.output_text.delete('1.0', tk.END)
        display_output = f"{simulated_output}\n\nProcessing Time: {elapsed_time:.2f} seconds"
        self.output_text.insert(tk.END, display_output)

    def update_timer(self):
        while self.timer_running:
            elapsed_time = time.time() - self.start_time
            self.output_text.delete('1.0', tk.END)
            self.output_text.insert(tk.END, f"Processing... {elapsed_time:.2f} seconds")
            time.sleep(0.1)

    def simulate_llm_response(self, input_text):
        self.outputs = generate_text(self.model, self.tokenizer, self.inputs, self.num_return_sequences, self.no_repeat_ngram_size, self.max_length, self.top_k, self.top_p, self.temperature)
        generated_text = decode_output(self.tokenizer, self.outputs, self.skip_special_tokens)
        print(generated_text)
        
        return generated_text


def intialize_model_mod():
    # Check for CUDA availability and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load tokenizer and model
    print("loading tokenizer")
    model_directory = r"C:\Path\To\Your\Llama\Directory"
    config = AutoConfig.from_pretrained(model_directory)
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    print("loading model")
    model = AutoModelForCausalLM.from_pretrained(model_directory, config=config)

    # Move model to the specified device (GPU or CPU)
    model.to(device)
    
    return tokenizer, model, device


def input_token_mod(tokenizer, prompt, device):
    # Tokenize input text and move tokens to the same device as model
    print("creating input")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    return inputs

def generate_text(model, tokenizer, inputs, num_return_sequences, no_repeat_ngram_size, max_length, top_k, top_p, temperature):
    # Generate text
    print("creating output")
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=no_repeat_ngram_size,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    print("output complete")
    return outputs

def decode_output(tokenizer, outputs, skip_special_tokens):
    # Decode and return generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=skip_special_tokens)

def main():
    # Initialization parameters
    """
        max_length: Specifies the maximum length of the generated text in tokens. In your example, it's set to 5000 tokens, meaning the maximum output length should not exceed this limit.

        num_return_sequences: Defines how many different sequences or completions you want the model to generate. Set to 1 and the model will generate one output sequence.

        no_repeat_ngram_size: Prevents the model from generating repetitive sequences by setting a limit on how often specific n-grams (sequences of n tokens) can appear consecutively.
            Set to 2 means that the model will avoid repeating any two-token sequence in its outputs.

        top_k: Limits the sampling to the top k most likely next tokens. It restricts the model from considering tokens with lower probabilities
            during text generation. In your example, top_k is set to 50, so the model will only consider the top 50 tokens with the highest probabilities for each token position.

        top_p (nucleus sampling): Controls diversity by choosing the smallest set of tokens whose cumulative probability exceeds this threshold p.
            It allows for dynamic sampling where more tokens are considered if necessary. In your case, top_p is set to 0.95, meaning the model will consider tokens until the cumulative probability reaches 95%.

        Temperature: Controls the randomness of predictions by scaling the logits before applying softmax during sampling. Lower values make the model 
        more deterministic and repetitive, while higher values increase diversity and randomness. A temperature of 0.7 (as in your example) 
        tends to produce slightly conservative outputs, balancing between generating novel responses and staying close to probable predictions.
    """
    root = tk.Tk()
    app = LLMGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

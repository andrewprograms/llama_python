# working 6-30-2024
# this script is an example of how to use the LLaMA model to generate text in python locally without apps like ollama
# this script uses torch/cuda to run the model on gpu (if available), and lets you know in a print statement which one it's using

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig # Import the transformers stuff for the LLaMA model
import torch  # Import torch for running the model on gpu
import tkinter as tk # Import the Tkinter package for the gui
from tkinter import scrolledtext # Import the ScrolledText widget for the gui
from tkinter import messagebox # Import the messagebox widget for the gui
import time # Import the time package for tracking the time it takes to generate the text
import threading # Import the threading package for tracking the time it takes to generate the text


# --- Module: Application Setup ---
class LLMGUI:
    def __init__(self, root):
        """ Initializes the GUI, starting parameters, and initializes the model """
        self.root = root # Set the root of the GUI
        self.setup_main_window() # Set up the main window
        self.create_widgets() # Create the widgets for the GUI
        self.timer_running = False # strt state of the timer
        self.start_time = 0 # the start time of the timer
        self.max_length = 50 # the maximum length of the generated text
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
        self.root.title("I am a llama farmer") # Set the title of the window
        self.root.geometry("900x600")  # Width x Height of the window

    def create_widgets(self):
        """ Creates the widgets for the GUI """
        # Input and output frames setup
        self.input_frame = tk.Frame(self.root, width=400, height=500, bg='lightgray') # Create the input frame
        self.input_frame.grid(row=0, column=0, sticky="nsew") # Position the input frame
        self.output_frame = tk.Frame(self.root, width=400, height=500, bg='white') # Create the output frame
        self.output_frame.grid(row=0, column=1, sticky="nsew") # Position the output frame
        self.root.grid_columnconfigure(0, weight=1) # Set the weight of the input frame
        self.root.grid_columnconfigure(1, weight=1) # Set the weight of the output frame

        # Input and output text boxes
        self.input_text = scrolledtext.ScrolledText(self.input_frame, wrap=tk.WORD, font=('Helvetica', 12), bg='white', fg='black') # Create the input text box
        self.input_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True) # Position the input text box
        self.submit_button = tk.Button(self.input_frame, text="Submit", font=('Helvetica', 12), command=self.start_process) # Create the submit button
        self.submit_button.pack(padx=10, pady=10, ipadx=5, ipady=5) # Position the submit button
        self.output_text = scrolledtext.ScrolledText(self.output_frame, wrap=tk.WORD, font=('Helvetica', 12), bg='white', fg='darkblue') # Create the output text box
        self.output_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True) # Position the output text box

    def start_process(self):
        """ Starts the process of generating text """
        prompt = self.input_text.get("1.0", tk.END).strip() # Get the text from the input box
        if not prompt: # Check if the input is empty
            messagebox.showinfo("Error", "Please enter some text before submitting.") # If the input is empty, show an error message
            return # Exit the function
        if len(prompt) > (self.max_length): # Check if the input is too long
            messagebox.showinfo("Error", "Please reduce input length") # If the input is too long, show an error message
            return # Exit the function
        # Start timer
        self.timer_running = True # Start the timer
        threading.Thread(target=self.process_input, args=(prompt,)).start() # Start the thread to process the input
        self.start_time = time.time() # Timer
        threading.Thread(target=self.update_timer).start() # Update the timer
        

    def process_input(self, prompt):
        """ Processes the input and generates text """
        self.inputs = input_token_mod(self.tokenizer, prompt, self.device) # Tokenize and move the input to the specified device
        simulated_output = self.simulate_llm_response(self.inputs) # Simulate the response from the LLM
        elapsed_time = time.time() - self.start_time # Get the elapsed time
        self.timer_running = False # Stop the timer
        self.output_text.delete('1.0', tk.END) # Clear the output text
        display_output = f"{simulated_output}\n\nProcessing Time: {elapsed_time:.2f} seconds" # Display the output
        self.output_text.insert(tk.END, display_output) # Insert the output

    def update_timer(self):
        """ Updates the timer """
        while self.timer_running:
            elapsed_time = time.time() - self.start_time # Get the elapsed time
            self.output_text.delete('1.0', tk.END) # Clear the output text
            self.output_text.insert(tk.END, f"Processing... {elapsed_time:.2f} seconds") # Insert the output
            time.sleep(0.1)

    def simulate_llm_response(self, input_text):
        """" Simulates the response from the LLM """
        self.outputs = generate_text( # Generate the text based on the self parameters below
            self.model,
            self.tokenizer,
            self.inputs,
            self.num_return_sequences,
            self.no_repeat_ngram_size,
            self.max_length,
            self.top_k,
            self.top_p,
            self.temperature
        )
        generated_text = decode_output(self.tokenizer, self.outputs, self.skip_special_tokens) # Decode the output
        print(generated_text) # Print the output into the terminal
        return generated_text # Return the output


def intialize_model_mod():
    """ Initializes the model and tokenizer """
    # Check for CUDA availability and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # Set the device to GPU if available
    print(f"Using device: {device}") # Print the device being used

    # Load tokenizer and model
    print("loading tokenizer")
    model_directory = r"C:\Path\To\Your\Llama\Directory" # Set this to the directory where you have the model downloaded
    config = AutoConfig.from_pretrained(model_directory) # Load the config file
    tokenizer = AutoTokenizer.from_pretrained(model_directory) # Load the tokenizer
    print("loading model")
    model = AutoModelForCausalLM.from_pretrained(model_directory, config=config) # Load the model

    # Move model to the specified device (GPU or CPU)
    print("moving model to device", device)
    model.to(device) # Move the model to the specified device
    
    return tokenizer, model, device # Return the tokenizer, model, and device


def input_token_mod(tokenizer, prompt, device):
    """ Tokenizes the input and moves it to the specified device. """
    # Tokenize input text and move tokens to the same device as model
    print("creating input")
    inputs = tokenizer(prompt, return_tensors="pt").to(device) # Tokenize and move the input to the specified device
    return inputs # Return the input

def generate_text(model, tokenizer, inputs, num_return_sequences, no_repeat_ngram_size, max_length, top_k, top_p, temperature):
    """ Generates text using the model """
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
    """ Decodes the output and returns the generated text """
    # Decode and return generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=skip_special_tokens) # Decode the output

def main():
    """ Main function """
    root = tk.Tk() # Create the Tkinter window
    app = LLMGUI(root) # Create the LLMGUI object
    root.mainloop() # Start the Tkinter event loop


if __name__ == "__main__": # Check if the script is being run directly
    main() # Call the main function

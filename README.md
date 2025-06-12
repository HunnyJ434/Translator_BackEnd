# GPT-2 Fine-Tuning: Ojibwe to English Translation

This project fine-tunes a GPT-2 model using a custom dataset of Ojibwe words and their corresponding English translations. It demonstrates how to use Hugging Face’s `transformers` and `datasets` libraries to train a causal language model for low-resource language translation tasks.

## Project Description

The dataset consists of simple pairs where each line contains an Ojibwe word or phrase and its English translation, separated by a colon:

```
mino-bimaadiziwin: To live the good life  
nibi: water  
bimaadiziwin: life  
```

The training script:
- Downloads the dataset using `wget`
- Parses and preprocesses it into input/output pairs
- Converts it into a format suitable for GPT-2 training
- Fine-tunes GPT-2 on the dataset
- Saves the fine-tuned model and tokenizer for inference

## Repository Structure

```
.
├── train.py                # Main training script
├── inference_loop.py       # Interactive translator after training
├── data.txt               # Downloaded Ojibwe-English data (auto-generated)
├── ojibwe-gpt2/           # Saved fine-tuned model and tokenizer
└── README.md              # Project documentation
```

## Installation

Install the required Python libraries:

```bash
pip install pandas transformers datasets torch
```

## How to Train

Run the training script:

```bash
python3 train.py
```

This will:
1. Download `data.txt` from a GitHub-hosted text file
2. Parse the file into structured input/output pairs
3. Train a GPT-2 model using Hugging Face `Trainer`
4. Save the model to `./ojibwe-gpt2/`

## How to Translate (After Training)

Run the interactive loop to translate Ojibwe words:

```bash
python3 inference_loop.py
```

You will be prompted to enter Ojibwe words and receive translated English sentences.

## Use Cases

- Language preservation for Indigenous languages  
- Learning tools for Ojibwe speakers and students  
- Low-resource machine translation tasks  

## Technical Notes

- The script uses `GPT2LMHeadModel` and `Trainer` API  
- Padding tokens are masked with `-100` for loss calculation  
- The dataset is split 90/10 for training and evaluation  
- Trained on CPU by default (GPU support available)  

## Future Enhancements

- Add a web demo with Gradio or Streamlit  
- Expand the dataset with full Ojibwe sentences  
- Incorporate BLEU score for evaluation  
- Fine-tune larger language models like GPT-2 Medium or Neo  

## License

This project is licensed under the MIT License.

## Acknowledgments

Special thanks to those working to preserve and revitalize Indigenous languages, and to open-source contributors who make language tools accessible to all.

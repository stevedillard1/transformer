Welcome

To get started, create a virtual environment with uv. In the command line, run:
    1)    uv init
    2)    uv sync
Now your virtual environment should have the required packages for our project.

Next, navigate to "train_seinfeld_llm.ipynb" and run all. This will perform a training loop, and save a model called "notebook_model" to the folder "saved_models". Additionally, in the bottom of the notebook is our results section (labelled "Results" in the notebook). Here you should find everything that was asked for in the assignment description under the results section.

If you are satisfied with the training in the notebook, you may skip this step. Otherwise, you can run a training iteration with "uv run main.py". Simply run the file. You will be prompted for some hyperparameters, but once that's done, the training loop will commence, and a model will be saved under a user-specified name.

Once you have a model that is worth testing, use generate.py to give the trained llm prompts. In the command line, run

    uv run generate.py <model_name>

    where <model_name> is the name under which the model was saved. For example, the model generated from trained_seinfeld_llm is called notebook_model. So the command would be:
        uv run generate.py notebook_model



The rest of the repo is pretty self-explanatory:

llm.py - contains all classes/modules involved in the llm architecture
    classes:
        llm
        transformer_block
        mlp
        attention_head

tokenizer.py - contains class tokenizer that handles everything from formatting text to encoding tokens.

data_types.py - contains the config instance.


Finally, our writeup / contributions summary can be found in the google doc linked below:
https://docs.google.com/document/d/16r0_YbphrDryD3SAoVnmpLmerQkTB0dHfsqJkLbdubs
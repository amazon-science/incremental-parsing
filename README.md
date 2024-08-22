## Incremental Parser

This repository contains code to perform incremental parsing for Python for constrained generation of code by LLMs. 


## Installation

For exact reproducibility, we are using CUDA version 12.0, driver version 525.85.12, on a A100 GPU.
Python version 3.9.19, and the Rust toolchain version 1.79.0.
We use the version of Santacoder that was released on December 20, 2022.

1. Install everything in requirements.txt.
2. Run `maturin develop --release` in your virtual environment.
   - Note that IDEs sometimes have trouble with automatic code completion for this. 
     As long as you get the message `Installed incremental_parsing_rust-0.1.0`, the library is installed in the virtual environment.
3. Edit `scripts/constrained_generation_{random/nice}_cuts.sh` so that `results_path` is an absolute path.
   The program will create a (large) folder at this path.
   Also, edit the loop max, device name, and min/max data indices so that it fits your hardware and eventually loops
   through data indices 0 through 9999.
4. `PYTHONPATH=. scripts/constrained_generation_random_cuts.sh`
   - Read the documentation for `hapless` (`hap --help`) for information about process management
5. When done, edit the source path at the bottom of `incremental_parsing/evaluation/evaluate_stack.py` to match the
   results path.
   Edit the destination path to be somewhere you want a csv file to be created.
6. Import the csv file into a sqlite table named `stack`, and then use `incremental_parsing/evaluation/gen_tables.sql`
   to obtain the numbers from the paper.

You can also use the following interactive scripts in the `notebooks` directory:

- `create_parse_hierarchy_viz_python.ipynb` creates a parse hierarchy from left and right contexts, and outputs a
  visualization of this.
  Note that there might be multiple active branches with different parse hierarchies; the visualizer requires you to
  select one branch.
    - `create_parse_hierarchy_viz_calc_lang.ipynb` is the same for a much simpler language, a calculator language with
      tuples.
      It is significantly easier to inspect the output and understand what is going on here.
- `interactive_constrained_generation.ipynb` generates code, and shows all the left contexts which are considered to be
  a member of the quotient language, plus their scores from the LLM.
- `interactive_recognition.py` lets you type and see whether some text is in the quotient language, is incrementally
  parsable, or cannot be a prefix of a member of the quotient language.
- `paper_examples.ipynb` reproduces code generation examples.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.


## License

This project is licensed under the Apache-2.0 License.


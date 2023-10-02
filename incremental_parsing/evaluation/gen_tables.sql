-- INSTRUCTIONS
-- evaluate_stack.py takes the output folder of stack_completion_right_context.py and makes a nice CSV file out of it
-- Creating a database out of the results makes it easy to work with them instead of needing to write redundant code
-- I used the Pycharm SQL import tool to create a SQLite table named `stack`
-- Then, all the numbers in the paper come from the results of running these statements.

-- data_idx: Index in the-stack-smol-xl/data/python
-- cut_idx: Random seed used to chop up the file into prefix & suffix
-- seed_idx: If the LLM/sampling itself uses randomness, can put multiple seeds. We always used seed 0
-- parse_orig: 1 if the file at data_idx is valid Python 3
-- parse_earley: 1 if the file at data_idx is valid Python 3, and in our subset of Python
-- succesful_generation: 1 if there were no errors in generation.
--      If not equal to parse_earley, there is a bug in the program (look for a file named data_idx.cut_idx.seed_idx.err)
-- successful_clipped_generation: Did the algorithm identify a left context as a member of the quotient language
-- parse_[un]constrained: Does the Python parser like the result?
-- differs: Does the text context differ?
-- error_message: What error message does the Python parser give when reading the constrained output?
-- num_branches_after_first_suffix_lexeme: How many sub-languages created by the LCFL
--      Divide this by the number of sub-languages created for indentation
-- num_branches_in_middle: How many sub-languages after lexing the right context

-- How many experiments in all
SELECT COUNT(*)
FROM stack
WHERE parse_earley = 1;

-- How many parse with constrained generation
SELECT COUNT(*)
FROM stack
WHERE parse_constrained = 1
  AND parse_earley = 1;

-- How many parse with unconstrained generation
SELECT COUNT(*)
FROM stack
WHERE parse_unconstrained = 1
  AND parse_earley = 1;

-- How many parse with checked unconstrained generation
SELECT COUNT(*)
FROM stack
WHERE parse_unconstrained_checked = 1
  AND parse_earley = 1;

-- Constrained fails but unconstrained succeeds
SELECT COUNT(*)
FROM stack
WHERE parse_unconstrained = 1
  AND parse_constrained = 0
  AND parse_earley = 1;

-- Unconstrained fails but constrained succeeds
SELECT COUNT(*)
FROM stack
WHERE parse_unconstrained = 0
  AND parse_constrained = 1
  AND parse_earley = 1;

-- Constrained fails but checked unconstrained succeeds
SELECT COUNT(*)
FROM stack
WHERE parse_unconstrained_checked = 1
  AND parse_constrained = 0
  AND parse_earley = 1;

-- Checked unconstrained fails but constrained succeeds
SELECT COUNT(*)
FROM stack
WHERE parse_unconstrained_checked = 0
  AND parse_constrained = 1
  AND parse_earley = 1;

-- Parser incorrectly identified a left context
SELECT COUNT(*)
FROM stack
WHERE parse_constrained = 0
  AND successful_clipped_generation = 1
  AND parse_earley = 1;

-- Unconstrained generation sometimes succeeds in these constrained-generation failure modes
SELECT COUNT(*)
FROM stack
WHERE parse_constrained = 0
  AND successful_clipped_generation = 1
  AND parse_unconstrained = 1
  AND parse_earley = 1;

-- Parser never identified left context
SELECT COUNT(*)
FROM stack
WHERE parse_constrained = 0
  AND successful_clipped_generation = 0
  AND parse_earley = 1;

-- Unconstrained generation rarely/never succeeds in this failure mode
SELECT COUNT(*)
FROM stack
WHERE parse_constrained = 0
  AND successful_clipped_generation = 0
  AND parse_unconstrained = 1
  AND parse_earley = 1;

-- What error messages are given show up
SELECT error_message, COUNT(*) as c
FROM stack
WHERE error_message NOT NULL
GROUP BY error_message
ORDER by c DESC;

-- How many branches are due to LCFL (divide by 2 because of extra indentation branches)
SELECT stack.num_branches_after_first_suffix_lexeme / 2, COUNT(*)
FROM stack
WHERE parse_earley = 1
GROUP BY stack.num_branches_after_first_suffix_lexeme;

-- How many branches due to LCFL + Indentation
SELECT stack.num_branches_in_middle, COUNT(*)
FROM stack
WHERE parse_earley = 1
GROUP BY stack.num_branches_in_middle;


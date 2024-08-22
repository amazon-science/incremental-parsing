-- INSTRUCTIONS
-- evaluate_stack.py takes the output folder of stack_completion_right_context.py and makes a nice CSV file out of it
-- Creating a database out of the results makes it easy to work with them instead of needing to write redundant code
-- I use the Pycharm SQL import tool to create a SQLite table named `stack`
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

-- Table body
SELECT COUNT(*), parse_constrained, parse_unconstrained, parse_unconstrained_checked
FROM stack
WHERE parse_earley = 1
GROUP BY parse_constrained, parse_unconstrained, parse_unconstrained_checked;

-- Bottom margin
SELECT COUNT(*), parse_constrained
FROM stack
WHERE parse_earley = 1
GROUP BY parse_constrained;

-- Side margin
SELECT COUNT(*), parse_unconstrained, parse_unconstrained_checked
FROM stack
WHERE parse_earley = 1
GROUP BY  parse_unconstrained, parse_unconstrained_checked;

-- Breakdown of failure cases
SELECT COUNT(*), parse_unconstrained, parse_unconstrained_checked, successful_clipped_generation
FROM stack
WHERE parse_earley = 1 and parse_constrained = 0 and successful_generation = 1
GROUP BY stack.parse_unconstrained, parse_unconstrained_checked, successful_clipped_generation;

SELECT median(constrained_overhead_mean), median(constrained_overhead_eval_mean), median(unconstrained_checking_overhead_mean)
FROM stack WHERE parse_unconstrained_checked = 1 and parse_constrained = 1;

-- What error messages are given show up
SELECT error_message, COUNT(*) as c
FROM stack
WHERE error_message NOT NULL
GROUP BY error_message
ORDER by c DESC;

-- Specific cases with error messages for debugging
SELECT stack.data_idx, stack.cut_idx, stack.error_message
from stack
where error_message NOT null;

-- How many branches are due to LCFL (divide by 2 because of extra indentation branches)
SELECT stack.num_branches_after_first_suffix_lexeme / 2, COUNT(*)
FROM stack
WHERE parse_earley = 1
GROUP BY stack.num_branches_after_first_suffix_lexeme;

SELECT AVG(stack.num_branches_after_first_suffix_lexeme / 2), median(stack.num_branches_after_first_suffix_lexeme / 2), stdev(stack.num_branches_after_first_suffix_lexeme/2)
FROM stack
WHERE parse_earley = 1;

-- How many branches due to LCFL + Indentation
SELECT stack.num_branches_in_middle, COUNT(*)
FROM stack
WHERE parse_earley = 1
GROUP BY stack.num_branches_in_middle;


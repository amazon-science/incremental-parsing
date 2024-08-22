use crate::bridge::bnf::NativeGrammar;
use crate::grammar::bnf::TerminalIdx;
use crate::parser::earley::EarleyStateCreationMethod::TopLevel;
use crate::parser::earley::{get_all_initial_earley_states, EarleyChartIdx, EarleyCollection};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{Bound, PyResult};
use std::collections::HashSet;

pub fn extension_charts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeEarleyCharts>()?;
    Ok(())
}

#[pyclass]
pub struct NativeEarleyCharts {
    earley_collection: EarleyCollection,
}

#[pymethods]
impl NativeEarleyCharts {
    #[staticmethod]
    pub fn create_initial_earley_charts<'a>(
        python: Python<'a>,
        grammar: &NativeGrammar,
    ) -> Py<PyTuple> {
        let mut collection = EarleyCollection::new();
        let root_chart = collection.add_chart();
        for state in get_all_initial_earley_states(&grammar.grammar, root_chart) {
            collection.add_state(&grammar.grammar, root_chart, state, TopLevel);
        }

        let charts = NativeEarleyCharts {
            earley_collection: collection,
        };
        let this_chart = &charts.earley_collection[root_chart];

        let allowed_terminals: HashSet<TerminalIdx> =
            this_chart.allowed_terminals(&grammar.grammar);
        // Terminals are already deduped, so okay to just use vec
        let allowed_terminal_names: Vec<&String> = allowed_terminals
            .into_iter()
            .map(|terminal| grammar.names.terminal(terminal))
            .collect();
        let completeable = this_chart.is_completable();
        let complete = this_chart.is_complete(&grammar.grammar);

        let result_tup = (
            charts,
            root_chart.0,
            allowed_terminal_names,
            completeable,
            complete,
        );

        result_tup.into_py(python)
    }

    pub fn parse<'a>(
        &mut self,
        python: Python<'a>,
        grammar: &NativeGrammar,
        chart_idx: usize,
        token_name: String,
    ) -> Py<PyTuple> {
        let src_chart_idx = EarleyChartIdx(chart_idx);

        let this_chart_idx = self.earley_collection.add_chart();
        let terminal_idx = grammar.names.terminal_idx(&token_name);
        if let Some(terminal_idx) = terminal_idx {
            // If terminal isn't in the grammar, adding a link won't do anything anyway
            self.earley_collection.add_link(
                &grammar.grammar,
                terminal_idx,
                src_chart_idx,
                this_chart_idx,
            );
        }

        let this_chart = &self.earley_collection[this_chart_idx];

        let allowed_terminals: HashSet<TerminalIdx> =
            this_chart.allowed_terminals(&grammar.grammar);
        let allowed_terminal_names: Vec<&String> = allowed_terminals
            .into_iter()
            .map(|terminal| grammar.names.terminal(terminal))
            .collect();
        let completable = this_chart.is_completable();
        let complete = this_chart.is_complete(&grammar.grammar);

        let result_tup = (
            this_chart_idx.0,
            allowed_terminal_names,
            completable,
            complete,
        );
        result_tup.into_py(python)
    }

    pub fn get_chart_len(&self, chart_idx: usize) -> usize {
        self.earley_collection[EarleyChartIdx(chart_idx)]
            .states
            .len()
    }

    pub fn get_earley_state(
        &self,
        python: Python<'_>,
        native_grammar: &NativeGrammar,
        chart_idx: usize,
        state_idx: usize,
    ) -> Py<PyTuple> {
        let (state, creation_methods) =
            &self.earley_collection[EarleyChartIdx(chart_idx)].states[state_idx];
        let nonterminal_name = native_grammar.names.nonterminal(state.nonterminal);

        let mut creation_methods_serialized = vec![];
        for creation_method in creation_methods {
            creation_method.serialize(&mut creation_methods_serialized);
        }

        let result_tup = (
            state.span_start.0,
            nonterminal_name,
            state.production_index,
            state.dot_index,
            state.production_length,
            creation_methods_serialized,
        );
        result_tup.into_py(python)
    }

    #[staticmethod]
    pub fn create_earley_nfa<'a>(
        grammar: &NativeGrammar,
        num_charts: usize,
        start_charts: Vec<usize>,
        transitions: Vec<(usize, usize, String)>,
    ) -> NativeEarleyCharts {
        let mut collection = EarleyCollection::new();

        for _ in 0..num_charts {
            collection.add_chart();
        }

        for start_chart in start_charts {
            for state in
                get_all_initial_earley_states(&grammar.grammar, EarleyChartIdx(start_chart))
            {
                collection.add_state(
                    &grammar.grammar,
                    EarleyChartIdx(start_chart),
                    state,
                    TopLevel,
                );
            }
        }

        for (origin_chart, dest_chart, terminal) in transitions {
            if let Some(terminal_idx) = grammar.names.terminal_idx(&terminal) {
                collection.add_link(
                    &grammar.grammar,
                    terminal_idx,
                    EarleyChartIdx(origin_chart),
                    EarleyChartIdx(dest_chart),
                );
            }
        }

        NativeEarleyCharts {
            earley_collection: collection,
        }
    }
}

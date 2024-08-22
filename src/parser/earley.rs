use crate::grammar::bnf::{NonterminalIdx, TerminalIdx, BNF};
use crate::grammar::names::BNFNames;
use crate::parser::earley::EarleyStateNextStep::ScanTerminal;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Display;
use std::hash::Hash;
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EarleyChartIdx(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EarleyStateIdx(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum EarleyStateNextStep {
    ScanTerminal(TerminalIdx),
    PredictNonterminal(NonterminalIdx),
    Complete(NonterminalIdx, EarleyChartIdx),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EarleyState {
    pub span_start: EarleyChartIdx,
    pub nonterminal: NonterminalIdx,
    pub production_index: usize,
    pub dot_index: usize,
    pub production_length: usize,
}

impl EarleyState {
    fn is_complete(&self) -> bool {
        self.dot_index == self.production_length
    }

    fn advance(&self) -> Self {
        Self {
            span_start: self.span_start,
            nonterminal: self.nonterminal,
            production_index: self.production_index,
            dot_index: self.dot_index + 1,
            production_length: self.production_length,
        }
    }

    fn get_next_step(&self, grammar: &BNF) -> EarleyStateNextStep {
        if self.is_complete() {
            return EarleyStateNextStep::Complete(self.nonterminal, self.span_start);
        }
        let rule = &grammar[self.nonterminal];
        let production = &rule.productions[self.production_index];
        let element = &production.elements[self.dot_index];
        match element {
            crate::grammar::bnf::BNFElement::BNFTerminal(t) => {
                EarleyStateNextStep::ScanTerminal(*t)
            }
            crate::grammar::bnf::BNFElement::BNFNonterminal(nt) => {
                EarleyStateNextStep::PredictNonterminal(*nt)
            }
        }
    }

    fn to_str<T: Hash + Eq + Clone + Display, NT: Hash + Eq + Clone + Display>(
        &self,
        grammar: &BNF,
        names: &BNFNames<T, NT>,
    ) -> String {
        let mut result = String::new();

        result.push_str(&names.nonterminal(self.nonterminal).to_string());
        result.push_str(" -> ");

        let rule = &grammar[self.nonterminal];
        let production = &rule.productions[self.production_index];

        let mut elem_strs = production
            .elements
            .iter()
            .map(|e| match e {
                crate::grammar::bnf::BNFElement::BNFTerminal(t) => names.terminal(*t).to_string(),
                crate::grammar::bnf::BNFElement::BNFNonterminal(nt) => {
                    names.nonterminal(*nt).to_string()
                }
            })
            .collect::<Vec<String>>();
        elem_strs.insert(self.dot_index, String::from("•"));

        result.push_str(&elem_strs.join(" "));
        result.push_str(&format!(" ({})", self.span_start.0));
        result
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EarleyStateCreationMethod {
    TopLevel,
    Scanned {
        from_chart: EarleyChartIdx,
        from_state: EarleyStateIdx,
    },
    Predicted {
        from_chart: EarleyChartIdx,
        from_state: EarleyStateIdx,
    },
    NullablePredictCompleted {
        from_chart: EarleyChartIdx,
        from_state: EarleyStateIdx,
    },
    Completed {
        finished_rule_chart: EarleyChartIdx,
        finished_rule_state: EarleyStateIdx,
        complete_into_chart: EarleyChartIdx,
        complete_into_state: EarleyStateIdx,
    },
}

impl EarleyStateCreationMethod {
    pub fn serialize(&self, result: &mut Vec<usize>) {
        match self {
            EarleyStateCreationMethod::TopLevel => result.push(0),
            EarleyStateCreationMethod::Scanned {
                from_chart,
                from_state,
            } => result.extend(&[1, from_chart.0, from_state.0]),
            EarleyStateCreationMethod::Predicted {
                from_chart,
                from_state,
            } => result.extend(&[2, from_chart.0, from_state.0]),
            EarleyStateCreationMethod::NullablePredictCompleted {
                from_chart,
                from_state,
            } => result.extend(&[3, from_chart.0, from_state.0]),
            EarleyStateCreationMethod::Completed {
                finished_rule_chart,
                finished_rule_state,
                complete_into_chart,
                complete_into_state,
            } => result.extend(&[
                4,
                finished_rule_chart.0,
                finished_rule_state.0,
                complete_into_chart.0,
                complete_into_state.0,
            ]),
        }
    }
}

pub struct EarleyChart {
    pub index: EarleyChartIdx,
    pub states: Vec<(EarleyState, Vec<EarleyStateCreationMethod>)>,
    pub state_mapping: HashMap<EarleyState, EarleyStateIdx>,
    links: Vec<(TerminalIdx, EarleyChartIdx)>,
    interested_completers: Vec<(NonterminalIdx, EarleyChartIdx, EarleyStateIdx)>,
}

type StateAddInfo = (EarleyChartIdx, EarleyState, EarleyStateCreationMethod);
type NewCompletorInfo = (NonterminalIdx, EarleyChartIdx);

impl EarleyChart {
    fn new(index: EarleyChartIdx) -> Self {
        Self {
            index,
            states: Vec::new(),
            state_mapping: HashMap::new(),
            links: Vec::new(),
            interested_completers: Vec::new(),
        }
    }

    fn add_state(
        &mut self,
        state: EarleyState,
        creation_method: EarleyStateCreationMethod,
    ) -> (EarleyStateIdx, bool) {
        if let Some(idx) = self.state_mapping.get(&state) {
            self.states[idx.0].1.push(creation_method);
            (*idx, false)
        } else {
            let idx = EarleyStateIdx(self.states.len());
            self.states.push((state, vec![creation_method]));
            self.state_mapping.insert(state, idx);
            (idx, true)
        }
    }

    fn scanner_predictor_completer(
        &self,
        grammar: &BNF,
        state_idx: EarleyStateIdx,
        earley_collection: &EarleyCollection,
    ) -> (Vec<StateAddInfo>, Option<NewCompletorInfo>) {
        let state = &self.states[state_idx.0].0;
        let mut new_interested_completer = None;
        let mut state_add_info = vec![];

        match state.get_next_step(grammar) {
            EarleyStateNextStep::ScanTerminal(t) => {
                for (terminal, dest_chart) in self.links.iter() {
                    if *terminal == t {
                        state_add_info.push((
                            *dest_chart,
                            state.advance(),
                            EarleyStateCreationMethod::Scanned {
                                from_chart: self.index,
                                from_state: state_idx,
                            },
                        ));
                    }
                }
            }
            EarleyStateNextStep::PredictNonterminal(nt) => {
                for (prod_index, production) in grammar[nt].productions.iter().enumerate() {
                    let new_state = EarleyState {
                        span_start: self.index,
                        nonterminal: nt,
                        production_index: prod_index,
                        dot_index: 0,
                        production_length: production.elements.len(),
                    };
                    state_add_info.push((
                        self.index,
                        new_state,
                        EarleyStateCreationMethod::Predicted {
                            from_chart: self.index,
                            from_state: state_idx,
                        },
                    ));
                }

                // if nt is nullable, add option to advance it anyway
                if grammar.nullable_rules().contains(&nt) {
                    state_add_info.push((
                        self.index,
                        state.advance(),
                        EarleyStateCreationMethod::NullablePredictCompleted {
                            from_chart: self.index,
                            from_state: state_idx,
                        },
                    ))
                }

                // Some future state might be able to complete this rule
                for (complete_nt, finished_rule_chart_idx, finished_rule_state_idx) in
                    self.interested_completers.iter()
                {
                    if *complete_nt == nt {
                        state_add_info.push((
                            *finished_rule_chart_idx,
                            state.advance(),
                            EarleyStateCreationMethod::Completed {
                                finished_rule_chart: *finished_rule_chart_idx,
                                finished_rule_state: *finished_rule_state_idx,
                                complete_into_chart: self.index,
                                complete_into_state: state_idx,
                            },
                        ))
                    }
                }
            }
            EarleyStateNextStep::Complete(nt, span_start) => {
                // Register this completer in case a state is added there in the future
                new_interested_completer = Some((nt, span_start));

                for (idx_of_complete_into, (state_complete_into, _)) in
                    earley_collection[span_start].states.iter().enumerate()
                {
                    if state_complete_into.get_next_step(grammar)
                        == EarleyStateNextStep::PredictNonterminal(nt)
                    {
                        state_add_info.push((
                            self.index,
                            state_complete_into.advance(),
                            EarleyStateCreationMethod::Completed {
                                finished_rule_chart: self.index,
                                finished_rule_state: state_idx,
                                complete_into_chart: span_start,
                                complete_into_state: EarleyStateIdx(idx_of_complete_into),
                            },
                        ));
                    }
                }
            }
        };
        (state_add_info, new_interested_completer)
    }

    fn add_interested_completer(
        &mut self,
        nonterminal: NonterminalIdx,
        dest_chart: EarleyChartIdx,
        dest_state: EarleyStateIdx,
    ) {
        if self
            .interested_completers
            .contains(&(nonterminal, dest_chart, dest_state))
        {
            return;
        }
        self.interested_completers
            .push((nonterminal, dest_chart, dest_state));
    }

    fn add_link(
        &mut self,
        grammar: &BNF,
        dest_chart: EarleyChartIdx,
        terminal: TerminalIdx,
    ) -> Vec<StateAddInfo> {
        self.links.push((terminal, dest_chart));

        let mut state_add_info = vec![];

        // Run the scanner
        for (state_idx, (state, _)) in self.states.iter().enumerate() {
            if ScanTerminal(terminal) == state.get_next_step(&grammar) {
                state_add_info.push((
                    dest_chart,
                    state.advance(),
                    EarleyStateCreationMethod::Scanned {
                        from_chart: self.index,
                        from_state: EarleyStateIdx(state_idx),
                    },
                ));
            }
        }

        state_add_info
    }

    pub fn allowed_terminals(&self, grammar: &BNF) -> HashSet<TerminalIdx> {
        let mut allowed_terms = HashSet::new();
        for (state, _) in self.states.iter() {
            if let ScanTerminal(terminal) = state.get_next_step(grammar) {
                allowed_terms.insert(terminal);
            }
        }
        allowed_terms
    }

    pub fn is_completable(&self) -> bool {
        !self.states.is_empty()
    }

    pub fn is_complete(&self, grammar: &BNF) -> bool {
        for (state, _) in self.states.iter() {
            if state.is_complete() && grammar.top_level_rules().contains(&state.nonterminal) {
                return true;
            }
        }
        false
    }
}

pub struct EarleyCollection {
    charts: Vec<EarleyChart>,
    // Don't recreate queue every time we call add_state
    process_queue: VecDeque<(EarleyChartIdx, EarleyState, EarleyStateCreationMethod)>,
}

impl EarleyCollection {
    pub fn new() -> Self {
        Self {
            charts: Vec::new(),
            process_queue: VecDeque::new(),
        }
    }

    pub fn add_chart(&mut self) -> EarleyChartIdx {
        let idx = EarleyChartIdx(self.charts.len());
        self.charts.push(EarleyChart::new(idx));
        idx
    }

    pub fn add_link(
        &mut self,
        grammar: &BNF,
        terminal: TerminalIdx,
        origin_chart: EarleyChartIdx,
        dest_chart: EarleyChartIdx,
    ) {
        let state_add_infos = self[origin_chart].add_link(grammar, dest_chart, terminal);
        for (chart_idx, state, method) in state_add_infos {
            self.add_state(grammar, chart_idx, state, method);
        }
    }
    // TODO Need to add basic tests

    pub fn add_state(
        &mut self,
        grammar: &BNF,
        chart_idx: EarleyChartIdx,
        state: EarleyState,
        creation_method: EarleyStateCreationMethod,
    ) {
        self.process_queue
            .push_back((chart_idx, state, creation_method));

        while let Some((chart_idx, state, creation_method)) = self.process_queue.pop_front() {
            let (state_idx, is_new_state) = self[chart_idx].add_state(state, creation_method);
            if is_new_state {
                let (state_add_info, new_completor) =
                    self[chart_idx].scanner_predictor_completer(grammar, state_idx, self);
                if let Some((nt, span_start)) = new_completor {
                    self[span_start].add_interested_completer(nt, chart_idx, state_idx);
                }
                self.process_queue.extend(state_add_info);
            }
        }
    }
}

impl Index<EarleyChartIdx> for EarleyCollection {
    type Output = EarleyChart;

    fn index(&self, index: EarleyChartIdx) -> &Self::Output {
        &self.charts[index.0]
    }
}

impl IndexMut<EarleyChartIdx> for EarleyCollection {
    fn index_mut(&mut self, index: EarleyChartIdx) -> &mut Self::Output {
        &mut self.charts[index.0]
    }
}

pub fn get_all_initial_earley_states<'a>(
    grammar: &'a BNF,
    initial_chart_idx: EarleyChartIdx,
) -> impl Iterator<Item = EarleyState> + 'a {
    grammar.top_level_rules().iter().flat_map(move |tlrule| {
        grammar[*tlrule]
            .productions
            .iter()
            .enumerate()
            .map(move |(prod_idx, production)| EarleyState {
                span_start: initial_chart_idx,
                dot_index: 0,
                production_length: production.elements.len(),
                nonterminal: *tlrule,
                production_index: prod_idx,
            })
    })
}

use crate::grammar::bnf::{BNFElement, BNFProduction, BNFRule, NonterminalIdx, TerminalIdx, BNF};
use crate::grammar::names::BNFNames;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

pub fn extension_bnf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BridgeBNFElement>()?;
    m.add_class::<BridgeBNFProduction>()?;
    m.add_class::<BridgeBNFRule>()?;
    m.add_class::<BridgeBNFGrammar>()?;
    m.add_class::<NativeGrammar>()?;
    Ok(())
}

#[pyclass]
#[derive(Clone, Debug)]
pub enum BridgeBNFElement {
    Terminal { name: String },
    Nonterminal { name: String },
}

#[pymethods]
impl BridgeBNFElement {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

impl BridgeBNFElement {
    pub fn collect_terminals(&self, terminals: &mut HashSet<String>) {
        match self {
            BridgeBNFElement::Terminal { name } => {
                terminals.insert(name.clone());
            }
            BridgeBNFElement::Nonterminal { name: _ } => {}
        }
    }

    pub fn to_native(
        &self,
        terminal_map: &HashMap<String, TerminalIdx>,
        nonterminal_map: &HashMap<String, NonterminalIdx>,
    ) -> BNFElement {
        match self {
            BridgeBNFElement::Terminal { name } => BNFElement::BNFTerminal(terminal_map[name]),
            BridgeBNFElement::Nonterminal { name } => {
                BNFElement::BNFNonterminal(nonterminal_map[name])
            }
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct BridgeBNFProduction {
    #[pyo3(get, set)]
    pub elements: Vec<BridgeBNFElement>,
}

#[pymethods]
impl BridgeBNFProduction {
    #[new]
    pub fn new(elements: Vec<BridgeBNFElement>) -> Self {
        Self { elements }
    }

    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

impl BridgeBNFProduction {
    pub fn collect_terminals(&self, terminals: &mut HashSet<String>) {
        for element in &self.elements {
            element.collect_terminals(terminals);
        }
    }

    pub fn to_native(
        &self,
        terminal_map: &HashMap<String, TerminalIdx>,
        nonterminal_map: &HashMap<String, NonterminalIdx>,
    ) -> BNFProduction {
        BNFProduction {
            elements: self
                .elements
                .iter()
                .map(|element| element.to_native(terminal_map, nonterminal_map))
                .collect(),
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct BridgeBNFRule {
    #[pyo3(get, set)]
    productions: Vec<BridgeBNFProduction>,
}

#[pymethods]
impl BridgeBNFRule {
    #[new]
    pub fn new(productions: Vec<BridgeBNFProduction>) -> Self {
        Self { productions }
    }

    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

impl BridgeBNFRule {
    pub fn collect_terminals(&self, terminals: &mut HashSet<String>) {
        for production in &self.productions {
            production.collect_terminals(terminals);
        }
    }

    pub fn to_native(&self, names: &BNFNames<String, String>) -> BNFRule {
        BNFRule {
            productions: self
                .productions
                .iter()
                .map(|production| {
                    production.to_native(names.terminal_map(), names.nonterminal_map())
                })
                .collect(),
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct BridgeBNFGrammar {
    #[pyo3(get, set)]
    pub rules: HashMap<String, BridgeBNFRule>,

    #[pyo3(get, set)]
    pub top_level_rules: Vec<String>,
}

#[pymethods]
impl BridgeBNFGrammar {
    #[new]
    pub fn new(rules: HashMap<String, BridgeBNFRule>, top_level_rules: Vec<String>) -> Self {
        Self {
            rules,
            top_level_rules,
        }
    }

    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    pub fn to_native(&self) -> NativeGrammar {
        self.clone().into()
    }
}

impl BridgeBNFGrammar {
    pub fn collect_terminals_nonterminals(&self) -> (Vec<String>, Vec<String>) {
        let mut terminals: HashSet<String> = HashSet::new();

        for rule in self.rules.values() {
            rule.collect_terminals(&mut terminals);
        }

        let nonterminals: Vec<String> = self.rules.keys().cloned().collect();

        (terminals.into_iter().collect(), nonterminals)
    }
}

#[pyclass(frozen)]
#[derive(Clone)]
pub struct NativeGrammar {
    pub grammar: BNF,
    pub names: BNFNames<String, String>,
}

#[pymethods]
impl NativeGrammar {
    fn __repr__(&self) -> String {
        "<Native BNF>".to_string()
    }
}

impl From<BridgeBNFGrammar> for NativeGrammar {
    fn from(bnf: BridgeBNFGrammar) -> Self {
        let (terminals, nonterminals) = bnf.collect_terminals_nonterminals();
        let names = BNFNames::new(terminals, nonterminals);

        let rules = names
            .nonterminals()
            .iter()
            .map(|ntname| bnf.rules[ntname].to_native(&names))
            .collect();

        let top_level_rules = bnf
            .top_level_rules
            .iter()
            .map(|ntname| names.nonterminal_idx(ntname))
            .collect();

        NativeGrammar {
            grammar: BNF::new(rules, top_level_rules),
            names,
        }
    }
}

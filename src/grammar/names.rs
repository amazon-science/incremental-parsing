use crate::grammar::bnf::{NonterminalIdx, TerminalIdx};
use std::collections::HashMap;
use std::fmt::Display;
use std::hash::Hash;

#[derive(Debug, Clone)]
pub struct BNFNames<T: Hash + Eq + Display, NT: Hash + Eq + Display> {
    terminals: Vec<T>,
    nonterminals: Vec<NT>,
    terminal_map: HashMap<T, TerminalIdx>,
    nonterminal_map: HashMap<NT, NonterminalIdx>,
}

impl<T: Hash + Eq + Clone + Display, NT: Hash + Eq + Clone + Display> BNFNames<T, NT> {
    pub fn new(terminals: Vec<T>, nonterminals: Vec<NT>) -> Self {
        let terminal_map = terminals
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, t)| (t, TerminalIdx(i)))
            .collect();
        let nonterminal_map = nonterminals
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, t)| (t, NonterminalIdx(i)))
            .collect();

        Self {
            terminals,
            nonterminals,
            terminal_map,
            nonterminal_map,
        }
    }

    pub fn terminal(&self, idx: TerminalIdx) -> &T {
        &self.terminals[idx.0]
    }

    pub fn nonterminal(&self, idx: NonterminalIdx) -> &NT {
        &self.nonterminals[idx.0]
    }

    pub fn terminals(&self) -> &Vec<T> {
        &self.terminals
    }

    pub fn nonterminals(&self) -> &Vec<NT> {
        &self.nonterminals
    }

    pub fn terminal_idx(&self, t: &T) -> Option<TerminalIdx> {
        self.terminal_map.get(t).cloned()
    }

    pub fn nonterminal_idx(&self, t: &NT) -> NonterminalIdx {
        self.nonterminal_map[t]
    }

    pub fn terminal_map(&self) -> &HashMap<T, TerminalIdx> {
        &self.terminal_map
    }

    pub fn nonterminal_map(&self) -> &HashMap<NT, NonterminalIdx> {
        &self.nonterminal_map
    }
}

enum NonterminalName {
    Simple(String),
    Prefix(usize, Box<NonterminalName>),
    Suffix(usize, Box<NonterminalName>),
    PrefixSuffix(usize, Box<NonterminalName>, usize),
}

impl Display for NonterminalName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NonterminalName::Simple(name) => write!(f, "{}", name),
            NonterminalName::Prefix(prefix, name) => write!(f, "{}<{}->", name, prefix),
            NonterminalName::Suffix(suffix, name) => write!(f, "{}<-{}>", name, suffix),
            NonterminalName::PrefixSuffix(prefix, name, suffix) => {
                write!(f, "{}<{}-{}>", name, prefix, suffix)
            }
        }
    }
}

use crate::grammar::names::BNFNames;
use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::hash::Hash;
use std::ops::Index;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct TerminalIdx(pub usize);
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct NonterminalIdx(pub usize);

#[derive(Hash, Eq, Copy, Clone, PartialEq)]
pub enum BNFElement {
    BNFTerminal(TerminalIdx),
    BNFNonterminal(NonterminalIdx),
}

impl BNFElement {
    pub fn to_string<T: Hash + Eq + Clone + Display, NT: Hash + Eq + Clone + Display>(
        &self,
        names: &BNFNames<T, NT>,
    ) -> String {
        match self {
            BNFElement::BNFTerminal(t) => names.terminal(*t).to_string(),
            BNFElement::BNFNonterminal(nt) => names.nonterminal(*nt).to_string(),
        }
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct BNFProduction {
    pub elements: Vec<BNFElement>,
}

impl BNFProduction {
    pub fn reverse(&self) -> Self {
        BNFProduction {
            elements: self.elements.iter().cloned().rev().collect(),
        }
    }

    pub fn to_string<T: Hash + Eq + Clone + Display, NT: Hash + Eq + Clone + Display>(
        &self,
        names: &BNFNames<T, NT>,
    ) -> String {
        self.elements
            .iter()
            .map(|e| e.to_string(names))
            .collect::<Vec<String>>()
            .join(" ")
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct BNFRule {
    pub productions: Vec<BNFProduction>,
}

impl BNFRule {
    pub fn reverse(&self) -> Self {
        BNFRule {
            productions: self.productions.iter().map(|p| p.reverse()).collect(),
        }
    }
}

#[derive(Clone)]
pub struct BNF {
    rules: Vec<BNFRule>,
    nullable_rules: HashSet<NonterminalIdx>,
    top_level_rules: Vec<NonterminalIdx>,
}

fn get_nullable_rules(rules: &Vec<BNFRule>) -> HashSet<NonterminalIdx> {
    let mut reverse_rule_mapping = HashMap::new();
    for (key, value) in rules.iter().enumerate() {
        for production in value.productions.iter() {
            for element in production.elements.iter() {
                match element {
                    BNFElement::BNFNonterminal(nt) => {
                        reverse_rule_mapping
                            .entry(*nt)
                            .or_insert(Vec::new())
                            .push(NonterminalIdx(key));
                    }
                    BNFElement::BNFTerminal(_) => {}
                }
            }
        }
    }

    let mut nullable_rules = HashSet::new();
    let mut queue = Vec::new();

    for (key, rule) in rules.iter().enumerate() {
        if rule.productions.iter().any(|p| p.elements.is_empty()) {
            let key = NonterminalIdx(key);
            nullable_rules.insert(key);
            queue.push(key);
        }
    }

    while let Some(rule_key) = queue.pop() {
        if let Some(referencer_rules) = reverse_rule_mapping.get(&rule_key) {
            for referencer_rule in referencer_rules.iter() {
                if !nullable_rules.contains(referencer_rule) {
                    for referencer_rule_production in rules[referencer_rule.0].productions.iter() {
                        if referencer_rule_production
                            .elements
                            .iter()
                            .all(|element| match element {
                                BNFElement::BNFNonterminal(ntidx) => nullable_rules.contains(ntidx),
                                _ => false,
                            })
                        {
                            nullable_rules.insert(*referencer_rule);
                            queue.push(*referencer_rule);
                        }
                    }
                }
            }
        }
    }

    nullable_rules
}

impl BNF {
    pub fn new(rules: Vec<BNFRule>, top_level_rules: Vec<NonterminalIdx>) -> Self {
        let nullable_rules = get_nullable_rules(&rules);

        BNF {
            rules,
            nullable_rules,
            top_level_rules,
        }
    }

    pub fn nullable_rules(&self) -> &HashSet<NonterminalIdx> {
        &self.nullable_rules
    }

    pub fn top_level_rules(&self) -> &Vec<NonterminalIdx> {
        &self.top_level_rules
    }
}

impl Index<NonterminalIdx> for BNF {
    type Output = BNFRule;

    fn index(&self, index: NonterminalIdx) -> &Self::Output {
        &self.rules[index.0]
    }
}

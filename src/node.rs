use rand::Rng;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

static OBJECT_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub type Map = HashMap<usize, [Option<usize>; 2]>;

#[derive(Debug, Clone)]
enum Operator {
    Plus,
    Mul,
    Pow,
    Relu,
}

#[derive(Debug, Clone)]
pub struct Node {
    id: usize,
    value: f64,
    pub gradient: f64,
    operator: Option<Operator>,
}

impl Node {
    fn new(value: f64, operator: Option<Operator>) -> Self {
        Node {
            id: OBJECT_COUNTER.fetch_add(1, Ordering::SeqCst),
            value: value,
            gradient: 0.0,
            operator: operator,
        }
    }
}

pub fn new_node(nodes: &mut Vec<Node>, value: f64) -> usize {
    nodes.push(Node::new(value, None));
    let node_id = nodes.last_mut().unwrap().id;
    node_id
}

fn append_node(nodes: &mut Vec<Node>, value: f64, operator: Option<Operator>) -> usize {
    nodes.push(Node::new(value, operator));
    let node_id = nodes.last_mut().unwrap().id;
    node_id
}

pub fn add(
    map: &mut Map,
    nodes: &mut Vec<Node>,
    index_self: usize,
    index_other: usize,
) -> (usize, f64) {
    let a = nodes.get(index_self).unwrap();
    let b = nodes.get(index_other).unwrap();

    let value = a.value + b.value;
    let node_id = append_node(nodes, value, Some(Operator::Plus));
    map.insert(node_id, [Some(index_self), Some(index_other)]);

    (node_id, value)
}

fn mul(
    map: &mut Map,
    nodes: &mut Vec<Node>,
    index_self: usize,
    index_other: usize,
) -> (usize, f64) {
    let a = nodes.get(index_self).unwrap();
    let b = nodes.get(index_other).unwrap();

    let value = a.value * b.value;
    let node_id = append_node(nodes, value, Some(Operator::Mul));

    map.insert(node_id, [Some(index_self), Some(index_other)]);

    (node_id, value)
}

fn pow(
    map: &mut Map,
    nodes: &mut Vec<Node>,
    index_self: usize,
    index_other: usize,
) -> (usize, f64) {
    let a = nodes.get(index_self).unwrap();
    let b = nodes.get(index_other).unwrap();

    let value = a.value.powf(b.value);
    let node_id = append_node(nodes, value, Some(Operator::Pow));

    map.insert(node_id, [Some(index_self), Some(index_other)]);

    (node_id, value)
}

fn relu(map: &mut Map, nodes: &mut Vec<Node>, index_self: usize) -> (usize, f64) {
    let a = nodes.get(index_self).unwrap();

    let mut value = 0.0;
    if a.value > 0.0 {
        value = a.value
    }

    let node_id = append_node(nodes, value, Some(Operator::Relu));

    map.insert(node_id, [Some(index_self), None]);

    (node_id, value)
}

pub struct Neuron {
    weights: Vec<usize>,
    bias: usize,
}

impl Neuron {
    fn new(nodes: &mut Vec<Node>, count_in: u64) -> Self {
        let mut rng = rand::thread_rng();

        let mut weights = Vec::new();
        for i in 0..count_in {
            weights.push(new_node(nodes, rng.gen_range(-1.0..1.0)))
        }

        let bias = new_node(nodes, rng.gen_range(-1.0..1.0));

        Neuron {
            weights: weights,
            bias: bias,
        }
    }

    fn connect(&self, map: &mut Map, x: &Vec<usize>, nodes: &mut Vec<Node>) -> usize {
        assert!(self.weights.len() == x.len());

        let (last_index, _) = self
            .weights
            .iter()
            .enumerate()
            .map(|(i, _)| mul(map, nodes, x[i], self.weights[i]))
            .last()
            .unwrap();

        let (sum, _) = add(map, nodes, self.bias, last_index);
        let (res, _) = relu(map, nodes, sum);
        res
    }

    fn parameters(&self) -> Vec<usize> {
        let mut p = Vec::new();

        p.extend(self.weights.clone());
        p.push(self.bias);

        p
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nodes: &mut Vec<Node>, count_in: u64, count_out: u64) -> Self {
        let mut neurons = Vec::new();

        for _ in 0..count_out {
            neurons.push(Neuron::new(nodes, count_in));
        }

        Layer { neurons: neurons }
    }

    pub fn connect(&self, map: &mut Map, nodes: &mut Vec<Node>, x: Vec<usize>) -> Vec<usize> {
        self.neurons
            .iter()
            .map(|neuron| neuron.connect(map, &x, nodes))
            .collect()
    }

    pub fn parameters(&self) -> Vec<usize> {
        let mut p = Vec::new();

        for neuron in self.neurons.iter() {
            p.extend(neuron.parameters());
        }

        p
    }
}

fn get_child_nodes(
    nodes: &mut Vec<Node>,
    index_a: Option<usize>,
    index_b: Option<usize>,
) -> [Option<&mut Node>; 2] {
    match (index_a, index_b) {
        (Some(x), Some(y)) => {
            assert!(y > x);

            let (lb, rb) = nodes.split_at_mut(y);

            [Some(&mut lb[x]), Some(&mut rb[0])]
        }
        (Some(x), None) => [Some(&mut nodes[x]), None],
        _ => [None, None],
    }
}

pub fn backwards(map: &mut Map, nodes: &mut Vec<Node>) {
    let mut visited = HashSet::new();
    let mut deque = VecDeque::new();

    let last_node = nodes.last().clone().unwrap();
    deque.push_back(last_node.id);

    while let Some(node_id) = deque.pop_front() {
        println!("{:?}", node_id);
        let node_clone = nodes[node_id].clone();

        if !visited.contains(&node_clone.id) {
            visited.insert(node_clone.id);

            match map.get(&node_clone.id) {
                Some(child_nodes) => {
                    let children = get_child_nodes(nodes, child_nodes[0], child_nodes[1]);

                    match children {
                        [Some(self_node), Some(other_node)] => {
                            deque.extend([self_node.id, other_node.id]);

                            match &node_clone.operator {
                                Some(Operator::Plus) => {
                                    self_node.gradient += node_clone.gradient;
                                    other_node.gradient += node_clone.gradient;
                                }
                                Some(Operator::Mul) => {
                                    self_node.gradient += other_node.value * node_clone.gradient;
                                    other_node.gradient += self_node.value * node_clone.gradient;
                                }
                                Some(Operator::Pow) => {
                                    self_node.gradient += other_node.value
                                        * self_node.value.powf(1.0 - other_node.value)
                                        * node_clone.gradient;
                                }
                                _ => {}
                            }
                        }
                        [Some(self_node), None] => {
                            deque.extend([self_node.id]);

                            match &node_clone.operator {
                                Some(Operator::Relu) => {
                                    if node_clone.value > 0.0 {
                                        self_node.gradient += node_clone.gradient;
                                    }
                                }
                                _ => {}
                            }
                        }
                        _ => {}
                    }
                }
                None => {}
            };
        }
    }
}

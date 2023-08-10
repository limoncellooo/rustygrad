// #![allow(dead_code, unused_variables)]

mod node;

use std::collections::HashMap;

fn main() {
    let mut map: node::Map = HashMap::new();
    let mut nodes: Vec<node::Node> = Vec::new();

    let count_in = 2;
    let mut inputs = Vec::new();
    for _ in 0..count_in {
        inputs.push(node::new_node(&mut nodes, 1.0));
    }

    let layer1 = node::Layer::new(&mut nodes, 2, 1);
    let final_layer = layer1.connect(&mut map, &mut nodes, inputs);

    nodes[final_layer[0]].gradient = 1.0;

    node::backwards(&mut map, &mut nodes);

    println!("{:?}", map);
    println!("{:?}", map);
    for n in nodes {
        println!("{:?}", n);
    }
}

mod test {
    #[test]
    fn basics() {
        // TODO
    }
}

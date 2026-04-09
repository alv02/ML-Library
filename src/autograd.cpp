#include "../include/autograd.hpp"
#include <algorithm>
#include <unordered_set>

function_var::function_var(Tensor *val, u32 flags)
    : val(val), flags(flags), grad(nullptr), grad_fn(nullptr) {}

function_var::~function_var() {
    delete val;
    if (grad)
        delete grad;
}

void topological_sort(Graph *graph, function_var *fv,
                      std::unordered_set<function_var *> &visited) {
    if (visited.count(fv))
        return;
    visited.insert(fv);

    if (fv->grad_fn) {
        for (u32 i = 0; i < fv->grad_fn->n_inputs; i++) {
            topological_sort(graph, fv->grad_fn->inputs[i], visited);
        }
    }
    graph->nodes.push_back(fv); // post-order: inputs before output
}

Graph *graph_create(function_var *output) {
    Graph *graph = new Graph();
    std::unordered_set<function_var *> visited;
    topological_sort(graph, output, visited);
    // nodes is [leaves...loss], reverse for backward [loss...leaves]
    std::reverse(graph->nodes.begin(), graph->nodes.end());
    return graph;
}

void graph_free(Graph *graph) {
    graph->nodes.clear();
    delete graph;
}

void graph_backward(Graph *graph) {
    if (graph->nodes.empty())
        return;
    if (!graph->nodes[0]->grad)
        graph->nodes[0]->grad = tensor_create_like(graph->nodes[0]->val);
    tensor_fill(graph->nodes[0]->grad, 1.0f); // dL/dL = 1

    for (function_var *fv : graph->nodes) {
        if (fv->grad_fn) {
            fv->grad_fn->backward(fv->grad);
        }
    }
}

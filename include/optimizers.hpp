#ifndef OPTIMIZERS_HPP
#define OPTIMIZERS_HPP

#include "autograd.hpp"

struct gd_optimizer {
    f32 lr;

    Graph *graph; // Non-owning reference — graph is owned by the model

    gd_optimizer(f32 learning_rate);
    ~gd_optimizer();
    void zero_grad();
    void step();
    void set_graph(Graph *graph);
};

#endif

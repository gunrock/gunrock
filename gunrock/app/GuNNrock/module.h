//
// Created by husong on 5/13/20.
//

#ifndef GUNROCK_MODULE_H
#define GUNROCK_MODULE_H

#include <gunrock/util/test_utils.cuh>

struct module {
    gunrock::util::GpuTimer timer;
    virtual ~module() = default;
    virtual void forward(bool) = 0;
    virtual void backward() = 0;
    virtual double GetLoss() {
        return 0;
    }
};

#endif //GUNROCK_MODULE_H

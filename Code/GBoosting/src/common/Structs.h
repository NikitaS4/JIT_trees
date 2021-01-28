#ifndef STRUCTS_H
#define STRUCTS_H

#include "PybindHeader.h"
#include "AtomicTypes.h"

using pytensor1 = xt::pytensor<FVal_t, 1>;
using pytensor2 = xt::pytensor<FVal_t, 2>;
using pytensorY = xt::pytensor<Lab_t, 1>;
using pytensor2Y = xt::pytensor<Lab_t, 2>;

#endif // STRUCTS_H

#ifndef STRUCTS_H
#define STRUCTS_H

#include "PybindHeader.h"

using FVal_t = double; // INPUT data type (the value of each feature)
using Lab_t = double; // OUTPUT data type
using pyarray = xt::pyarray<FVal_t>;
using pyarrayY = xt::pyarray<Lab_t>;

#endif // STRUCTS_H

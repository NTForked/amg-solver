#include "coarsener.h"

namespace amg {

transfer_type ruge_stuben::transfer_operator(const spmat_csr &A) {
    return std::make_tuple(nullptr, nullptr);
}

ptr_spmat_csr ruge_stuben::coarse_operator(const spmat_csr &A,
                                           const spmat_csr &P,
                                           const spmat_csr &R) {
    return nullptr;
}

}

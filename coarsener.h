#ifndef AMG_COARSENER_H
#define AMG_COARSENER_H

#include "types.h"

namespace amg {

class coarsener
{
public:
    virtual ~coarsener() {}
    virtual transfer_type transfer_operator(const spmat_csr &A) = 0;
    virtual ptr_spmat_csr coarse_operator(const spmat_csr &A, const spmat_csr &R, const spmat_csr &P) = 0;
};

class ruge_stuben : public coarsener
{
public:
    transfer_type transfer_operator(const spmat_csr &A);
    ptr_spmat_csr coarse_operator(const spmat_csr &A, const spmat_csr &R, const spmat_csr &P) {
        return ptr_spmat_csr(new spmat_csr(R*A*P));
    }
private:
    void connect();
    void cfsplit();
};

class aggregation : public coarsener
{
public:
    transfer_type transfer_operator(const spmat_csr &A);
    ptr_spmat_csr coarse_operator(const spmat_csr &A, const spmat_csr &R, const spmat_csr &P);
};

}
#endif

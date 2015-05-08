#ifndef AMG_COARSENER_H
#define AMG_COARSENER_H

#include "types.h"

namespace amg {

class coarsener
{
public:
    virtual ~coarsener() {}
    virtual transfer_type transfer_operator(const spmat_csr &A) = 0;
    virtual ptr_spmat_csr coarse_operator(const spmat_csr &A, const spmat_csr &P, const spmat_csr &R) = 0;
};

class ruge_stuben : public coarsener
{
public:
    ruge_stuben();
    ruge_stuben(const scalar eps_strong, const bool do_trunc, const scalar eps_trunc);
    transfer_type transfer_operator(const spmat_csr &A);
    ptr_spmat_csr coarse_operator(const spmat_csr &A, const spmat_csr &P, const spmat_csr &R);
    int debug_cfsplit() const;
private:
    void connect(const spmat_csr &A, spmat_csr_char &S, spmat_csr_char &ST, std::vector<char> &cf_tag);
    void cfsplit(const spmat_csr &A, const spmat_csr_char &S, const spmat_csr_char &ST, std::vector<char> &cf_tag);
    const scalar eps_strong_;
    const bool do_trunc_;
    const scalar eps_trunc_;
};

class aggregation : public coarsener
{
public:
    transfer_type transfer_operator(const spmat_csr &A);
    ptr_spmat_csr coarse_operator(const spmat_csr &A, const spmat_csr &P, const spmat_csr &R);
};

}
#endif

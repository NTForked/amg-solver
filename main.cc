#include <iostream>
#include <string>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <zjucad/ptree/ptree.h>

#include "smoother.h"

#define CALL_SUB_PROG(prog)                      \
    int prog(ptree &pt);                         \
    if ( pt.get<string>("prog.value") == #prog)  \
        return prog(pt);

using namespace std;
using boost::property_tree::ptree;

int test_smoother(ptree &pt) {
    shared_ptr<amg::smoother> smooth(new amg::gauss_seidel);
    return 0;
}

int main(int argc, char *argv[])
{
    ptree pt;
    try {
        zjucad::read_cmdline(argc, argv, pt);
        CALL_SUB_PROG(test_smoother);
    } catch (const boost::property_tree::ptree_error &e) {
        cerr << "Usage: " << endl;
        zjucad::show_usage_info(std::cerr, pt);
    } catch (const std::exception &e) {
        cerr << "# " << e.what() << endl;
    }
    return 0;
}

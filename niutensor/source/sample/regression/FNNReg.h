#include "../../tensor/XGlobal.h"
#include "../../tensor/XTensor.h"
#include "../../tensor/core/CHeader.h"
using namespace nts;

namespace fnnreg
{
    struct FNNRegModel
    {
        XTensor weight1;

        XTensor weight2;

        XTensor b;

        int h_size;

        int devID;
    };

    struct FNNRegNet
    {
        /*before bias*/
        XTensor hidden_state1;
        /*before active function*/
        XTensor hidden_state2;
        /*after active function*/
        XTensor hidden_state3;
        /*output*/
        XTensor output;
    };
    int FNNRegMain(int argc, const char ** argv);
};
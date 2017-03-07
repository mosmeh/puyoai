#include <gflags/gflags.h>
#include <glog/logging.h>

#include "ai.h"

DEFINE_int32(seed, 1, "seed");

int main(int argc, char* argv[])
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
#if !defined(_MSC_VER)
    google::InstallFailureSignalHandler();
#endif

    std::mt19937 engine(FLAGS_seed);
    kurumi::DFKurumiAI(argc, argv, engine).runLoop();
    return 0;
}

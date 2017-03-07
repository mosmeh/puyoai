#include <gflags/gflags.h>
#include <glog/logging.h>

#include "reinforce_kurumi_ai.h"

int main(int argc, char* argv[])
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
#if !defined(_MSC_VER)
    google::InstallFailureSignalHandler();
#endif

    kurumi::ReinforceKurumiAI(argc, argv).runLoop();
    return 0;
}


#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include "TFile.h"
#include "TTree.h"
#include "predictor/data_loader.h"

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
    }
  // open the file
   TFile *f = TFile::Open("/public/data/RNN/Samples/InclusivePt/small_data.root");
   if (f == 0) {
      // if we cannot open the file, print an error message and return immediatly
      printf("Error: cannot open input data!\n");
      return -1;
   }

  std::cout << "ok\n";
  return 0;
}

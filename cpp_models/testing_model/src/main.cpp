#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include "TFile.h"
#include "TTree.h"
#include "predictor/data_loader.h"
#include "TH1F.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include <vector>
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
   TTree *tree = nullptr;
   f->GetObject("NormalizedTree",tree);
   tree->GetEntry(0);
   TBranch *pT = tree->GetBranch("calo_cluster_pT");
   std::vector<float> calo_cluster_pT;
   tree->SetBranchAddress("calo_cluster_pT", &calo_cluster_pT);
   // Create a TTreeReader for the tree, for instance by passing the
   // TTree's name and the TDirectory / TFile it is in.
   // TTreeReader myReader("NormalizedTree", f);
   // // The branch "px" contains floats; access them as myPx.
   // TTreeReaderValue<float> calo_cluster_pT(myReader, "calo_cluster_pT");
   std::cout<<calo_cluster_pT << "ok\n";
  return 0;
}

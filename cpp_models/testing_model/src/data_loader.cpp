#include "predictor/data_loader.h"
#include <torch/torch.h>

ROOT_Dataset::ROOT_Dataset(std::string& file_location, std::string& tree_name)
    //load root files and trees
    {
        TFile *file = TFile::Open(file_location.c_str());
        TTree *tree= nullptr;
        file->GetObject(tree_name.c_str(),tree);
    };

ROOT_Dataset::~ROOT_Dataset()
{
    file->Close();
};
void ROOT_Dataset::read_data()
{
    tree->GetEntry(0);
    TBranch *pT = tree->GetBranch("calo_cluster_pT");
    std::vector<float> calo_cluster_pT;
    tree->SetBranchAddress("calo_cluster_pT", &calo_cluster_pT);
};

torch::data::Example<> ROOT_Dataset::get(size_t index)
{
    tree->GetEntry(index);
};

// // Generate your data set. At this point you can add transforms to you data set, e.g. stack your
// // batches into a single tensor.
// auto data_set = MyDataset(loc_states, loc_labels).map(torch::data::transforms::Stack<>());
//
// // Generate a data loader.
// auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
//     std::move(data_set),
//     batch_size);
//
// // In a for loop you can now use your data.
// for (auto& batch : data_loader) {
//     auto data = batch.data;
//     auto labels = batch.target;
//     // do your usual stuff
// }

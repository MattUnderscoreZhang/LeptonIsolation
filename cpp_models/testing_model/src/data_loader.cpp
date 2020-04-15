#include "predictor/data_loader.h"
#include <torch/torch.h>

ROOT_Dataset::ROOT_Dataset(std::string& file_location, std::string& tree_name)
    //load root files and trees
    {
        TFile *file = TFile::Open(file_location.c_str());
        TTree *tree=(TTree*)file->Get(tree_name.c_str());
    };

ROOT_Dataset::~ROOT_Dataset()
{
    file->Close();
};
void ROOT_Dataset::read_data()
{
    this->tree;
};

torch::data::Example<> ROOT_Dataset::get(size_t index)
{

    // return 0;
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

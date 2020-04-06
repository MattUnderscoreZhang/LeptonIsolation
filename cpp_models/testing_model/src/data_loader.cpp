#include "predictor/data_loader.h"

ROOT_Dataset::ROOT_Dataset(std::string& file_location, std::string& tree_name)
    //load root files and trees
    {
        file = new TFile(file_location);
        tree = (TTree*)file->Get(tree_name);
    };

ROOT_Dataset::~ROOT_Dataset()
{
    file->Close()
}

ROOT_Dataset::torch::optional<size_t> size() const override
{
    return 0;
};

ROOT_Dataset::torch::data::Example<> get(size_t index) override
{

    return 0;
};

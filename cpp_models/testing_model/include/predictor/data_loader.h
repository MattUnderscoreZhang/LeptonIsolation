#pragma once

#include <vector>
#include <tuple>
#include <torch/torch.h>
#include "TFile.h"
#include "TTree.h"

class ROOT_Dataset : public torch::data::Dataset<ROOT_Dataset>
{
    private:
        std::string file_location;
        TFile *file;
        TTree *tree;

    public:
        /*constructors and destructors*/
        explicit ROOT_Dataset(std::string& file_location, std::string& tree_name);
        //load root files and trees
        ~ROOT_Dataset(); // closes TFile

        /*methods*/
        void read_data();
        torch::optional<size_t> size() const override
        {
            // return 0;
        }; // create override for datasize
        torch::data::Example<> get(size_t index) override;// create override for getting data item
        // create list of readable events
        // store tree in _store_tree_in_memory



};

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h> // Add this include for Eigen matrix type conversion
#include "recommender.h"

namespace py = pybind11;

PYBIND11_MODULE(ialspp, m) {

    m.def("SaveModel", &SaveModel);
    m.def("LoadModel", &LoadModel);

    // Register Eigen matrix type conversion
    py::class_<Eigen::Matrix<float, -1, 1, 0, -1, 1>>(m, "EigenMatrix")
        .def(py::init<>())
        .def(py::init<const Eigen::Matrix<float, -1, 1, 0, -1, 1>&>())
        .def("rows", &Eigen::Matrix<float, -1, 1, 0, -1, 1>::rows)
        .def("cols", &Eigen::Matrix<float, -1, 1, 0, -1, 1>::cols)
        .def("__getitem__", [](const Eigen::Matrix<float, -1, 1, 0, -1, 1>& m, int i) {
            if (i >= 0 && i < m.rows()) {
                return m(i);
            } else {
                throw py::index_error();
            }
        });

    py::class_<Recommender>(m, "Recommender")
        .def("Train", &Recommender::Train)
        .def("EvaluateDataset", &Recommender::EvaluateDataset)
        .def("EvaluateUser", &Recommender::EvaluateUser)
        .def("Score", &Recommender::Score);

    py::class_<IALSppRecommender, Recommender>(m, "IALSppRecommender")
        .def(py::init<int, int, int, float, float, float, float, int>())
        .def("SetPrintTrainStats", &IALSppRecommender::SetPrintTrainStats)
        .def("Train", &IALSppRecommender::Train)
        .def("EvaluateDataset", &IALSppRecommender::EvaluateDataset)
        .def("EvaluateUser", &IALSppRecommender::EvaluateUser)
        .def("Score", &IALSppRecommender::Score);

    py::class_<Dataset>(m, "Dataset")
        .def(py::init<const std::string&, bool>(), py::arg("filename"), py::arg("string_id") = false)
        .def("by_user", &Dataset::by_user)
        .def("by_item", &Dataset::by_item)
        .def("max_user", &Dataset::max_user)
        .def("max_item", &Dataset::max_item)
        .def("user_encoder", &Dataset::user_encoder)
        .def("item_encoder", &Dataset::item_encoder)
        .def("num_tuples", &Dataset::num_tuples);
    
    py::class_<Encoder>(m, "Encoder")
        .def(py::init<>())
        .def("insert", &Encoder::insert)
        .def("encode", &Encoder::encode)
        .def("decode", &Encoder::decode)
        .def("size", &Encoder::size);

    m.def("ProjectBlock", &ProjectBlock, "Project a block of embeddings");
}

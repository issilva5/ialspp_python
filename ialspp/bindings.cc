#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h> // Add this include for Eigen matrix type conversion
#include "recommender.h"

namespace py = pybind11;

PYBIND11_MODULE(ialspp, m) {

    m.def("SaveModel", &SaveModel, py::arg("path"), py::arg("model"));
    m.def("LoadModel", &LoadModel, py::arg("path"));

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
        .def("Train", &Recommender::Train, py::arg("dataset"))
        .def("EvaluateDataset", &Recommender::EvaluateDataset, py::arg("test_train_data"), py::arg("test_test_data"))
        .def("EvaluateUser", &Recommender::EvaluateUser, py::arg("scores"), py::arg("ground_truth"), py::arg("exclude"))
        .def("Score", &Recommender::Score, py::arg("user"), py::arg("user_history"));

    py::class_<IALSppRecommender, Recommender>(m, "IALSppRecommender")
        .def(py::init<int, int, int, float, float, float, float, int>(),
                py::arg("embedding_dim"),
                py::arg("num_users"),
                py::arg("num_items"),
                py::arg("regularization"),
                py::arg("regularization_exp"),
                py::arg("unobserved_weight"),
                py::arg("stddev"),
                py::arg("block_size")
        )
        .def("SetPrintTrainStats", &IALSppRecommender::SetPrintTrainStats, py::arg("print_train_stats"))
        .def("Train", &IALSppRecommender::Train, py::arg("dataset"))
        .def("EvaluateDataset", &IALSppRecommender::EvaluateDataset, py::arg("test_train_data"), py::arg("test_test_data"))
        .def("EvaluateUser", &IALSppRecommender::EvaluateUser, py::arg("scores"), py::arg("ground_truth"), py::arg("exclude"))
        .def("Score", &IALSppRecommender::Score, py::arg("user"), py::arg("user_history"));

    py::class_<Dataset>(m, "Dataset")
        .def(py::init<const std::string&, bool>(), py::arg("filename"), py::arg("string_id") = false)
        .def("by_user", &Dataset::by_user)
        .def("by_item", &Dataset::by_item)
        .def("max_user", &Dataset::max_user)
        .def("max_item", &Dataset::max_item)
        .def("user_encoder", &Dataset::user_encoder)
        .def("item_encoder", &Dataset::item_encoder)
        .def("num_tuples", &Dataset::num_tuples)
        .def(py::pickle(
            [](const Dataset &p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(p.serialize());
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 1)
                    throw std::runtime_error("Invalid state!");
                return Dataset::deserialize(t[0].cast<std::string>());
            }
        ));
    
    py::class_<Encoder>(m, "Encoder")
        .def(py::init<>())
        .def("insert", &Encoder::insert, py::arg("s"))
        .def("encode", &Encoder::encode, py::arg("s"))
        .def("decode", &Encoder::decode, py::arg("i"))
        .def("size", &Encoder::size)
        .def(py::pickle(
            [](const Encoder &p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(p.serialize());
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 1)
                    throw std::runtime_error("Invalid state!");
                return Encoder::deserialize(t[0].cast<std::string>());
            }
        ));

    m.def("ProjectBlock", &ProjectBlock, "Project a block of embeddings");
}

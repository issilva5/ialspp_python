// Copyright 2024 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <stdlib.h>
#include <stdio.h>
#include <thread>
#include <unordered_map>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Core"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

using SpVector = std::vector<std::pair<int, int>>;
using SpMatrix = std::unordered_map<int, SpVector>;

class Encoder {
  public:

    Encoder() {}

    Encoder(int n, std::unordered_map<std::string, int> stoi, std::unordered_map<int, std::string> itos) {
      n_ = n;
      stoi_ = stoi;
      itos_ = itos;
    }

    int insert(std::string s) {
      if(stoi_.find(s) != stoi_.end()) return stoi_[s];

      stoi_[s] = n_;
      itos_[n_] = s;
      n_++;

      return stoi_[s];
    }

    int encode(std::string s) {
      if(stoi_.find(s) != stoi_.end()) return stoi_[s];
      return -1;
    }

    std::string decode(int i) {
      if(itos_.find(i) != itos_.end()) return itos_[i];
      return "";
    }

    const int size() const { return n_; }

    std::string serialize() const {
      json j;
      j["n"] = n_;
      j["stoi"] = stoi_;
      j["itos"] = itos_;
      return j.dump();
    }

    static Encoder deserialize(const std::string &state) {
        json j = json::parse(state);
        int n = j["n"];
        std::unordered_map<std::string, int> stoi = j["stoi"].get<std::unordered_map<std::string, int>>();
        std::unordered_map<int, std::string> itos = j["itos"].get<std::unordered_map<int, std::string>>();
        return Encoder(n, stoi, itos);
    }
  
  private:
    std::unordered_map<std::string, int> stoi_;
    std::unordered_map<int, std::string> itos_;
    int n_ = 0;
};

class Dataset {
 public:
  Dataset(const std::string& filename, bool string_id = false) {
    max_user_ = -1;
    max_item_ = -1;
    num_tuples_ = 0;
    std::ifstream infile(filename);
    std::string line;

    int user, item;
    std::string users, items;

    // Discard header.
    std::getline(infile, line);

    // Read the data.
    while (std::getline(infile, line)) {

      std::stringstream ss(line);
      std::getline(ss, users, ',');
      std::getline(ss, items, ',');

      if (!string_id) {
        user = std::atoi(users.c_str());
        item = std::atoi(items.c_str());
      } else {
        user = user_encoder_.insert(users);
        item = item_encoder_.insert(items);
      }

      by_user_[user].push_back({item, num_tuples_});
      by_item_[item].push_back({user, num_tuples_});
      max_user_ = std::max(max_user_, user);
      max_item_ = std::max(max_item_, item);
      ++num_tuples_;
    }
    std::cout << "max_user=" << max_user()
              << "\tmax_item=" << max_item()
              << "\tdistinct user=" << by_user_.size()
              << "\tdistinct item=" << by_item_.size()
              << "\tnum_tuples=" << num_tuples()
              << std::endl;
  }

  Dataset(int max_user, int max_item, int num_tuples, SpMatrix by_user, SpMatrix by_item, Encoder user_encoder,
          Encoder item_encoder) {
    
    max_user_ = max_user;
    max_item_ = max_item;
    num_tuples_ = num_tuples;

    by_user_ = by_user;
    by_item_ = by_item;
    user_encoder_ = user_encoder;
    item_encoder_ = item_encoder;
  
  }

  const SpMatrix& by_user() const { return by_user_; }
  const SpMatrix& by_item() const { return by_item_; }
  const int max_user() const { return max_user_; }
  const int max_item() const { return max_item_; }
  const int num_tuples() const { return num_tuples_; }
  const Encoder& user_encoder() const { return user_encoder_; }
  const Encoder& item_encoder() const { return item_encoder_; }

  std::string serialize() const {
      json j;
      j["by_user"] = by_user_;
      j["by_item"] = by_item_;
      j["user_encoder"] = user_encoder_.serialize();
      j["item_encoder"] = item_encoder_.serialize();
      j["max_user"] = max_user_;
      j["max_item"] = max_item_;
      j["num_tuples"] = num_tuples_;
      return j.dump();
  }

  static Dataset deserialize(const std::string &state) {
      json j = json::parse(state);

      int max_user = j["max_user"];
      int max_item = j["max_item"];
      int num_tuples = j["num_tuples"];

      SpMatrix by_user = j["by_user"].get<SpMatrix>();
      SpMatrix by_item = j["by_item"].get<SpMatrix>();

      Encoder user_encoder = Encoder::deserialize(j["user_encoder"]);
      Encoder item_encoder = Encoder::deserialize(j["item_encoder"]);

      return Dataset(max_user, max_item, num_tuples, by_user, by_item, user_encoder, item_encoder);
  }

 private:
  SpMatrix by_user_;
  SpMatrix by_item_;
  Encoder user_encoder_;
  Encoder item_encoder_;
  int max_user_;
  int max_item_;
  int num_tuples_;
};

class Recommender {
 public:
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      MatrixXf;
  typedef Eigen::VectorXf VectorXf;

  virtual ~Recommender() {}

  virtual VectorXf Score(const int user_id, const SpVector& user_history) {
    return VectorXf::Zero(1);
  }

  virtual void Train(const Dataset& dataset) {}

  VectorXf EvaluateUser(const VectorXf& all_scores,
                        const SpVector& ground_truth,
                        const SpVector& exclude);

  // Templated implementation for evaluating a dataset. Requires a function that
  // scores all items for a given user or history.
  template <typename F>
      VectorXf EvaluateDatasetInternal(
      const Dataset& data, const SpMatrix& eval_by_user,
      F score_user_and_history) {
    std::mutex m;
    auto eval_by_user_iter = eval_by_user.begin();  // protected by m
    VectorXf metrics = VectorXf::Zero(3);

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(std::thread([&]{
        while (true) {
          // Get a new user to work on.
          m.lock();
          if (eval_by_user_iter == eval_by_user.end()) {
            m.unlock();
            return;
          }
          int u = eval_by_user_iter->first;
          SpVector ground_truth = eval_by_user_iter->second;
          ++eval_by_user_iter;
          m.unlock();

          // Process the user.
          const SpVector& user_history = data.by_user().at(u);
          VectorXf scores = score_user_and_history(u, user_history);
          VectorXf this_metrics = this->EvaluateUser(scores, ground_truth,
                                                     user_history);

          // Update the metric.
          m.lock();
          metrics += this_metrics;
          m.unlock();
        }
      }));
    }

    // Join all threads.
    for (int i = 0; i < threads.size(); ++i) {
      threads[i].join();
    }
    metrics /= eval_by_user.size();
    return metrics;
  }

  // Common implementation for evaluating a dataset. It uses the scoring
  // function of the class.
  virtual VectorXf EvaluateDataset(
      const Dataset& data, const SpMatrix& eval_by_user);
};

Recommender::VectorXf Recommender::EvaluateUser(
    const VectorXf& all_scores,
    const SpVector& ground_truth,
    const SpVector& exclude) {
  VectorXf scores = all_scores;
  for (int i = 0; i < exclude.size(); ++i) {
    assert(exclude[i].first < scores.size());
    scores[exclude[i].first] = std::numeric_limits<float>::lowest();
  }

  std::vector<size_t> topk(scores.size());
  std::iota(topk.begin(), topk.end(), 0);
  std::stable_sort(topk.begin(), topk.end(),
                   [&scores](size_t i1, size_t i2) {
                     return scores[i1] > scores[i2];
                   });
  auto recall = [](int k, const std::set<int>& gt_set,
                   const std::vector<size_t>& topk) -> float {
    double result = 0.0;
    for (int i = 0; i < k; ++i) {
      if (gt_set.find(topk[i]) != gt_set.end()) {
        result += 1.0;
      }
    }
    return result / std::min<float>(k, gt_set.size());};

  auto ndcg = [](int k, const std::set<int>& gt_set,
                 const std::vector<size_t>& topk) -> float {
    double result = 0.0;
    for (int i = 0; i < k; ++i) {
      if (gt_set.find(topk[i]) != gt_set.end()) {
        result += 1.0 / log2(i+2);
      }
    }
    double norm = 0.0;
    for (int i = 0; i < std::min<int>(k, gt_set.size()); ++i) {
      norm += 1.0 / log2(i+2);
    }
    return result / norm;};

  std::set<int> gt_set;
  std::transform(ground_truth.begin(), ground_truth.end(),
                 std::inserter(gt_set, gt_set.begin()),
                 [](const std::pair<int, int>& p) { return p.first; });
  VectorXf result(3);
  result << recall(20, gt_set, topk),
            recall(50, gt_set, topk),
            ndcg(100, gt_set, topk);
  return result;
}

Recommender::VectorXf Recommender::EvaluateDataset(
    const Dataset& data, const SpMatrix& eval_by_user) {
  return EvaluateDatasetInternal(
      data, eval_by_user,
      [&](const int user_id, const SpVector& history) -> VectorXf {
        return Score(user_id, history);
      });
}



const Recommender::VectorXf ProjectBlock(
    const SpVector& user_history,
    const Recommender::VectorXf& user_embedding,
    const Recommender::VectorXf& local_user_embedding,
    const Recommender::MatrixXf& local_item_embedding,
    const Recommender::VectorXf& prediction,
    const Recommender::MatrixXf& local_gramian,
    const Recommender::MatrixXf& local_global_gramian,
    const float reg, const float unobserved_weight) {
  assert(user_history.size() > 0);
  int local_embedding_dim = local_item_embedding.cols();
  assert(local_embedding_dim > 0);

  Recommender::VectorXf new_value(local_embedding_dim);

  Eigen::MatrixXf matrix = unobserved_weight * local_gramian;

  for (int i = 0; i < local_embedding_dim; ++i) {
    matrix(i, i) += reg;
  }

  const int kMaxBatchSize = 128;
  auto matrix_symm = matrix.selfadjointView<Eigen::Lower>();
  Eigen::VectorXf rhs = Eigen::VectorXf::Zero(local_embedding_dim);
  const int batch_size = std::min(static_cast<int>(user_history.size()),
                                  kMaxBatchSize);
  int num_batched = 0;
  Eigen::MatrixXf factor_batch(local_embedding_dim, batch_size);
  for (const auto& item_and_rating_index : user_history) {
    const int cp = item_and_rating_index.first;
    const int rating_index = item_and_rating_index.second;
    assert(cp < local_item_embedding.rows());
    assert(rating_index < prediction.size());
    const Recommender::VectorXf cp_v = local_item_embedding.row(cp);

    const float residual = (prediction.coeff(rating_index) - 1.0);

    factor_batch.col(num_batched) = cp_v;
    rhs += cp_v * residual;

    ++num_batched;
    if (num_batched == batch_size) {
      matrix_symm.rankUpdate(factor_batch);
      num_batched = 0;
    }
  }
  if (num_batched != 0) {
    auto factor_block = factor_batch.block(
        0, 0, local_embedding_dim, num_batched);
    matrix_symm.rankUpdate(factor_block);
  }

  // add "prediction" for the unobserved items
  rhs += unobserved_weight * local_global_gramian * user_embedding;
  // add the regularization.
  rhs += reg * local_user_embedding;

  Eigen::LLT<Eigen::MatrixXf, Eigen::Lower> cholesky(matrix);
  assert(cholesky.info() == Eigen::Success);
  new_value = local_user_embedding - cholesky.solve(rhs);

  return new_value;
}



class IALSppRecommender : public Recommender {
 public:
  IALSppRecommender(int embedding_dim, int num_users, int num_items, float reg,
                  float reg_exp, float unobserved_weight, float stdev,
                  int block_size)
      : user_embedding_(num_users, embedding_dim),
        item_embedding_(num_items, embedding_dim) {
    // Initialize embedding matrices
    float adjusted_stdev = stdev / sqrt(embedding_dim);
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> d(0, adjusted_stdev);
    auto init_matrix = [&](Recommender::MatrixXf* matrix) {
      for (int i = 0; i < matrix->size(); ++i) {
        *(matrix->data() + i) = d(gen);
      }
    };
    init_matrix(&user_embedding_);
    init_matrix(&item_embedding_);

    regularization_ = reg;
    regularization_exp_ = reg_exp;
    embedding_dim_ = embedding_dim;
    unobserved_weight_ = unobserved_weight;
    block_size_ = std::min(block_size, embedding_dim_);
  }

  VectorXf Score(const int user_id, const SpVector& user_history) override {
    if (user_id >= 0 && user_id < user_embedding_.rows()) {
        // Access the user embedding corresponding to user_id
        VectorXf user_emb = user_embedding_.row(user_id);

        // Compute the score by multiplying the item_embedding with user_emb
        return item_embedding_ * user_emb;
    } else {
        // Handle invalid user_id
        throw std::invalid_argument("Invalid user ID");
    }
  }

  // Custom implementation of EvaluateDataset that does the projection using the
  // iterative optimization algorithm.
  VectorXf EvaluateDataset(
      const Dataset& data, const SpMatrix& eval_by_user) override {
    int num_epochs = 8;

    std::unordered_map<int, VectorXf> user_to_emb;
    VectorXf prediction(data.num_tuples());

    // Initialize the user and predictions to 0.0. (Note: this code needs to
    // change if the embeddings would have biases).
    for (const auto& user_and_history : data.by_user()) {
      user_to_emb[user_and_history.first] = VectorXf::Zero(embedding_dim_);
      for (const auto& item_and_rating_index : user_and_history.second) {
        prediction.coeffRef(item_and_rating_index.second) = 0.0;
      }
    }

    // Train the user embeddings for num_epochs.
    for (int e = 0; e < num_epochs; ++e) {
      // Predict the dataset using the new user embeddings and the existing item
      // embeddings.
      for (const auto& user_and_history : data.by_user()) {
        const VectorXf& user_emb = user_to_emb[user_and_history.first];
        for (const auto& item_and_rating_index : user_and_history.second) {
          prediction.coeffRef(item_and_rating_index.second) =
              item_embedding_.row(item_and_rating_index.first).dot(user_emb);
        }
      }

      // Optimize the user embeddings for each block.
      for (int start = 0; start < embedding_dim_; start += block_size_) {
        assert(start < embedding_dim_);
        int end = std::min(start + block_size_, embedding_dim_);

        Step(data.by_user(), start, end, &prediction,
             [&](const int user_id) -> VectorXf& {
               return user_to_emb[user_id];
             },
             item_embedding_,
             /*index_of_item_bias=*/1);
      }
    }

    // Evalute the dataset.
    return EvaluateDatasetInternal(
        data, eval_by_user,
        [&](const int user_id, const SpVector& history) -> VectorXf {
          return item_embedding_ * user_to_emb[user_id];
        });
  }

  void Train(const Dataset& data) override {
    // Predict the dataset.
    VectorXf prediction(data.num_tuples());
    for (const auto& user_and_history : data.by_user()) {
      VectorXf user_emb = user_embedding_.row(user_and_history.first);
      for (const auto& item_and_rating_index : user_and_history.second) {
        prediction.coeffRef(item_and_rating_index.second) =
            item_embedding_.row(item_and_rating_index.first).dot(user_emb);
      }
    }

    for (int start = 0; start < embedding_dim_; start += block_size_) {
      assert(start < embedding_dim_);
      int end = std::min(start + block_size_, embedding_dim_);

      Step(data.by_user(), start, end, &prediction,
          [&](const int index) -> MatrixXf::RowXpr {
            return user_embedding_.row(index);
          },
          item_embedding_,
          /*index_of_item_bias=*/1);
      ComputeLosses(data, prediction);

      // Optimize the item embeddings
      Step(data.by_item(), start, end, &prediction,
          [&](const int index) -> MatrixXf::RowXpr {
            return item_embedding_.row(index);
          },
          user_embedding_,
          /*index_of_item_bias=*/0);
      ComputeLosses(data, prediction);
    }
  }

  void ComputeLosses(const Dataset& data, const VectorXf& prediction) {
    if (!print_trainstats_) {
      return;
    }
    auto time_start = std::chrono::steady_clock::now();
    int num_items = item_embedding_.rows();
    int num_users = user_embedding_.rows();

    // Compute observed loss.
    float loss_observed = (prediction.array() - 1.0).matrix().squaredNorm();

    // Compute regularizer.
    double loss_reg = 0.0;
    for (auto user_and_history : data.by_user()) {
      loss_reg += user_embedding_.row(user_and_history.first).squaredNorm() *
          RegularizationValue(user_and_history.second.size(), num_items);
    }
    for (auto item_and_history : data.by_item()) {
      loss_reg += item_embedding_.row(item_and_history.first).squaredNorm() *
          RegularizationValue(item_and_history.second.size(), num_users);
    }

    // Unobserved loss.
    MatrixXf user_gramian = user_embedding_.transpose() * user_embedding_;
    MatrixXf item_gramian = item_embedding_.transpose() * item_embedding_;
    float loss_unobserved = this->unobserved_weight_ * (
        user_gramian.array() * item_gramian.array()).sum();

    float loss = loss_observed + loss_unobserved + loss_reg;

    auto time_end = std::chrono::steady_clock::now();

    printf("Loss=%f, Loss_observed=%f Loss_unobserved=%f Loss_reg=%f Time=%d\n",
           loss, loss_observed, loss_unobserved, loss_reg,
           std::chrono::duration_cast<std::chrono::milliseconds>(
               time_end - time_start));
  }

  // Computes the regularization value for a user (or item). The value depends
  // on the number of observations for this user (or item) and the total number
  // of items (or users).
  const float RegularizationValue(int history_size, int num_choices) const {
    return this->regularization_ * pow(
              history_size + this->unobserved_weight_ * num_choices,
              this->regularization_exp_);
  }

  template <typename F>
  void Step(const SpMatrix& data_by_user,
            const int block_start,
            const int block_end,
            VectorXf* prediction,
            F get_user_embedding_ref,
            const MatrixXf& item_embedding,
            const int index_of_item_bias) {
    MatrixXf local_item_emb = item_embedding.block(
        0, block_start, item_embedding.rows(), block_end-block_start);

    // TODO: consider creating the local_gramian as a block from the
    // local_global_gramian
    MatrixXf local_gramian = local_item_emb.transpose() * local_item_emb;
    MatrixXf local_global_gramian = local_item_emb.transpose() * item_embedding;

    // Used for per user regularization.
    int num_items = item_embedding.rows();

    std::mutex m;
    auto data_by_user_iter = data_by_user.begin();  // protected by m
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(std::thread([&]{
        while (true) {
          // Get a new user to work on.
          m.lock();
          if (data_by_user_iter == data_by_user.end()) {
            m.unlock();
            return;
          }
          int u = data_by_user_iter->first;
          SpVector train_history = data_by_user_iter->second;
          ++data_by_user_iter;
          m.unlock();

          assert(!train_history.empty());
          float reg = RegularizationValue(train_history.size(), num_items);
          VectorXf old_user_emb = get_user_embedding_ref(u);
          VectorXf old_local_user_emb = old_user_emb.segment(
              block_start, block_end - block_start);
          VectorXf new_local_user_emb = ProjectBlock(
              train_history,
              old_user_emb,
              old_local_user_emb,
              local_item_emb,
              *prediction,
              local_gramian,
              local_global_gramian,
              reg, this->unobserved_weight_);
          // Update the ratings (without a lock)
          VectorXf delta_local_user_emb =
              new_local_user_emb - old_local_user_emb;
          for (const auto& item_and_rating_index : train_history) {
            prediction->coeffRef(item_and_rating_index.second) +=
                delta_local_user_emb.dot(
                    local_item_emb.row(item_and_rating_index.first));
          }
          // Update the user embedding.
          m.lock();
          get_user_embedding_ref(u).segment(
              block_start, block_end - block_start) = new_local_user_emb;
          m.unlock();
        }
      }));
    }
    // Join all threads.
    for (int i = 0; i < threads.size(); ++i) {
      threads[i].join();
    }
  }

  const MatrixXf& user_embedding() const { return user_embedding_; }

  const MatrixXf& item_embedding() const { return item_embedding_; }

  const int embedding_dim() const { return embedding_dim_; }

  const float regularization() const { return regularization_; }

  const float regularization_exp() const { return regularization_exp_; }

  const float unobserved_weight() const { return unobserved_weight_; }

  const int block_size() const { return block_size_; }

  void SetPrintTrainStats(const bool print_trainstats) {
    print_trainstats_ = print_trainstats;
  }

  void SetUserEmbedding(const MatrixXf& user_embedding) {
        user_embedding_ = user_embedding;
  }

  void SetItemEmbedding(const MatrixXf& item_embedding) {
      item_embedding_ = item_embedding;
  }

  std::string serialize() const {

    std::ostringstream oss;

    // Save embedding dimensions
    int embedding_dim = recommender.embedding_dim();
    oss.write(reinterpret_cast<const char*>(&embedding_dim), sizeof(int));

    // Save user embedding
    const Recommender::MatrixXf& user_embedding = recommender.user_embedding();
    int num_users = user_embedding.rows();
    oss.write(reinterpret_cast<const char*>(&num_users), sizeof(int));
    oss.write(reinterpret_cast<const char*>(user_embedding.data()),
                  user_embedding.rows() * user_embedding.cols() * sizeof(float));

    // Save item embedding
    const Recommender::MatrixXf& item_embedding = recommender.item_embedding();
    int num_items = item_embedding.rows();
    oss.write(reinterpret_cast<const char*>(&num_items), sizeof(int));
    oss.write(reinterpret_cast<const char*>(item_embedding.data()),
                  item_embedding.rows() * item_embedding.cols() * sizeof(float));

    // Save additional parameters
    // You'll need to replace these placeholders with the actual parameters
    // Here, I'm using example values
    float regularization = recommender.regularization();
    float regularization_exp = recommender.regularization_exp();
    float unobserved_weight = recommender.unobserved_weight();
    int block_size = recommender.block_size();
    oss.write(reinterpret_cast<const char*>(&regularization), sizeof(float));
    oss.write(reinterpret_cast<const char*>(&regularization_exp), sizeof(float));
    oss.write(reinterpret_cast<const char*>(&unobserved_weight), sizeof(float));
    oss.write(reinterpret_cast<const char*>(&block_size), sizeof(int));

    return oss.str();
  }

  static IALSppRecommender deserialize(const std::string &state) {
    std::istringstream iss(state);

    // Read embedding dimensions
    int embedding_dim;
    iss.read(reinterpret_cast<char*>(&embedding_dim), sizeof(int));

    // Read user embedding
    int num_users;
    iss.read(reinterpret_cast<char*>(&num_users), sizeof(int));
    Recommender::MatrixXf user_embedding(num_users, embedding_dim);
    iss.read(reinterpret_cast<char*>(user_embedding.data()),
                user_embedding.rows() * user_embedding.cols() * sizeof(float));

    // Read item embedding
    int num_items;
    iss.read(reinterpret_cast<char*>(&num_items), sizeof(int));
    Recommender::MatrixXf item_embedding(num_items, embedding_dim);
    iss.read(reinterpret_cast<char*>(item_embedding.data()),
                item_embedding.rows() * item_embedding.cols() * sizeof(float));

    // Read additional parameters
    float regularization, regularization_exp, unobserved_weight, stdev;
    int block_size;
    iss.read(reinterpret_cast<char*>(&regularization), sizeof(float));
    iss.read(reinterpret_cast<char*>(&regularization_exp), sizeof(float));
    iss.read(reinterpret_cast<char*>(&unobserved_weight), sizeof(float));
    iss.read(reinterpret_cast<char*>(&block_size), sizeof(int));

    IALSppRecommender* recommender;
    recommender = new IALSppRecommender(
      embedding_dim,
      num_users,
      num_items,
      regularization,
      regularization_exp,
      unobserved_weight,
      1,
      block_size);
    
    recommender->SetUserEmbedding(user_embedding);
    recommender->SetItemEmbedding(item_embedding);

    // Create and return the recommender object
    return *recommender;
  }

 private:
  MatrixXf user_embedding_;
  MatrixXf item_embedding_;

  float regularization_;
  float regularization_exp_;
  int embedding_dim_;
  float unobserved_weight_;
  int block_size_;

  bool print_trainstats_;
};

void SaveModel(const std::string& filename, const IALSppRecommender& recommender) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
        throw std::runtime_error("Failed to open file for writing");
    }

    // Save embedding dimensions
    int embedding_dim = recommender.embedding_dim();
    outfile.write(reinterpret_cast<const char*>(&embedding_dim), sizeof(int));

    // Save user embedding
    const Recommender::MatrixXf& user_embedding = recommender.user_embedding();
    int num_users = user_embedding.rows();
    outfile.write(reinterpret_cast<const char*>(&num_users), sizeof(int));
    outfile.write(reinterpret_cast<const char*>(user_embedding.data()),
                  user_embedding.rows() * user_embedding.cols() * sizeof(float));

    // Save item embedding
    const Recommender::MatrixXf& item_embedding = recommender.item_embedding();
    int num_items = item_embedding.rows();
    outfile.write(reinterpret_cast<const char*>(&num_items), sizeof(int));
    outfile.write(reinterpret_cast<const char*>(item_embedding.data()),
                  item_embedding.rows() * item_embedding.cols() * sizeof(float));

    // Save additional parameters
    // You'll need to replace these placeholders with the actual parameters
    // Here, I'm using example values
    float regularization = recommender.regularization();
    float regularization_exp = recommender.regularization_exp();
    float unobserved_weight = recommender.unobserved_weight();
    int block_size = recommender.block_size();
    outfile.write(reinterpret_cast<const char*>(&regularization), sizeof(float));
    outfile.write(reinterpret_cast<const char*>(&regularization_exp), sizeof(float));
    outfile.write(reinterpret_cast<const char*>(&unobserved_weight), sizeof(float));
    outfile.write(reinterpret_cast<const char*>(&block_size), sizeof(int));

    // Close the file
    outfile.close();
}

IALSppRecommender* LoadModel(const std::string& filename) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile.is_open()) {
        throw std::runtime_error("Failed to open file for reading");
    }

    // Read embedding dimensions
    int embedding_dim;
    infile.read(reinterpret_cast<char*>(&embedding_dim), sizeof(int));

    // Read user embedding
    int num_users;
    infile.read(reinterpret_cast<char*>(&num_users), sizeof(int));
    Recommender::MatrixXf user_embedding(num_users, embedding_dim);
    infile.read(reinterpret_cast<char*>(user_embedding.data()),
                user_embedding.rows() * user_embedding.cols() * sizeof(float));

    // Read item embedding
    int num_items;
    infile.read(reinterpret_cast<char*>(&num_items), sizeof(int));
    Recommender::MatrixXf item_embedding(num_items, embedding_dim);
    infile.read(reinterpret_cast<char*>(item_embedding.data()),
                item_embedding.rows() * item_embedding.cols() * sizeof(float));

    // Read additional parameters
    float regularization, regularization_exp, unobserved_weight, stdev;
    int block_size;
    infile.read(reinterpret_cast<char*>(&regularization), sizeof(float));
    infile.read(reinterpret_cast<char*>(&regularization_exp), sizeof(float));
    infile.read(reinterpret_cast<char*>(&unobserved_weight), sizeof(float));
    infile.read(reinterpret_cast<char*>(&block_size), sizeof(int));

    // Close the file
    infile.close();

    IALSppRecommender* recommender;
    recommender = new IALSppRecommender(
      embedding_dim,
      num_users,
      num_items,
      regularization,
      regularization_exp,
      unobserved_weight,
      1,
      block_size);
    
    recommender->SetUserEmbedding(user_embedding);
    recommender->SetItemEmbedding(item_embedding);

    // Create and return the recommender object
    return recommender;
}
